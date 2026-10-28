"""Small bounded projection for a discrete kinematic reference constraint."""
import numpy as np


def bounded_nullspace_projection(row, angular_jacobian, required, lower, upper):
    """Minimize correction norm in angular nullspace subject to box and normal.

    All inputs describe one linearized reference, not physical collision safety.
    Return correction, feasibility and solver diagnostics; never relax constraints.
    """
    row, angular, lower, upper = (np.asarray(x, dtype=float) for x in
                                  (row, angular_jacobian, lower, upper))
    if (row.shape != (6,) or angular.shape != (3, 6) or lower.shape != (6,)
            or upper.shape != (6,) or not all(np.isfinite(x).all() for x in
                                             (row, angular, lower, upper)) or not np.isfinite(required)):
        raise ValueError('Expected finite six-joint projection inputs')
    if np.any(lower > upper):
        return np.zeros(6), False, dict(method='empty_bounds', iterations=0)
    _, singular, vectors = np.linalg.svd(angular, full_matrices=True)
    basis = vectors[int(np.sum(singular > 1e-9)):].T
    normal = row @ basis
    denominator = float(normal @ normal)
    if denominator < 1e-12:
        return np.zeros(6), False, dict(method='zero_normal_authority', iterations=0)
    y = normal * (max(float(required), 0.) / denominator)
    def feasible(x):
        return bool(np.isfinite(x).all() and np.all(x >= lower-1e-9)
                    and np.all(x <= upper+1e-9) and row @ x >= required-1e-9
                    and np.max(np.abs(angular @ x)) <= 1e-9)
    correction = basis @ y
    if feasible(correction):
        return correction, True, dict(method='analytic', iterations=0)
    # Solve in the angular nullspace so componentwise joint clipping cannot
    # destroy either the normal constraint or the orientation constraint.
    from scipy.optimize import minimize
    matrix = np.vstack((basis, -basis, normal[None, :]))
    bound = np.r_[lower, -upper, required]
    result = minimize(lambda x: .5 * float(x @ x), y, jac=lambda x: x,
                      method='SLSQP', constraints=[dict(type='ineq',
                        fun=lambda x: matrix @ x - bound, jac=lambda x: matrix)],
                      options=dict(maxiter=40, ftol=1e-12))
    correction = basis @ result.x
    valid = feasible(correction)
    return correction, valid, dict(method='bounded_slsqp', iterations=int(result.nit),
        optimizer_success=bool(result.success), message=str(result.message),
        normal_residual_m=float(row @ correction-required))
