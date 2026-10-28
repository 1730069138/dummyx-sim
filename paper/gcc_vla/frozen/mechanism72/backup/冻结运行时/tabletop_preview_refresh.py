"""Bounded policy-refresh decision from the existing execution forecast."""
def needs_policy_refresh(debug,chunk_index,dt,slack_m_s,minimum_steps=10):
    # index refers to the action being executed; refresh happens on the next loop.
    if chunk_index+1<minimum_steps or not debug['pickup_gate'] or debug['prediction_step']<=0:
        return False
    risk=debug['predicted_clearance_m']+slack_m_s*dt*(debug['prediction_step']-1)
    return bool(risk<debug['target_clearance_m'])
