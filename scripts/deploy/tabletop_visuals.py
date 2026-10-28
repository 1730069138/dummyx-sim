"""Display-only DINO target markers shared by live viewer and state replay."""
import numpy as np
import mujoco


def draw_box_targets(scene, opening_center, attraction_point, *, enabled, active=False, origin=None):
    def sphere(position, color, label, radius):
        if scene.ngeom >= len(scene.geoms):return
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([radius]*3),
                          np.asarray(position,dtype=float),np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
        geom.label=label;scene.ngeom+=1
    sphere(opening_center,[0.,1.,1.,.9],'DINO box rim',.007)
    sphere(attraction_point,[1.,0.,1.,1.] if enabled else [.5,.5,.5,.9],
           'Attraction target ON' if enabled else 'Attraction target OFF',.011)
    if enabled and active and origin is not None and scene.ngeom<len(scene.geoms):
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1.,0.,1.,.8]))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.003,np.asarray(origin),np.asarray(attraction_point))
        scene.ngeom+=1
