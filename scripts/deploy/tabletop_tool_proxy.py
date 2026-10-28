"""Conservative capsules fitted to each *physical* convex tool collision part.

Uses tool/robot geometry only. No pillar, bin, or scene obstacle truth is read.
Each capsule contains every vertex of its convex mesh and hence the whole part.
"""
import numpy as np

def fit_body_component_capsules(arm,data,body):
    m=arm.model;R=data.xmat[body].reshape(3,3);origin=data.xpos[body];caps=[]
    seen=set()
    for geom in np.flatnonzero(m.geom_bodyid==body):
        if not m.geom_contype[geom]:continue
        mesh=int(m.geom_dataid[geom])
        if mesh<0:raise ValueError('Component proxy requires convex mesh geometry')
        key=(mesh,tuple(m.geom_pos[geom]),tuple(m.geom_quat[geom]))
        if key in seen:continue
        seen.add(key)
        start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
        vertices=(m.mesh_vert[start:start+count]@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]-origin)@R
        center=vertices.mean(0);_,vectors=np.linalg.eigh((vertices-center).T@(vertices-center));axis=vectors[:,-1]
        if axis[np.argmax(np.abs(axis))]<0:axis=-axis
        projection=(vertices-center)@axis;radial=vertices-center-projection[:,None]*axis
        radius=float(np.linalg.norm(radial,axis=1).max())+1e-6
        caps.append(dict(p1=center+projection.min()*axis,p2=center+projection.max()*axis,r=radius,geom_id=int(geom),body_id=int(body),mesh_name=m.mesh(mesh).name))
    if not caps:raise ValueError('No tool collision components')
    return caps

def world_tool_capsules(data,tool_body,local_capsules):
    R=data.xmat[tool_body].reshape(3,3);origin=data.xpos[tool_body]
    return [dict(p1=origin+R@cap['p1'],p2=origin+R@cap['p2'],r=cap['r']) for cap in local_capsules]


def fit_tool_component_capsules(arm,data):
    return fit_body_component_capsules(arm,data,arm.target_body)

def world_body_capsules(data,local_capsules):
    return [dict(p1=data.xpos[c['body_id']]+data.xmat[c['body_id']].reshape(3,3)@c['p1'],p2=data.xpos[c['body_id']]+data.xmat[c['body_id']].reshape(3,3)@c['p2'],r=c['r']) for c in local_capsules]

def batch_segment_distances(p1,p2,p3,p4):
    """Vectorized equivalent of the deployment segment distance (including degeneracy)."""
    p1=np.asarray(p1,float);p2=np.asarray(p2,float);p3=np.asarray(p3,float);p4=np.asarray(p4,float)
    u=p2-p1;v=p4-p3;r=p1-p3;a=np.sum(u*u,axis=1);e=float(v@v);f=r@v;c=np.sum(u*r,axis=1);b=u@v
    regular=a>1e-6;s=np.zeros(len(a));t=np.zeros(len(a))
    if e<=1e-6:
        s[regular]=np.clip(-c[regular]/a[regular],0,1)
    else:
        den=a*e-b*b;mask=regular&(den!=0);s[mask]=np.clip((b[mask]*f[mask]-c[mask]*e)/den[mask],0,1);t=(b*s+f)/e
        low=regular&(t<0);high=regular&(t>1);s[low]=np.clip(-c[low]/a[low],0,1);s[high]=np.clip((b[high]-c[high])/a[high],0,1);t=np.clip(t,0,1)
    closest=p1+u*s[:,None];other=p3+v*t[:,None];delta=closest-other;dist=np.linalg.norm(delta,axis=1);normal=np.zeros_like(delta);normal[:,0]=1.;good=dist>1e-6;normal[good]=delta[good]/dist[good,None]
    return dist,normal,closest,other

def batch_capsule_clearances(capsules,obstacle):
    distances,normals,points,_=batch_segment_distances([c['p1'] for c in capsules],[c['p2'] for c in capsules],obstacle['p1'],obstacle['p2'])
    return distances-np.array([c['r'] for c in capsules])-obstacle['r'],normals,points
