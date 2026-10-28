"""DINO-selected RGB-D geometry for the tabletop scene, with no object-pose lookup.

Assumptions: one static upright red pillar and one static axis-aligned open box,
visible top/rim and interior bottom. Dimensions and locations come from depth.
Camera calibration is simulator-provided. Failure never falls back to scene poses.
"""
from pathlib import Path
import json
import time
import cv2
import numpy as np
from groundingdino_obstacle import (GroundingDINOWrapper, render_rgb_depth,
    bbox_depth_to_world_points, camera_intrinsics_from_fovy)


def horizontal_levels(points, tolerance=.002, minimum=30):
    points=np.asarray(points); points=points[np.isfinite(points).all(axis=1)]
    if len(points)<minimum: raise ValueError('Insufficient finite depth points')
    bins=np.floor(points[:,2]/tolerance).astype(np.int64)
    levels,counts=np.unique(bins,return_counts=True)
    ordered=np.argsort(counts)[::-1]; result=[]
    for index in ordered:
        if counts[index]<minimum: continue
        z=float(np.median(points[bins==levels[index],2]))
        if any(abs(z-old)<2*tolerance for old in result):continue
        result.append(z)
    return result


def horizontal_cloud(model,data,camera_name,depth):
    h,w=depth.shape
    valid=np.isfinite(depth)&(depth>1e-5)
    cloud=np.full((h,w,3),np.nan)
    cloud[valid]=bbox_depth_to_world_points(model,data,camera_name,depth,(0,0,w,h))
    dy,dx=np.gradient(cloud,axis=(0,1))
    normal=np.cross(dx,dy);length=np.linalg.norm(normal,axis=2)
    horizontal=(length>1e-12)&(np.abs(normal[:,:,2])>.995*length)
    return cloud[horizontal&valid]


def fit_support(world_points, xy, below, required_half):
    near=world_points[(np.linalg.norm(world_points[:,:2]-xy,axis=1)<.45)&(world_points[:,2]<below-.008)]
    for z in sorted(horizontal_levels(near),reverse=True):
        plane=near[np.abs(near[:,2]-z)<.002]
        lo,hi=np.percentile(plane[:,:2],[1,99],axis=0)
        # A surrounding broad support plane, not an object-sized top face.
        if np.all(hi-lo>.15) and np.all(xy-required_half>=lo) and np.all(xy+required_half<=hi):
            return dict(center=np.r_[(lo+hi)/2,z],rotation=np.eye(3),
                        half_size=np.r_[(hi-lo)/2,0.],top=z)
    raise ValueError('No visible surrounding support plane')


def fit_pillar(points, full_cloud):
    # DINO supplies identity/ROI; red chroma refinement is applied by the caller.
    if len(points)<30:raise ValueError('Insufficient pillar depth')
    lo,hi=np.percentile(points,[1,99],axis=0)
    xy=(lo[:2]+hi[:2])/2
    support=fit_support(full_cloud,xy,hi[2],(hi[:2]-lo[:2])/2+.02)
    height=hi[2]-support['top'];extent=hi[:2]-lo[:2]
    if height<=.02 or np.any(extent<=.005) or np.any(extent>=height*2):
        raise ValueError('Pillar depth does not support an upright volume')
    if np.ptp(points[:,2])<.4*height:raise ValueError('Insufficient pillar side coverage')
    radius=float(np.linalg.norm(extent/2)+.01)
    return dict(p1=np.r_[xy,support['top']],p2=np.r_[xy,hi[2]],r=radius),support


def fit_box(points, full_cloud):
    points=points[np.isfinite(points).all(axis=1)]
    for z in sorted(horizontal_levels(points),reverse=True):
        rim=points[np.abs(points[:,2]-z)<.002]
        lo,hi=np.percentile(rim[:,:2],[.5,99.5],axis=0);span=hi-lo
        if np.any(span<.04):continue
        normalized=(rim[:,:2]-lo)/span
        # Four measured rim edges and an open center; reject a solid tabletop.
        edges=[normalized[:,0]<.12,normalized[:,0]>.88,normalized[:,1]<.12,normalized[:,1]>.88]
        if any(np.count_nonzero(edge)<5 for edge in edges):continue
        center_points=np.all((normalized>.22)&(normalized<.78),axis=1)
        if np.mean(center_points)>.12:continue
        xy=(lo+hi)/2
        try: support=fit_support(full_cloud,xy,z,span/2+.02)
        except ValueError: continue
        interior=points[np.all((points[:,:2]>lo+span*.12)&(points[:,:2]<hi-span*.12),axis=1)&(points[:,2]<z-.006)&(points[:,2]>support['top']+.002)]
        if len(interior)<30:continue
        levels=horizontal_levels(interior)
        if not levels:continue
        bottom=levels[0]
        if not support['top']<bottom<z-.006:continue
        # Conservative observed interior; no model dimensions or body transforms.
        half=span*.38
        return dict(opening_center=np.r_[xy,z],placement_goal=np.r_[xy,z+.05],
                    attraction_point=np.r_[xy,z+.05],interior_half_size=half,
                    bottom_top=bottom,rim_top=z,support=support),rim
    raise ValueError('No complete visible box rim and interior bottom')


def localize_scene(model,data,renderer,detector,output_dir,*,need_pillar=True,
                   camera_name='rear_cam',pillar_text='red pillar.',box_text='open brown box.',
                   box_threshold=.35,text_threshold=.25):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); metadata=dict(status='failed',camera=camera_name,
        source='groundingdino_rgbd',assumptions=__doc__,candidates=[],need_pillar=need_pillar)
    try:
        rgb,depth=render_rgb_depth(renderer,data,camera_name)
        camera_id=model.camera(camera_name).id
        full=horizontal_cloud(model,data,camera_name,depth)
        np.savez_compressed(output/'capture.npz',depth=depth,
            intrinsics=camera_intrinsics_from_fovy(model,camera_id,*depth.shape),
            camera_position=data.cam_xpos[camera_id].copy(),camera_rotation=data.cam_xmat[camera_id].reshape(3,3).copy(),simulation_time=data.time)
        overlay=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR); cv2.imwrite(str(output/'rgb.png'),overlay)
        result={}
        for kind,caption in ([('pillar',pillar_text)] if need_pillar else [])+[('box',box_text)]:
            valid=[]
            detections=detector.detect_all_boxes(rgb,caption,box_threshold,text_threshold)
            for detection in detections:
                entry=dict(kind=kind,**detection)
                try:
                    x1,y1,x2,y2=map(int,detection['xyxy'])
                    if not (0<=x1<x2<=depth.shape[1] and 0<=y1<y2<=depth.shape[0]):raise ValueError('Invalid detector ROI')
                    cloud=bbox_depth_to_world_points(model,data,camera_name,depth,(x1,y1,x2,y2))
                    if kind=='pillar':
                        color=rgb[y1:y2,x1:x2].reshape(-1,3).astype(float)
                        finite=(np.isfinite(depth[y1:y2,x1:x2])&(depth[y1:y2,x1:x2]>1e-5)).reshape(-1)
                        color=color[finite];red=(color[:,0]>1.4*color[:,1])&(color[:,0]>1.4*color[:,2])&(color[:,0]>40)
                        geometry,support=fit_pillar(cloud[red],full)
                        geometry['support']=support
                        key=(geometry['p1']+geometry['p2'])/2
                    else:
                        geometry,rim=fit_box(cloud,full);key=geometry['opening_center']
                    if not any(np.linalg.norm(key-item[0])<.01 for item in valid):valid.append((key,geometry))
                    entry['geometry_status']='valid'
                except (ValueError,IndexError) as exc:
                    entry['geometry_status']=str(exc)
                metadata['candidates'].append(entry)
                x1,y1,x2,y2=map(int,detection['xyxy'])
                cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0) if entry['geometry_status']=='valid' else (0,0,255),1)
            if len(valid)!=1:raise ValueError(f'{kind}: expected one depth-qualified detection, got {len(valid)}')
            result[kind]=valid[0][1]
        metadata.update(status='ok',geometry=result)
        return result,metadata
    except Exception as exc:
        metadata['reason']=str(exc)
        raise
    finally:
        metadata['elapsed_seconds']=time.perf_counter()-started
        if 'overlay' in locals():cv2.imwrite(str(output/'detections.png'),overlay)
        def encode(value):
            if isinstance(value,np.ndarray):return value.tolist()
            if isinstance(value,np.generic):return value.item()
            raise TypeError(type(value).__name__)
        (output/'perception.json').write_text(json.dumps(metadata,default=encode,ensure_ascii=False,indent=2)+'\n')
