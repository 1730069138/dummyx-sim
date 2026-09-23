"""RGB-D localization of the known upright, static 4 x 4 x 16 cm pillar.

The table top is at world z=0.20 m. The pillar is assumed axis-aligned.
Only camera calibration and RGB-D enter localization; obstacle body/geom
state is never read here. A visible, sufficiently complete top face is required.
DINO and camera helpers follow deploy_screwdriver_client_vlsa.py, kept local
so the APF entry does not acquire VLSA's cvxpy/VLM dependencies.
"""
import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

TABLE_Z = 0.20
HALF_WIDTH = 0.02
HALF_HEIGHT = 0.08
GEOMETRY_MARGIN = 0.01  # Initial 1 cm allowance; validate for each experiment.


class GroundingDINOWrapper:
    def __init__(self, config_path, checkpoint_path, device):
        try:
            from PIL import Image
            import groundingdino.datasets.transforms as T
            from groundingdino.util.inference import load_model, predict
        except ImportError as exc:
            raise ImportError(
                "Perception mode requires GroundingDINO, torch, torchvision and pillow."
            ) from exc

        self.Image = Image
        self.predict_fn = predict
        self.device = device
        self.model = load_model(config_path, checkpoint_path, device=device)

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def detect_all_boxes(self, image_rgb, caption, box_threshold, text_threshold):
        """Return thresholded candidates sorted by descending confidence."""
        image_pil = self.Image.fromarray(
            np.asarray(image_rgb, dtype=np.uint8), mode="RGB"
        )
        image_tensor, _ = self.transform(image_pil, None)

        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        if boxes is None or len(boxes) == 0:
            return []

        boxes_np = boxes.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        h, w = image_rgb.shape[:2]
        detections = []

        for idx, (cx, cy, bw, bh) in enumerate(boxes_np):
            x1 = int(np.floor((cx - bw / 2.0) * w))
            y1 = int(np.floor((cy - bh / 2.0) * h))
            x2 = int(np.ceil((cx + bw / 2.0) * w))
            y2 = int(np.ceil((cy + bh / 2.0) * h))

            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, x1 + 1, w))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, y1 + 1, h))

            phrase = phrases[idx] if phrases and idx < len(phrases) else caption
            detections.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "confidence": float(logits_np[idx]),
                    "phrase": str(phrase),
                }
            )

        detections.sort(key=lambda det: det["confidence"], reverse=True)
        return detections


def render_rgb_depth(renderer, data, camera_name):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    return rgb, depth


def camera_intrinsics_from_fovy(model, camera_id, height, width):
    fovy = np.deg2rad(float(model.cam_fovy[camera_id]))
    fy = 0.5 * height / np.tan(0.5 * fovy)
    fx = fy  # MuJoCo assumes square pixels for this camera model.
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return fx, fy, cx, cy


def bbox_depth_to_world_points(model, data, camera_name, depth, bbox_xyxy):
    """
    MuJoCo fixed cameras look along local -Z, with +X right and +Y up.
    Renderer depth is metric distance to the camera plane (optical-axis depth).
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise RuntimeError(f"Camera not found: {camera_name}")

    h, w = depth.shape
    fx, fy, cx, cy = camera_intrinsics_from_fovy(model, cam_id, h, w)

    x1, y1, x2, y2 = bbox_xyxy
    uu, vv = np.meshgrid(
        np.arange(x1, x2, dtype=np.float64),
        np.arange(y1, y2, dtype=np.float64),
    )
    zz = depth[y1:y2, x1:x2].astype(np.float64)

    valid = np.isfinite(zz) & (zz > 1e-5)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    u = uu[valid]
    v = vv[valid]
    z_depth = zz[valid]

    x_cam = (u - cx) * z_depth / fx
    y_cam = -(v - cy) * z_depth / fy
    z_cam = -z_depth

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    p_cam = data.cam_xpos[cam_id].copy()
    r_cam = data.cam_xmat[cam_id].reshape(3, 3).copy()
    points_world = points_cam @ r_cam.T + p_cam
    return points_world


def fit_known_pillar(points):
    """Estimate XY from a complete top face, rejecting unsupported geometry.

    points is an H x W x 3 organized cloud cropped to a detection box. NaNs
    mark invalid depth. Top-face components separate disconnected background.
    This is deliberately a known-shape localizer, not arbitrary shape fitting.
    """
    top_z = TABLE_Z + 2 * HALF_HEIGHT
    top = np.isfinite(points).all(axis=2) & (np.abs(points[:, :, 2] - top_z) < 0.004)
    count, labels = cv2.connectedComponents(top.astype(np.uint8), connectivity=8)
    candidates = []
    for label in range(1, count):
        face = points[labels == label]
        if len(face) < 12:
            continue
        lo, hi = np.min(face[:, :2], axis=0), np.max(face[:, :2], axis=0)
        extent = hi - lo
        # Refuse a narrow visible sliver instead of inventing a hidden center.
        if np.any(extent < 0.030) or np.any(extent > 0.047):
            continue
        xy = (lo + hi) / 2
        flat = points.reshape(-1, 3)
        near = flat[np.isfinite(flat).all(axis=1)
                    & (np.abs(flat[:, 0] - xy[0]) < HALF_WIDTH + 0.004)
                    & (np.abs(flat[:, 1] - xy[1]) < HALF_WIDTH + 0.004)
                    & (flat[:, 2] > TABLE_Z + 0.01)
                    & (flat[:, 2] < top_z + 0.004)]
        if len(near) < 30 or np.ptp(near[:, 2]) < 0.09:
            continue
        center = np.array([xy[0], xy[1], TABLE_Z + HALF_HEIGHT])
        capsule = {
            "p1": center - np.array([0.0, 0.0, HALF_HEIGHT]),
            "p2": center + np.array([0.0, 0.0, HALF_HEIGHT]),
            "r": float(np.sqrt(2) * HALF_WIDTH + GEOMETRY_MARGIN),
        }
        candidates.append((capsule, near))
    if len(candidates) != 1:
        raise ValueError(f"Expected one complete upright pillar top, found {len(candidates)}")
    return candidates[0]


def localize_pillar(model, data, renderer, detector, caption, box_threshold, text_threshold, output_dir):
    """Capture once while physics is paused; save evidence even on detection failure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rgb, depth = render_rgb_depth(renderer, data, "rear_cam")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rear_cam")
    np.savez_compressed(output_dir / "capture.npz", depth=depth,
                        intrinsics=camera_intrinsics_from_fovy(model, cam_id, *depth.shape),
                        camera_position=data.cam_xpos[cam_id].copy(),
                        camera_rotation=data.cam_xmat[cam_id].reshape(3, 3).copy(),
                        simulation_time=data.time)
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / "rgb.png"), overlay)
    metadata = {"status": "failed", "caption": caption, "camera": "rear_cam",
                "box_threshold": box_threshold, "text_threshold": text_threshold,
                "simulation_time": float(data.time), "candidates": [],
                "assumption": "static axis-aligned 0.04x0.04x0.16m pillar on z=0.20m table"}
    try:
        detections = detector.detect_all_boxes(rgb, caption, box_threshold, text_threshold)
        valid = []
        for detection in detections:
            x1, y1, x2, y2 = detection["xyxy"]
            entry = dict(detection)
            # Preserve pixel topology, including invalid depth, for components.
            roi_depth = depth[y1:y2, x1:x2]
            mask = np.isfinite(roi_depth) & (roi_depth > 1e-5)
            cloud = np.full((*roi_depth.shape, 3), np.nan)
            cloud[mask] = bbox_depth_to_world_points(model, data, "rear_cam", depth, detection["xyxy"])
            try:
                capsule, points = fit_known_pillar(cloud)
                center = (capsule["p1"] + capsule["p2"]) / 2
                # Duplicate DINO boxes may describe the same pillar.
                if not any(np.linalg.norm(center - item[2]) < 0.01 for item in valid):
                    valid.append((capsule, points, center))
                entry["geometry_status"] = "valid"
            except ValueError as exc:
                entry["geometry_status"] = str(exc)
            accepted = entry["geometry_status"] == "valid"
            color = (0, 255, 0) if accepted else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.putText(overlay, f"{'valid' if accepted else 'reject'} {detection['confidence']:.2f}",
                        (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            metadata["candidates"].append(entry)
        if len(valid) != 1:
            raise ValueError(f"Expected one localized pillar, found {len(valid)} from {len(detections)} detections")
        capsule, points, center = valid[0]
        metadata.update(status="ok", center=center.tolist(),
                        capsule={key: value.tolist() if isinstance(value, np.ndarray) else value
                                 for key, value in capsule.items()})
        np.save(output_dir / "points.npy", points)
        return capsule, metadata
    except Exception as exc:
        metadata["reason"] = str(exc)
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        cv2.imwrite(str(output_dir / "detections.png"), overlay)
        (output_dir / "perception.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
