"""
triangulate_person_autodetect_nearest.py
----------------------------------------
最大3台の RealSense を自動検出し、YOLOv5(person) + 深度で各カメラごとに
「最も近い人」1名のみを選択して3D復元 → XY俯瞰プロットします。
USB-A/C どちらでも可。帯域不足時は自動フォールバック。

操作:
  ESC … 終了
  S   … 俯瞰図スクリーンショット保存（captures/ 以下）
"""

from pathlib import Path
import time
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import math

# ────────── 設定 ──────────
MODEL_SPEC = "yolov5n"  # "yolov5n"=軽量 / "yolov5s" など / 自前ptならパス
DEVICE = "cpu"  # CUDA: "cuda:0"
CONF_THRES = 0.35
TARGET_CLASS_ID = 0  # COCO: 0 = person

DEPTH_AVG_KERNEL = 7  # BBox中心パッチ中央値 (奇数)
INFER_SIZE = 640  # YOLO入力解像度
FRAME_SKIP = 0  # 0=毎フレーム推論、1=1枚おき…

# 近さ選別用（不正/外れ値を排除）
MIN_VALID_Z_M = 0.3  # 30cm 未満は誤検出扱い
MAX_VALID_Z_M = 8.0  # 8m 超は誤検出扱い

# 配置と描画
SIDE_M = 0.30  # 正三角形の一辺[m]
CAM_HEIGHT = 0.80  # カメラ高さ[m]
WORLD_SCALE = 1200  # 俯瞰 px/m
CANVAS_SIZE = (900, 900)  # h, w
# ────────────────────────


def load_model():
    print("🔄 Loading YOLOv5 model …")
    if MODEL_SPEC.endswith(".pt"):
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=MODEL_SPEC,
            device=DEVICE,
            trust_repo=True,
        )
    else:
        model = torch.hub.load(
            "ultralytics/yolov5", MODEL_SPEC, device=DEVICE, trust_repo=True
        )
    model.conf = CONF_THRES
    print("✅ Model loaded")
    return model


def tri_vertices(side: float):
    R = math.sqrt(3) * side / 3.0
    A = (0.0, R)
    B = (-side / 2.0, -R / 2.0)
    C = (side / 2.0, -R / 2.0)
    return A, B, C


def yaw_to_center(x, y):
    return math.degrees(math.atan2(-y, -x))


def rvec_from_ypr(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Ryaw = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Rpitch = np.array(
        [[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]]
    )
    Rroll = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return Ryaw @ Rpitch @ Rroll


def world_to_canvas(pt_w: np.ndarray, canvas_h: int, canvas_w: int, scale: float):
    cx, cy = canvas_w // 2, canvas_h // 2
    x_px = int(cx + pt_w[0] * scale)
    y_px = int(cy - pt_w[1] * scale)
    return (x_px, y_px)


def depth_at_bbox(depth_img: np.ndarray, cx: int, cy: int, k: int):
    h, w = depth_img.shape
    k2 = k // 2
    x1, x2 = max(cx - k2, 0), min(cx + k2 + 1, w)
    y1, y2 = max(cy - k2, 0), min(cy + k2 + 1, h)
    patch = depth_img[y1:y2, x1:x2]
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def deproject_color_px_to_cam(color_intr, px, depth_value_m):
    X, Y, Z = rs.rs2_deproject_pixel_to_point(
        color_intr, [float(px[0]), float(px[1])], depth_value_m
    )
    return np.array([X, Y, Z], dtype=np.float32)


class RSDevice:
    """帯域に応じてプロファイルを自動フォールバックして起動"""

    PROFILE_CANDIDATES = [
        (640, 480, 30, 640, 480, 30),
        (640, 480, 15, 640, 480, 15),
        (848, 480, 30, 640, 480, 30),
        (424, 240, 30, 640, 360, 30),
        (424, 240, 30, 640, 360, 15),
        (424, 240, 15, 640, 360, 15),
        (320, 240, 15, 640, 240, 15),
    ]

    def __init__(self, serial: str):
        self.serial = serial
        self.pipe = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.depth_scale = None
        self.color_intr = None
        self.color_size = (640, 480)
        self.frame_count = 0
        self._start_with_candidates()
        self.win_name = f"Camera [{serial}]"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.color_size[0], self.color_size[1])

    def _start_with_candidates(self):
        last_err = None
        for dw, dh, dfps, cw, ch, cfps in self.PROFILE_CANDIDATES:
            try:
                cfg = rs.config()
                cfg.enable_device(self.serial)
                cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
                cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)
                prof = self.pipe.start(cfg)
                self.depth_scale = (
                    prof.get_device().first_depth_sensor().get_depth_scale()
                )
                color_prof = prof.get_stream(rs.stream.color).as_video_stream_profile()
                self.color_intr = color_prof.get_intrinsics()
                self.color_size = (cw, ch)
                print(
                    f"✅ {self.serial} started: depth={dw}x{dh}@{dfps}, color={cw}x{ch}@{cfps}"
                )
                return
            except Exception as e:
                last_err = e
                try:
                    self.pipe.stop()
                except Exception:
                    pass
                continue
        raise RuntimeError(
            f"❌ {self.serial} start failed for all candidates: {last_err}"
        )

    def grab(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return None
        depth_img = np.asanyarray(depth.get_data())
        color_img = np.asanyarray(color.get_data())
        return depth, depth_img, color_img

    def stop(self):
        try:
            self.pipe.stop()
        except Exception:
            pass
        cv2.destroyWindow(self.win_name)


def pick_serials_auto(max_n=3) -> List[str]:
    ctx = rs.context()
    connected = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
    if not connected:
        raise RuntimeError("RealSense が見つかりません。USB接続を確認してください。")
    picked = connected[:max_n]
    if len(picked) < max_n:
        print(f"⚠ {len(picked)}台のみ接続。{max_n}台未満のため、統合は平均で行います。")
    print("Connected serials:", picked)
    return picked


def build_cam_poses(serials: List[str], side_m: float, cam_height: float):
    A, B, C = tri_vertices(side_m)
    verts = [A, B, C]
    poses = {}
    for i, sid in enumerate(serials):
        x, y = verts[i]
        poses[sid] = {
            "t": (x, y, cam_height),
            "yaw": yaw_to_center(x, y),
            "pitch": -10.0,
            "roll": 0.0,
        }
    return poses


def draw_triangle_and_cams(canvas, cam_poses: Dict[str, dict]):
    H, W, _ = canvas.shape
    if len(cam_poses) == 3:
        keys = list(cam_poses.keys())
        for i in range(3):
            p1w = np.array([cam_poses[keys[i]]["t"][0], cam_poses[keys[i]]["t"][1], 0])
            p2w = np.array(
                [
                    cam_poses[keys[(i + 1) % 3]]["t"][0],
                    cam_poses[keys[(i + 1) % 3]]["t"][1],
                    0,
                ]
            )
            cv2.line(
                canvas,
                world_to_canvas(p1w, H, W, WORLD_SCALE),
                world_to_canvas(p2w, H, W, WORLD_SCALE),
                (200, 200, 200),
                2,
            )
    for sid, pose in cam_poses.items():
        x, y, _ = pose["t"]
        yaw = np.radians(pose["yaw"])
        p = world_to_canvas(np.array([x, y, 0]), H, W, WORLD_SCALE)
        tip = world_to_canvas(
            np.array([x + 0.07 * np.cos(yaw), y + 0.07 * np.sin(yaw), 0]),
            H,
            W,
            WORLD_SCALE,
        )
        cv2.circle(canvas, p, 5, (255, 255, 255), -1)
        cv2.arrowedLine(canvas, p, tip, (255, 255, 255), 2, tipLength=0.4)
        cv2.putText(
            canvas,
            sid[-4:],
            (p[0] + 6, p[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    save_dir = Path("captures")
    save_dir.mkdir(exist_ok=True)
    save_idx = 0

    model = load_model()
    serials = pick_serials_auto(3)
    cam_poses = build_cam_poses(serials, SIDE_M, CAM_HEIGHT)
    devices = {s: RSDevice(s) for s in serials}

    extrinsics = {}
    for sid, pose in cam_poses.items():
        Rcw = rvec_from_ypr(pose["yaw"], pose["pitch"], pose["roll"])
        tcw = np.array(pose["t"], dtype=np.float32)
        extrinsics[sid] = (Rcw, tcw)

    H, W = CANVAS_SIZE
    win_top = "Top-Down (XY)"
    cv2.namedWindow(win_top, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_top, W, H)

    try:
        while True:
            world_points = []

            for sid, dev in devices.items():
                grabbed = dev.grab()
                if grabbed is None:
                    continue
                depth_frame, depth_img, color_img = grabbed

                do_infer = (FRAME_SKIP == 0) or (
                    dev.frame_count % (FRAME_SKIP + 1) == 0
                )
                dev.frame_count += 1
                if not do_infer:
                    cv2.imshow(dev.win_name, color_img)
                    continue

                # YOLO推論
                results = model(color_img, size=INFER_SIZE)
                det = results.xyxy[0]  # [x1,y1,x2,y2,conf,cls]

                # —— ここがポイント：最も「近い」person だけを採用 ——
                best = None
                best_depth_m = 1e9
                for *xyxy, conf, cls in det:
                    if int(cls) != TARGET_CLASS_ID:
                        continue
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    depth_raw = depth_at_bbox(depth_img, cx, cy, DEPTH_AVG_KERNEL)
                    if depth_raw is None:
                        continue
                    depth_m = depth_raw * dev.depth_scale
                    if not (MIN_VALID_Z_M <= depth_m <= MAX_VALID_Z_M):
                        continue

                    # 深度最小を優先。同深度なら conf が高い方
                    if (depth_m < best_depth_m) or (
                        abs(depth_m - best_depth_m) < 1e-3
                        and float(conf) > (best[6] if best else -1)
                    ):
                        best_depth_m = depth_m
                        best = (x1, y1, x2, y2, cx, cy, float(conf))

                if best is None:
                    cv2.putText(
                        color_img,
                        "no valid person",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(dev.win_name, color_img)
                    continue

                x1, y1, x2, y2, cx, cy, conf = best
                # deproject
                p_cam = deproject_color_px_to_cam(
                    dev.color_intr, (cx, cy), best_depth_m
                )
                Rcw, tcw = extrinsics[sid]
                p_w = Rcw @ p_cam + tcw
                world_points.append(p_w)

                # ビュー描画（最も近い人のみ）
                cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_img, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(
                    color_img,
                    f"{best_depth_m*1000:.0f}mm  conf={conf:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(dev.win_name, color_img)

            # 俯瞰図
            canvas = np.zeros((H, W, 3), np.uint8)
            draw_triangle_and_cams(canvas, cam_poses)

            if world_points:
                pts_xy = np.array(
                    [[p[0], p[1]] for p in world_points], dtype=np.float32
                )
                xy_mean = pts_xy.mean(axis=0)
                pt_mean_w = np.array([xy_mean[0], xy_mean[1], 0.0], dtype=np.float32)

                for p in world_points:
                    pp = world_to_canvas(np.array([p[0], p[1], 0]), H, W, WORLD_SCALE)
                    cv2.circle(canvas, pp, 6, (0, 128, 255), -1)
                pm = world_to_canvas(pt_mean_w, H, W, WORLD_SCALE)
                cv2.circle(canvas, pm, 10, (0, 255, 0), -1)
                cv2.putText(
                    canvas,
                    f"Est XY = ({xy_mean[0]:.3f} m, {xy_mean[1]:.3f} m)",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(win_top, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord("s"), ord("S")):
                out = save_dir / f"topdown_{save_idx:03d}.png"
                cv2.imwrite(str(out), canvas)
                print(f"💾 Saved {out}")
                save_idx += 1

    finally:
        for dev in devices.values():
            dev.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
