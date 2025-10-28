"""
drone_distance.py
-----------------
YOLOv5 でドローンを検知し、RealSense D435/D435i の深度情報を使って
距離（mm）をリアルタイム表示します。

操作:
  ESC … 終了
  S   … スクリーンショット保存（captures/ 以下）
"""

from pathlib import Path
import os

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

# ────────── ユーザ設定 ──────────
MODEL_PATH = "drone_yolov5nu.pt"  # 学習済みウェイト
CONF_THRES = 0.30  # 検出信頼度しきい値
BOX_COLOR = (0, 255, 0)  # BGR
LABEL_COLOR = (0, 255, 255)
DEPTH_AVG_KERNEL = 7  # k×k の中央値 (奇数)
DEVICE = "cpu"  # CUDA 使用なら "cuda:0"
# ────────────────────────────────

# RealSense 初期化
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipe = rs.pipeline()
align = rs.align(rs.stream.color)
prof = pipe.start(cfg)
depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()

# YOLOv5 モデル読み込み
print("🔄 Loading YOLOv5 model …")
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    device=DEVICE,
    trust_repo=True,  # GitHub 上のスクリプトを信頼
)
model.conf = CONF_THRES  # 既定閾値を設定
print("✅ Model loaded")


# 深度取得ヘルパ
def depth_at_bbox(depth_img: np.ndarray, cx: int, cy: int, k: int = DEPTH_AVG_KERNEL):
    """BBox 中央 k×k パッチの深度中央値 [m] を返す（invalid=0 は除外）"""
    h, w = depth_img.shape
    k2 = k // 2
    x1, x2 = max(cx - k2, 0), min(cx + k2 + 1, w)
    y1, y2 = max(cy - k2, 0), min(cy + k2 + 1, h)
    patch = depth_img[y1:y2, x1:x2]
    patch = patch[patch > 0]
    return float(np.median(patch)) * depth_scale if patch.size else None


# メインループ
print("▶ ESC = quit,  S = save frame")
save_dir = Path("captures")
save_dir.mkdir(exist_ok=True)
save_idx = 0

try:
    while True:
        frames = align.process(pipe.wait_for_frames())
        depth_img = np.asanyarray(frames.get_depth_frame().get_data())
        color_img = np.asanyarray(frames.get_color_frame().get_data())

        # 推論 (BGR→RGB はライブラリで自動変換)
        results = model(color_img, size=640)

        # 1枚目の推論結果を処理
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) != 0:  # 0 = "drone" を想定
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            dist_m = depth_at_bbox(depth_img, cx, cy)
            if dist_m is None:
                continue

            cv2.rectangle(color_img, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.circle(color_img, (cx, cy), 4, BOX_COLOR, -1)
            cv2.putText(
                color_img,
                f"{dist_m*1000:.0f} mm",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                LABEL_COLOR,
                2,
            )

        cv2.imshow("Drone distance", color_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord("s"), ord("S")):
            cv2.imwrite(str(save_dir / f"frame_{save_idx:03d}.png"), color_img)
            print(f"💾 Saved captures/frame_{save_idx:03d}.png")
            save_idx += 1

finally:
    pipe.stop()
    cv2.destroyAllWindows()
