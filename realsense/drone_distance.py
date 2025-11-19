import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
import pyrealsense2 as rs
import time
import sys
import requests


# =========================
# 設定
# =========================
MODEL_PATH = "yolo11n_drone.pt"  # 学習済みドローン検出モデル
CONF_THRES = 0.25
IOU_THRES = 0.45
DETECT_INTERVAL = 10  # YOLO再検出の間隔 (大→軽い/遅延減, 小→頑丈/重い)
DEPTH_KERNEL = 5      # 中心周りの深度を平均するサイズ（奇数）

# StampFly コントローラ側 HTTP 設定
CTRL_HOST = "192.168.4.1"
CTRL_PORT = 80
RANGE_ENDPOINT = f"http://{CTRL_HOST}:{CTRL_PORT}/range"

SEND_INTERVAL_SEC = 0.05  # 何秒ごとに距離を送るか（20Hz）

# OpenCV ちょい高速化
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(0)
except Exception:
    pass


# =========================
# ユーティリティ
# =========================
@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


def ema(prev, new, alpha=0.35):
    """2D点の指数移動平均"""
    if prev is None:
        return new
    return (
        int(prev[0] * (1 - alpha) + new[0] * alpha),
        int(prev[1] * (1 - alpha) + new[1] * alpha),
    )


def create_tracker():
    """環境差に強いトラッカー生成（CSRT優先→CSRT(非legacy)→KCF→MOSSE）"""
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        return cv2.legacy.TrackerMOSSE_create()
    raise RuntimeError(
        "OpenCVにCSRT/KCF/MOSSEトラッカーが見つかりません。"
        "opencv-contrib-python を確認してください。"
    )


# 距離送信ヘルパ
_last_send_time = 0.0


def send_range_to_drone(dist_m: float):
    """RealSenseで計算した距離[m]を StampFly コントローラへ送信"""
    global _last_send_time

    now = time.time()
    if now - _last_send_time < SEND_INTERVAL_SEC:
        return  # 送りすぎ防止

    _last_send_time = now

    try:
        requests.get(
            RANGE_ENDPOINT,
            params={"m": f"{dist_m:.3f}"},
            timeout=0.05,
        )
    except Exception as e:
        # 通信が切れていても計測自体は続けたいので、警告だけにする
        print(f"[WARN] send_range_to_drone failed: {e}", file=sys.stderr)


# =========================
# RealSense 初期化
# =========================
pipeline = rs.pipeline()
config = rs.config()

# depth / color を 30fps で取得（解像度はデフォルト）
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

try:
    profile = pipeline.start(config)
except Exception as e:
    print("[ERROR] RealSense pipeline.start で失敗しました:", e)
    raise

align_to = rs.stream.color
align = rs.align(align_to)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # 通常 0.001[m]
print(f"[INFO] depth_scale: {depth_scale} m/unit")


# =========================
# モデル 初期化
# =========================
model = YOLO(MODEL_PATH)
try:
    model.fuse()
except Exception:
    pass


# =========================
# メインループ
# =========================
tracker = None
track_ok = False
last_detect_box = None  # BBox
smoothed_c = None       # (x, y)
frame_idx = 0
dist_smooth = None      # 距離のEMA用

try:
    while True:
        frames = pipeline.poll_for_frames()
        if not frames:
            continue

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        h, w = frame.shape[:2]

        # --- 一定間隔で YOLO 再検出 ---
        do_detect = (frame_idx % DETECT_INTERVAL == 0) or (not track_ok)
        det_box = None

        if do_detect:
            results = model.predict(
                source=frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False
            )[0]

            candidates = []
            if results.boxes is not None and len(results.boxes) > 0:
                xyxys = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                for xyxy, conf in zip(xyxys, confs):
                    x1p, y1p, x2p, y2p = map(int, xyxy)
                    candidates.append(
                        BBox(x1p, y1p, x2p, y2p, float(conf))
                    )

            if candidates:
                # ここでは単純に conf 最大の箱を採用
                det_box = max(candidates, key=lambda b: b.conf)

        # --- 検出があればトラッカー更新 ---
        if det_box is not None:
            last_detect_box = det_box
            x, y, x2, y2 = det_box.x1, det_box.y1, det_box.x2, det_box.y2
            w0, h0 = x2 - x, y2 - y
            tracker = create_tracker()
            track_ok = tracker.init(frame, (x, y, w0, h0))

        # --- 検出が無くても追跡継続 ---
        current_dist = None
        if tracker is not None:
            ok, box = tracker.update(frame)
            track_ok = ok
            if ok:
                x, y, w0, h0 = map(int, box)
                cx, cy = x + w0 // 2, y + h0 // 2
                smoothed_c = ema(smoothed_c, (cx, cy))

                # ==== 距離計算 ====
                if smoothed_c is not None:
                    sx, sy = smoothed_c
                    sx = int(np.clip(sx, 0, w - 1))
                    sy = int(np.clip(sy, 0, h - 1))

                    k = DEPTH_KERNEL // 2
                    xs = range(max(0, sx - k), min(w, sx + k + 1))
                    ys = range(max(0, sy - k), min(h, sy + k + 1))
                    depth_values = []
                    for yy in ys:
                        for xx in xs:
                            d = depth_frame.get_distance(xx, yy)  # [m]
                            if d > 0:
                                depth_values.append(d)

                    if depth_values:
                        d_med = float(np.median(depth_values))
                        if dist_smooth is None:
                            dist_smooth = d_med
                        else:
                            # 距離のEMAでさらになめらかに
                            dist_smooth = 0.7 * dist_smooth + 0.3 * d_med
                        current_dist = dist_smooth

                        # ★ ここで M5 に距離[m]を送る
                        send_range_to_drone(current_dist)
            else:
                # トラッカー喪失
                tracker = None
                track_ok = False
                smoothed_c = None
                dist_smooth = None

        # --- 最低限の描画（確認用） ---
        vis = frame.copy()
        if tracker is not None and track_ok:
            x, y, w0, h0 = map(int, box)
            cv2.rectangle(vis, (x, y), (x + w0, y + h0), (0, 255, 0), 2)
            if smoothed_c is not None:
                cv2.circle(vis, smoothed_c, 4, (0, 140, 255), -1)

        if current_dist is not None:
            text = f"distance={current_dist:.2f} m"
            color = (0, 255, 0)
        else:
            text = "distance=---"
            color = (0, 0, 255)

        cv2.putText(
            vis,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

        cv2.imshow("drone distance (minimal)", vis)

        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
