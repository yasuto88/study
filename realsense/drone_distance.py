import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass

# =========================
# 設定
# =========================
MODEL_PATH = "yolo11n_drone.pt"  # 学習済みドローン検出モデル
CAM_ID = 0  # カメラID
CONF_THRES = 0.25
IOU_THRES = 0.45
DETECT_INTERVAL = 10  # YOLO再検出の間隔 (大→軽い/遅延減, 小→頑丈/重い)
MOTION_MIN_AREA = 300  # 動体マスクの最小面積
EMA_ALPHA = 0.35  # 位置の平滑化強さ
REACQUIRE_MAX_FRAMES = 30  # トラッカーロスト後の再取得猶予
SHOW_EVERY = 1  # 何フレームに1回描画するか（間引き）
ROI_MARGIN = 10  # ROIの外側に足すマージン（ピクセル）

# =========================
# ちょい高速化（OpenCV）
# =========================
cv2.setUseOptimized(True)
try:
    # Ultralytics側で0にする想定だが、環境で未実装なことがあるのでtry
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


def iou(a: BBox, b: BBox) -> float:
    xx1, yy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    xx2, yy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
    inter = w * h
    area = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / area if area > 0 else 0.0


def ema(prev, new, alpha=EMA_ALPHA):
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
        "OpenCVにCSRT/KCF/MOSSEトラッカーが見つかりません。opencv-contrib-python を確認してください。"
    )


# =========================
# モデル/カメラ 初期化
# =========================
model = YOLO(MODEL_PATH)
# Conv+BN融合でちょい加速
try:
    model.fuse()
except Exception:
    pass

# WindowsならDirectShow指定のほうが遅延が少ないことが多い
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
# 古いフレームを溜めない
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 背景差分（固定背景を想定）
bg = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=16, detectShadows=False
)

tracker = None
track_ok = False
track_box = None  # (x, y, w, h)
smoothed_c = None
last_detect_box = None
frames_since_detect = 0
lost_count = 0
frame_idx = 0

trail = deque(maxlen=50)  # 可視化用の軌跡

while True:
    # ループで古いフレームを捨てる（grabで進め、最新だけretrieve）
    for _ in range(2):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        break

    h, w = frame.shape[:2]

    # --- 動きマスク ---
    fg = bg.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = fg

    # --- ROIを決定（動体領域の外接矩形） ---
    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    roi_rect = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) >= MOTION_MIN_AREA:
            rx, ry, rw, rh = cv2.boundingRect(c)
            x0 = max(0, rx - ROI_MARGIN)
            y0 = max(0, ry - ROI_MARGIN)
            x1 = min(w, rx + rw + ROI_MARGIN)
            y1 = min(h, ry + rh + ROI_MARGIN)
            roi_rect = (x0, y0, x1, y1)

    # --- 一定間隔で再検出 ---
    do_detect = (frames_since_detect % DETECT_INTERVAL == 0) or (not track_ok)
    det_box = None

    if do_detect:
        # ROIがあればそこだけ推論（高速化＆遅延低減）
        if roi_rect is not None:
            x0, y0, x1, y1 = roi_rect
            roi = frame[y0:y1, x0:x1]
            results = model.predict(
                source=roi, conf=CONF_THRES, iou=IOU_THRES, verbose=False
            )[0]
            offset = (x0, y0)
        else:
            results = model.predict(
                source=frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False
            )[0]
            offset = (0, 0)

        candidates = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxys = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            for xyxy, conf, cls in zip(xyxys, confs, clss):
                x1p, y1p, x2p, y2p = map(int, xyxy)
                # ROIを使った場合は座標を元画像へオフセット
                x1g = x1p + offset[0]
                y1g = y1p + offset[1]
                x2g = x2p + offset[0]
                y2g = y2p + offset[1]

                cx, cy = (x1g + x2g) // 2, (y1g + y2g) // 2
                if 0 <= cx < w and 0 <= cy < h and motion_mask[cy, cx] > 0:
                    patch = motion_mask[
                        max(0, y1g) : min(h, y2g), max(0, x1g) : min(w, x2g)
                    ]
                    area = int(patch.sum() / 255)
                    if area >= MOTION_MIN_AREA:
                        candidates.append(BBox(x1g, y1g, x2g, y2g, float(conf)))

        if candidates:
            if last_detect_box is None:
                det_box = max(candidates, key=lambda b: b.conf)
            else:
                det_box = max(candidates, key=lambda b: iou(b, last_detect_box))

    # --- 検出があればトラッカー更新 ---
    if det_box is not None:
        last_detect_box = det_box
        x, y, x2, y2 = det_box.x1, det_box.y1, det_box.x2, det_box.y2
        w0, h0 = x2 - x, y2 - y
        tracker = create_tracker()
        track_ok = tracker.init(frame, (x, y, w0, h0))
        lost_count = 0

    # --- 検出が無くても追跡継続 ---
    if tracker is not None:
        ok, box = tracker.update(frame)
        track_ok = ok
        if ok:
            x, y, w0, h0 = map(int, box)
            track_box = (x, y, w0, h0)
            cx, cy = x + w0 // 2, y + h0 // 2
            smoothed_c = ema(smoothed_c, (cx, cy))
            trail.append(smoothed_c)
            frames_since_detect += 1
        else:
            lost_count += 1
            if lost_count > REACQUIRE_MAX_FRAMES:
                tracker = None
                track_ok = False
                frames_since_detect = 0
                smoothed_c = None
                trail.clear()
    else:
        frames_since_detect = 0

    # --- 描画（間引き可） ---
    if frame_idx % SHOW_EVERY == 0:
        vis = frame.copy()

        # 動きマスクプレビュー（右下）
        small = cv2.resize(motion_mask, (w // 5, h // 5))
        small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        vis[h - h // 5 - 10 : h - 10, w - w // 5 - 10 : w - 10] = small_bgr

        # ROI可視化
        if roi_rect is not None:
            x0, y0, x1, y1 = roi_rect
            cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 80, 255), 2)

        if track_ok and track_box is not None:
            x, y, ww, hh = track_box
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            if smoothed_c:
                cv2.circle(vis, smoothed_c, 4, (0, 140, 255), -1)
        elif last_detect_box is not None:
            b = last_detect_box
            cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), (255, 200, 0), 2)

        # 軌跡
        for i in range(1, len(trail)):
            cv2.line(vis, trail[i - 1], trail[i], (0, 255, 255), 2)

        cv2.putText(
            vis,
            f"track_ok={track_ok}  lost={lost_count}  since_detect={frames_since_detect}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("drone-detect+track+motion", vis)

    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
