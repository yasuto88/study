import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
import pyrealsense2 as rs
import time
import sys
import requests
import torch  # GPU(CUDA)利用の有無を確認するため
import math


# =========================
# 設定
# =========================
MODEL_PATH = "yolo11n_drone.pt"  # 学習済みドローン検出モデル
CONF_THRES = 0.25
IOU_THRES = 0.45
DETECT_INTERVAL = 20  # YOLO再検出の間隔 (大→軽い/遅延減, 小→頑丈/重い)
DEPTH_KERNEL = 5      # 中心周りの深度を平均するサイズ（奇数）

# トラッカー喪失後に距離をホールド / 深度のみ更新する最大フレーム数
MAX_HOLD_FRAMES = 15  # 30fpsなら ~0.5秒

# StampFly コントローラ側 HTTP 設定
CTRL_HOST = "192.168.4.1"
CTRL_PORT = 80
RANGE_ENDPOINT = f"http://{CTRL_HOST}:{CTRL_PORT}/range"

SEND_INTERVAL_SEC = 0.05  # 何秒ごとに距離を送るか（20Hz）

# ★ カメラの論理順を固定する
FIXED_ORDER = [
    "029522250211",  # → Cam0
    "029522250255",  # → Cam1
    "029522250039",  # → Cam2
]


# OpenCV ちょい高速化
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(0)
except Exception:
    pass


# =========================
# GPU / CPU デバイス設定
# =========================
if torch.cuda.is_available():
    YOLO_DEVICE = 0  # or "cuda:0"
    print(f"[INFO] CUDA が利用可能なため GPU(device={YOLO_DEVICE}) を使用します。")
else:
    YOLO_DEVICE = "cpu"
    print("[WARN] CUDA が見つからないため CPU で実行します。")


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
    """環境差に強いトラッカー生成（MOSSE優先→KCF→CSRT）"""
    # MOSSE（めちゃ軽い）
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        return cv2.legacy.TrackerMOSSE_create()
    # KCF（そこそこ）
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    # CSRT（重いけど精度高め）
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    raise RuntimeError(
        "OpenCVにMOSSE/KCF/CSRTトラッカーが見つかりません。"
        "opencv-contrib-python を確認してください。"
    )


# 距離送信ヘルパ
_last_send_time = 0.0


def send_range_to_drone(dist_m: float):
    """距離[m]を StampFly コントローラへ送信"""
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
        print(f"[WARN] send_range_to_drone failed: {e}", file=sys.stderr)

# ウィンドウ配置サイズ（全カメラ共通）
WIN_W = 320
WIN_H = 240
SPACING = 10
START_X = 50
START_Y = 50

# =========================
# RealSense デバイス固定順 初期化
# =========================

# ★ 論理順（Cam0, Cam1, Cam2）にしたいシリアル番号を指定する
FIXED_ORDER = [
    "029522250211",  # Cam0
    "029522250255",  # Cam1
    "029522250039",  # Cam2
]

@dataclass
class CameraState:
    serial: str
    pipeline: rs.pipeline
    align: rs.align
    depth_scale: float
    window_name: str
    tracker: any = None
    track_ok: bool = False
    last_detect_box: BBox | None = None
    smoothed_c: tuple[int, int] | None = None
    dist_smooth: float | None = None
    last_dist: float | None = None
    lost_frames: int = 0
    frame_idx: int = 0
    last_box: tuple[int, int, int, int] | None = None
    last_vis: np.ndarray | None = None


# RealSense デバイス取得
ctx = rs.context()
device_list = ctx.query_devices()

if len(device_list) == 0:
    raise RuntimeError("RealSense デバイスが見つかりません")

# ★ シリアル → dev の辞書
device_map = {}
for dev in device_list:
    serial = dev.get_info(rs.camera_info.serial_number)
    device_map[serial] = dev

# ★ cameras を固定順で並べる
cameras: list[CameraState] = []

for idx, serial in enumerate(FIXED_ORDER):
    if serial not in device_map:
        raise RuntimeError(f"指定したカメラ {serial} が接続されていません")

    dev = device_map[serial]
    print(f"[INFO] 初期化（順固定） serial={serial}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)

    # 深度/カラーを開始
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] depth_scale({serial}): {depth_scale} m/unit")

    align = rs.align(rs.stream.color)

    window_name = f"RealSense {idx} ({serial})"

    # ウィンドウ配置（任意）
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(window_name, 320, 240)
    except Exception:
        pass
    cv2.moveWindow(window_name, 50 + idx * 350, 50)

    cameras.append(
        CameraState(
            serial=serial,
            pipeline=pipeline,
            align=align,
            depth_scale=depth_scale,
            window_name=window_name,
        )
    )


if not cameras:
    raise RuntimeError("RealSense の初期化にすべて失敗しました。")

# 3台だけを使う想定（もし4台以上あってもとりあえず先頭3台）
if len(cameras) > 3:
    print("[INFO] 3台まで使用します。それ以外のカメラは無視します。")
    cameras = cameras[:3]

if len(cameras) < 3:
    print("[WARN] カメラが3台未満のため、三辺測量は行えません。")


# =========================
# カメラ配置（世界座標）設定
# =========================
# 一辺 0.30 m の正三角形:
SIDE_LEN = 0.60  # [m]
tri_height = math.sqrt(3) / 2 * SIDE_LEN

# ★カメラ検出順を Cam0, Cam1, Cam2 として…
#   Cam0: (0, 0)
#   Cam1: (0.30, 0)
#   Cam2: (0.15, 0.30 * √3 / 2)
# 物理配置が違う場合はここを書き換えてください。
CAMERA_POSITIONS = np.array(
    [
        [0.0, 0.0],
        [SIDE_LEN, 0.0],
        [SIDE_LEN / 2.0, tri_height],
    ],
    dtype=np.float32,
)


def trilaterate_xy(positions: np.ndarray, distances: list[float]):
    """
    2D 三辺測量 (x,y) を求める
    positions: shape (3,2)  各カメラの (x,y) [m]
    distances: [r1, r2, r3] 各カメラからの距離 [m]
    """
    if len(positions) != 3 or len(distances) != 3:
        return None

    p1 = positions[0]
    p2 = positions[1]
    p3 = positions[2]
    r1, r2, r3 = distances

    # 連立方程式を構成:
    # (x - x1)^2 + (y - y1)^2 = r1^2
    # (x - x2)^2 + (y - y2)^2 = r2^2
    # (x - x3)^2 + (y - y3)^2 = r3^2
    # → 1式目を2,3から引いて線形方程式にする
    A = np.array(
        [
            [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])],
            [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1])],
        ],
        dtype=np.float64,
    )
    b = np.array(
        [
            r1**2
            - r2**2
            + p2[0] ** 2
            - p1[0] ** 2
            + p2[1] ** 2
            - p1[1] ** 2,
            r1**2
            - r3**2
            + p3[0] ** 2
            - p1[0] ** 2
            + p3[1] ** 2
            - p1[1] ** 2,
        ],
        dtype=np.float64,
    )

    try:
        x, y = np.linalg.solve(A, b)
        return float(x), float(y)
    except np.linalg.LinAlgError:
        return None


# =========================
# Triangulation プロット用
# =========================
TRI_WIN_NAME = "Triangulation"
TRI_SIZE = 600  # 600x600 ピクセル
WORLD_SCALE = 200.0  # [px/m] 1m = 200px くらい
traj_points: list[tuple[float, float]] = []  # 世界座標での軌跡

cv2.namedWindow(TRI_WIN_NAME, cv2.WINDOW_NORMAL)
try:
    cv2.resizeWindow(TRI_WIN_NAME, TRI_SIZE, TRI_SIZE)
except Exception:
    pass
try:
    cv2.moveWindow(TRI_WIN_NAME, 50, START_Y + WIN_H + 80)
except Exception:
    pass


def world_to_canvas(x: float, y: float):
    """
    世界座標 (m) → 画像座標 (px)
    画像中心を (0,0) とし、+x 右、+y 上 になるように変換。
    """
    cx = TRI_SIZE // 2 + int(x * WORLD_SCALE)
    cy = TRI_SIZE // 2 - int(y * WORLD_SCALE)
    return cx, cy


def draw_triangulation_window(current_pos: tuple[float, float] | None):
    """
    三辺測量の結果とカメラ位置・軌跡を描画
    """
    canvas = np.zeros((TRI_SIZE, TRI_SIZE, 3), dtype=np.uint8)

    # 背景を少しグレーに
    canvas[:] = (30, 30, 30)

    # グリッド（0.5mごとなど、軽めに）
    for dx in range(-5, 6):
        x_px1, y_px1 = world_to_canvas(dx * 0.5, -5 * 0.5)
        x_px2, y_px2 = world_to_canvas(dx * 0.5, 5 * 0.5)
        cv2.line(canvas, (x_px1, y_px1), (x_px2, y_px2), (60, 60, 60), 1)

    for dy in range(-5, 6):
        x_px1, y_px1 = world_to_canvas(-5 * 0.5, dy * 0.5)
        x_px2, y_px2 = world_to_canvas(5 * 0.5, dy * 0.5)
        cv2.line(canvas, (x_px1, y_px1), (x_px2, y_px2), (60, 60, 60), 1)

    # 原点
    origin_px = world_to_canvas(0.0, 0.0)
    cv2.circle(canvas, origin_px, 4, (255, 255, 255), -1)
    cv2.putText(
        canvas, "O", (origin_px[0] + 5, origin_px[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    # カメラ位置（青）
    for i in range(min(3, len(cameras))):
        px, py = world_to_canvas(float(CAMERA_POSITIONS[i, 0]), float(CAMERA_POSITIONS[i, 1]))
        cv2.circle(canvas, (px, py), 6, (255, 0, 0), -1)
        label = f"C{i}"
        cv2.putText(
            canvas, label, (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )

    # これまでの軌跡（緑の折れ線）
    if len(traj_points) >= 2:
        for i in range(1, len(traj_points)):
            x1, y1 = traj_points[i - 1]
            x2, y2 = traj_points[i]
            p1 = world_to_canvas(x1, y1)
            p2 = world_to_canvas(x2, y2)
            cv2.line(canvas, p1, p2, (0, 200, 0), 2)

    # 現在位置（赤）
    if current_pos is not None:
        x, y = current_pos
        px, py = world_to_canvas(x, y)
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
        coord_text = f"X={x:.2f}m, Y={y:.2f}m"
        cv2.putText(
            canvas, coord_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    cv2.imshow(TRI_WIN_NAME, canvas)


# =========================
# モデル 初期化
# =========================
model = YOLO(MODEL_PATH)
try:
    model.fuse()
except Exception:
    pass

# GPU ウォームアップ（任意）
if YOLO_DEVICE != "cpu":
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model.predict(
        source=dummy,
        device=YOLO_DEVICE,
        conf=0.01,
        iou=0.5,
        imgsz=512,
        half=True,
        verbose=False,
    )
    print("[INFO] YOLO ウォームアップ完了。")


# =========================
# メインループ
# =========================
try:
    while True:
        distances = []  # 各カメラの距離（min送信用）
        current_world_pos: tuple[float, float] | None = None

        for cam in cameras:
            vis = None
            current_dist = None

            # ---- フレーム取得 ----
            frames = cam.pipeline.poll_for_frames()
            if frames:
                aligned_frames = cam.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if depth_frame and color_frame:
                    frame = np.asanyarray(color_frame.get_data())
                    h, w = frame.shape[:2]

                    # --- 一定間隔で YOLO 再検出 ---
                    do_detect = (cam.frame_idx % DETECT_INTERVAL == 0) or (not cam.track_ok)
                    det_box = None

                    if do_detect:
                        results = model.predict(
                            source=frame,
                            conf=CONF_THRES,
                            iou=IOU_THRES,
                            device=YOLO_DEVICE,
                            imgsz=512,
                            half=True,
                            verbose=False,
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
                            det_box = max(candidates, key=lambda b: b.conf)

                    # --- 検出があればトラッカー更新 ---
                    if det_box is not None:
                        cam.last_detect_box = det_box
                        x, y, x2, y2 = det_box.x1, det_box.y1, det_box.x2, det_box.y2
                        w0, h0 = x2 - x, y2 - y
                        cam.tracker = create_tracker()
                        cam.track_ok = cam.tracker.init(frame, (x, y, w0, h0))
                        cam.lost_frames = 0

                    # --- 検出が無くても追跡継続 or 距離ホールド ---
                    if cam.tracker is not None:
                        ok, box = cam.tracker.update(frame)
                        cam.track_ok = ok

                        if ok:
                            cam.lost_frames = 0
                            x, y, w0, h0 = map(int, box)
                            cam.last_box = (x, y, w0, h0)

                            cx, cy = x + w0 // 2, y + h0 // 2
                            cam.smoothed_c = ema(cam.smoothed_c, (cx, cy))

                            # ==== 距離計算 ====
                            if cam.smoothed_c is not None:
                                sx, sy = cam.smoothed_c
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
                                    if cam.dist_smooth is None:
                                        cam.dist_smooth = d_med
                                    else:
                                        cam.dist_smooth = 0.7 * cam.dist_smooth + 0.3 * d_med
                                    current_dist = cam.dist_smooth

                        else:
                            # トラッカー喪失
                            cam.lost_frames += 1

                            if cam.smoothed_c is not None and cam.lost_frames <= MAX_HOLD_FRAMES:
                                sx, sy = cam.smoothed_c
                                sx = int(np.clip(sx, 0, w - 1))
                                sy = int(np.clip(sy, 0, h - 1))

                                k = DEPTH_KERNEL // 2
                                xs = range(max(0, sx - k), min(w, sx + k + 1))
                                ys = range(max(0, sy - k), min(h, sy + k + 1))
                                depth_values = []
                                for yy in ys:
                                    for xx in xs:
                                        d = depth_frame.get_distance(xx, yy)
                                        if d > 0:
                                            depth_values.append(d)

                                if depth_values:
                                    d_med = float(np.median(depth_values))
                                    if cam.dist_smooth is None:
                                        cam.dist_smooth = d_med
                                    else:
                                        cam.dist_smooth = 0.7 * cam.dist_smooth + 0.3 * d_med
                                    current_dist = cam.dist_smooth
                                else:
                                    # 深度が取れない場合は最後の距離だけホールド
                                    current_dist = cam.dist_smooth
                            else:
                                # 完全ロスト
                                cam.tracker = None
                                cam.track_ok = False
                                cam.smoothed_c = None
                                cam.dist_smooth = None
                                cam.last_box = None
                                current_dist = None

                    # --- 描画 ---
                    vis = frame.copy()
                    if cam.tracker is not None and cam.track_ok and cam.last_box is not None:
                        x, y, w0, h0 = cam.last_box
                        cv2.rectangle(vis, (x, y), (x + w0, y + h0), (0, 255, 0), 2)
                        if cam.smoothed_c is not None:
                            cv2.circle(vis, cam.smoothed_c, 4, (0, 140, 255), -1)
                    elif cam.smoothed_c is not None:
                        cv2.circle(vis, cam.smoothed_c, 4, (0, 255, 255), -1)

                    label = f"{cam.serial}: "
                    if current_dist is not None:
                        cam.last_dist = current_dist
                        text = f"{label}{current_dist:.2f} m"
                        color = (0, 255, 0)
                        distances.append(current_dist)
                    else:
                        text = f"{label}---"
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

                    cam.frame_idx += 1
                    cam.last_vis = vis  # 最新フレームを保存

            # フレームが取れなかった場合でも、前回の画像を使う
            if cam.last_vis is not None:
                try:
                    vis_small = cv2.resize(cam.last_vis, (WIN_W, WIN_H))
                except Exception:
                    vis_small = cam.last_vis
                cv2.imshow(cam.window_name, vis_small)

        # --- 三辺測量で絶対位置を算出 ---
        if len(cameras) >= 3:
            # 先頭3台の距離を使用
            dists_for_tri = [cameras[i].last_dist for i in range(3)]
            if all((d is not None and d > 0.0) for d in dists_for_tri):
                current_world_pos = trilaterate_xy(CAMERA_POSITIONS, dists_for_tri)
                if current_world_pos is not None:
                    # 軌跡は一定数だけ保持
                    traj_points.append(current_world_pos)
                    if len(traj_points) > 500:
                        traj_points.pop(0)

        # Triangulation ウィンドウ描画
        draw_triangulation_window(current_world_pos)

        # --- 全カメラのうち最小距離を送信（従来通り） ---
        if distances:
            min_dist = min(distances)
            send_range_to_drone(min_dist)

        # キー入力チェック（どのウィンドウにフォーカスがあってもOK）
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

finally:
    for cam in cameras:
        try:
            cam.pipeline.stop()
        except Exception:
            pass
    cv2.destroyAllWindows()
