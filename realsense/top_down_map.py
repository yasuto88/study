"""
simple_wall_map.py
RealSense D435/D435i  +  OpenCV  (= cv2)

・深度点を俯瞰マップに投影
・見えたところをそのまま線として描画
・ESC で終了

生成物: 600×600 px の 2D マップ (cv2.imshow)
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# -------------- 可変パラメータ ---------------- #
PX = 600  # マップ画像の一辺 [px]
RANGE_X_M = 3.0  # 左右 ±m をマップ左右
RANGE_Z_M = 4.0  # 手前 0 〜 奥 m をマップ上下
HEIGHT_MIN, HEIGHT_MAX = -0.2, 2.0  # 使う高さ [m]
ACCUM_FRAMES = 20  # ノイズを減らすため連続何フレーム分を合成するか
# --------------------------------------------- #

SCALE_X = PX / (2 * RANGE_X_M)
SCALE_Z = PX / RANGE_Z_M

# RealSense 準備
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe = rs.pipeline()
align = rs.align(rs.stream.color)
profile = pipe.start(cfg)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
pc = rs.pointcloud()

# 俯瞰図バッファ
accum = np.zeros((PX, PX), np.float32)


def world_to_img(x, z):
    u = np.int32(PX / 2 + x * SCALE_X)
    v = np.int32(PX - z * SCALE_Z)
    return u, v


print("▶ ESC キーで終了")

try:
    while cv2.waitKey(1) != 27:
        # --- 俯瞰マップを初期化 ---
        occ = np.zeros_like(accum)

        # --- 何フレームか重ねてノイズ低減 ---
        for _ in range(ACCUM_FRAMES):
            frames = align.process(pipe.wait_for_frames())
            depth = frames.get_depth_frame()
            if not depth:
                continue

            points = pc.calculate(depth)
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

            # 高さフィルタ
            y_ok = (vtx[:, 1] > HEIGHT_MIN) & (vtx[:, 1] < HEIGHT_MAX)
            vtx = vtx[y_ok]

            if vtx.size == 0:
                continue
            u, v = world_to_img(vtx[:, 0], vtx[:, 2])

            # マップ内に収まる点だけ
            in_map = (u >= 0) & (u < PX) & (v >= 0) & (v < PX)
            occ[v[in_map], u[in_map]] += 1  # 出現回数で累積

        # 正規化して 0/255 画像に
        occ_norm = cv2.normalize(occ, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 薄い点を膨張させ“線”に見せる
        walls = cv2.dilate(occ_norm, np.ones((3, 3), np.uint8), iterations=1)

        # 見やすく反転 (壁＝白, 背景＝黒)
        map_img = cv2.bitwise_not(walls)

        cv2.imshow("Top-Down Wall Sketch", map_img)

finally:
    pipe.stop()
    cv2.destroyAllWindows()
