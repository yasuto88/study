"""
click_depth.py  ―  RealSense D435/D435i
マウスでクリックした画素の距離と 3D 座標を表示
ESC で終了
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# ------- RealSense パイプライン設定 -------
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipe = rs.pipeline()
align = rs.align(rs.stream.color)  # Depth→Color 座標合わせ
profile = pipe.start(cfg)

# 深度スケール (mm 変換用)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()  # meters
print(f"Depth scale = {depth_scale*1000:.3f} mm/LSB")

# PointCloud 用オブジェクト
pc = rs.pointcloud()
clicked = None  # クリック座標 (x, y)
text = ""  # 表示用テキスト


def mouse_cb(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = (x, y)


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", mouse_cb)

try:
    while cv2.waitKey(1) != 27:  # ESC で終了
        frames = align.process(pipe.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # クリックされたら距離を計算
        if clicked:
            cx, cy = clicked
            d = depth[cy, cx]  # z16: 単位 = 1 (depth_scale m)
            if d == 0:
                text = "Depth: Invalid (0)"
            else:
                dist_m = d * depth_scale  # 距離 [m]
                # --------- 3D 座標に変換 ---------
                pc.map_to(color_frame)
                points = pc.calculate(depth_frame)
                vtx = (
                    np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
                )
                idx = cy * depth.shape[1] + cx
                X, Y, Z = vtx[idx]
                text = f"{dist_m*1000:.1f} mm  |  X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f} m"

            clicked = None  # 連続計算を防ぐ

        # 表示
        vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
        )
        overlay = color.copy()
        cv2.addWeighted(vis, 0.6, overlay, 0.4, 0, overlay)

        if text:
            cv2.putText(
                overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        cv2.imshow("RGB", overlay)

finally:
    pipe.stop()
    cv2.destroyAllWindows()
