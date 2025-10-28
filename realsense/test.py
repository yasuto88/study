"""
webcam_infer.py ― YOLOv11 で Web カメラをリアルタイム推論
  - モデル: model_-27-june-2025-16_17.pt
  - デバイス: CPU
  - 終了: ウィンドウで Q キー
"""

from ultralytics import YOLO
import cv2

# ── 1) モデルをロード ──
MODEL_PATH = "model.pt"
model = YOLO(MODEL_PATH)  # 重みとネットワークを復元
print("Loaded:", MODEL_PATH)
print("Classes:", model.names)  # 学習時に登録したクラス名

# ── 2) カメラをオープン ──
cap = cv2.VideoCapture(0)  # 0 = 既定のカメラ
if not cap.isOpened():
    raise RuntimeError(
        "⚠️ カメラが開けません。index を変えるかドライバをご確認ください。"
    )

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 3) 推論 → アノテーション追加 ──
        results = model(frame, device="cpu", imgsz=640, verbose=False)[0]
        annotated = results.plot()  # BGR (OpenCV) 画像に描画済み

        # ── 4) 画面に表示 ──
        cv2.imshow("YOLO Detection (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
