import cv2
from ultralytics import YOLO


# ============================================================
# MODÈLE
# ============================================================

model = YOLO("runs/pose/models/YOLOv10n_gestures.pt")


# ============================================================
# CAMÉRA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la caméra")


# ============================================================
# BOUCLE
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Impossible de récupérer l'image")
        break

    # Détection YOLO
    result = model(
        frame,
        conf=0.4,
        verbose=False
    )[0]

    # --------------------------------------------------------
    # Affichage des détections dans la console
    # --------------------------------------------------------

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        gesture = result.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0]

        print(
            f"{gesture:20} "
            f"conf={confidence:.2f} "
            f"box=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
        )

    # --------------------------------------------------------
    # YOLO dessine automatiquement :
    # - bounding boxes
    # - classe
    # - confiance
    # --------------------------------------------------------

    annotated_frame = result.plot()

    cv2.imshow(
        "HaGRID Gesture Detection",
        annotated_frame
    )

    # Q pour quitter
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()