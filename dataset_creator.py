import cv2
import os
import time
import random
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Ancien modèle : utilisé uniquement pour trouver les mains
TEACHER_MODEL = r"runs\pose\models\hand_pose-2\weights\best.pt"

DATASET_PATH = "hand_gesture_dataset"

# Remplace les noms par tes vraies positions
CLASS_NAMES = [
    "position_1",
    "position_2",
    "position_3",
    "position_4",
    "position_5",
    "position_6",
]

# Temps entre deux sauvegardes
SAVE_INTERVAL = 0.20

# Confiance minimale du teacher pour accepter sa box
CONF_THRESHOLD = 0.50

# 80% train / 20% validation
TRAIN_RATIO = 0.80


# ============================================================
# CRÉATION DU DATASET
# ============================================================

for split in ["train", "val"]:
    os.makedirs(
        os.path.join(DATASET_PATH, "images", split),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(DATASET_PATH, "labels", split),
        exist_ok=True
    )


# ============================================================
# CRÉATION AUTOMATIQUE DU YAML
# ============================================================

yaml_path = os.path.join(DATASET_PATH, "hand_gestures.yaml")

absolute_dataset_path = os.path.abspath(DATASET_PATH).replace("\\", "/")

with open(yaml_path, "w") as f:

    f.write(f"path: {absolute_dataset_path}\n\n")

    f.write("train: images/train\n")
    f.write("val: images/val\n\n")

    f.write("names:\n")

    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {i}: {name}\n")


print("Dataset :", absolute_dataset_path)
print("YAML    :", yaml_path)


# ============================================================
# CHARGEMENT DU TEACHER
# ============================================================

model = YOLO(TEACHER_MODEL)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")


# ------------------------------------------------------------
# Etat
# ------------------------------------------------------------

current_class = None
last_save = 0

# timestamp permet d'éviter d'écraser les anciennes images
session_id = int(time.time())

counter = 0


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Erreur de lecture webcam")
        break

    # --------------------------------------------------------
    # Détection avec ton ancien YOLO Pose
    # --------------------------------------------------------

    results = model(
        frame,
        imgsz=640,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]

    # Copie annotée UNIQUEMENT pour l'affichage
    display = result.plot()


    # ========================================================
    # INTERFACE
    # ========================================================

    if current_class is None:

        status = "PAUSE - appuie sur 1-6"

        color = (255, 255, 255)

    else:

        status = (
            f"RECORDING : "
            f"{current_class + 1} - "
            f"{CLASS_NAMES[current_class]}"
        )

        color = (0, 0, 255)


    cv2.putText(
        display,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        display,
        "1-6 = classe | 0 = pause | Q = quitter",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAUVEGARDE AUTOMATIQUE
    # ========================================================

    current_time = time.time()

    if (
        current_class is not None
        and result.boxes is not None
        and len(result.boxes) > 0
        and current_time - last_save >= SAVE_INTERVAL
    ):

        H, W = frame.shape[:2]

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()


        # ----------------------------------------------------
        # On ne garde que les boxes suffisamment fiables
        # ----------------------------------------------------

        valid_boxes = []

        for box, confidence in zip(boxes, confidences):

            if confidence >= CONF_THRESHOLD:
                valid_boxes.append(box)


        if len(valid_boxes) > 0:

            # ------------------------------------------------
            # Split aléatoire train / val
            # ------------------------------------------------

            if random.random() < TRAIN_RATIO:
                split = "train"
            else:
                split = "val"


            # ------------------------------------------------
            # Nom unique
            # ------------------------------------------------

            filename = f"{session_id}_{counter:06d}"

            image_path = os.path.join(
                DATASET_PATH,
                "images",
                split,
                filename + ".jpg"
            )

            label_path = os.path.join(
                DATASET_PATH,
                "labels",
                split,
                filename + ".txt"
            )


            # ------------------------------------------------
            # Sauvegarde IMAGE ORIGINALE
            #
            # IMPORTANT :
            # on ne sauvegarde PAS 'display'
            # car display contient les dessins YOLO.
            # ------------------------------------------------

            cv2.imwrite(image_path, frame)


            # ------------------------------------------------
            # Création annotation YOLO
            # ------------------------------------------------

            with open(label_path, "w") as f:

                for box in valid_boxes:

                    x1, y1, x2, y2 = box

                    # XYXY -> XYWH
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2

                    width = x2 - x1
                    height = y2 - y1


                    # Normalisation 0 -> 1
                    x_center /= W
                    y_center /= H

                    width /= W
                    height /= H


                    # Format YOLO :
                    #
                    # class x_center y_center width height
                    #
                    f.write(
                        f"{current_class} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}\n"
                    )


            print(
                f"[{split.upper()}] "
                f"{filename} | "
                f"{CLASS_NAMES[current_class]} | "
                f"{len(valid_boxes)} main(s)"
            )

            counter += 1
            last_save = current_time


    # ========================================================
    # AFFICHAGE
    # ========================================================

    cv2.imshow(
        "Hand Gesture Dataset Recorder",
        display
    )


    # ========================================================
    # CLAVIER
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # 1 -> classe 0
    # 2 -> classe 1
    # ...
    # 6 -> classe 5

    if ord("1") <= key <= ord("6"):

        current_class = key - ord("1")

        print()
        print("==============================")
        print(
            "Classe :",
            current_class,
            CLASS_NAMES[current_class]
        )
        print("==============================")


    # Pause
    elif key == ord("0"):

        current_class = None

        print("PAUSE")


    # Quitter
    elif key == ord("q"):

        break


# ============================================================
# FIN
# ============================================================

cap.release()
cv2.destroyAllWindows()

print()
print("Capture terminee")
print(f"{counter} images enregistrees")
print("Dataset :", absolute_dataset_path)
print("YAML :", yaml_path)