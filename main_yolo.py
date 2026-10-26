import cv2
from ultralytics import YOLO
from lg_controller import LGController
import time
from wakeonlan import send_magic_packet


model = YOLO("runs/pose/models/YOLOv10n_gestures.pt")

try:
    tv = LGController("192.168.0.24")
    tv.start()
except:
    print("Failed to start")

# Laisse le temps à la connexion de se faire
time.sleep(2)
cap = cv2.VideoCapture(0)
def get_grabbing_direction(result, dead_zone=100):

    three2_center = None
    grabbing_center = None

    # Cherche les deux gestes
    for i, cls in enumerate(result.boxes.cls):

        gesture = result.names[int(cls)]

        if gesture not in ("three2", "grabbing"):
            continue

        # Bounding box
        x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy()

        # Centre de la box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if gesture == "three2":
            three2_center = (cx, cy)

        elif gesture == "grabbing":
            grabbing_center = (cx, cy)

    # Il faut obligatoirement voir les deux mains
    if three2_center is None or grabbing_center is None:
        return None

    # Position de grabbing relativement à three2
    dx = grabbing_center[0] - three2_center[0]
    dy = grabbing_center[1] - three2_center[1]

    # Trop proches -> aucune direction
    if abs(dx) < dead_zone and abs(dy) < dead_zone:
        return None

    # Axe dominant
    if abs(dx) > abs(dy):

        if dx > 0:
            return "LEFT"
        else:
            return "RIGHT"

    else:

        if dy > 0:
            return "DOWN"
        else:
            return "UP"
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False)[0]
    detected_gestures = [
        result.names[int(cls)]
        for cls in result.boxes.cls
    ]
    if "like" in detected_gestures:
        if "one" in detected_gestures:
            print("one_on")
            send_magic_packet("20:28:BC:89:9D:EA")
            tv = LGController("192.168.0.24")
            tv.start()
            time.sleep(1)
        if "peace" in detected_gestures:
            print("two_on")
            try:
                tv.volume_up()
                time.sleep(0.5)
            except:
                print("allumez la télé")
    if "dislike" in detected_gestures:
        if "one" in detected_gestures:
            print("one_off")
            try:
                tv.power_off()
                time.sleep(1)
            except:
                print("allumez la télé")
        if "peace" in detected_gestures:
            print("two_off")
            try:
                tv.volume_down()
                time.sleep(0.5)
            except:
                print("allumez la télé")
    if "stop" in detected_gestures:
        if "one" in detected_gestures:
            print("netflix opening")
            try:
                tv.launch_app("netflix")
                time.sleep(1)
            except:
                print("allumez la télé")
        if "peace" in detected_gestures:
            print("youtube opening")
            try:
                tv.launch_app("youtube.leanback.v4")
                time.sleep(1)
            except:
                print("allumez la télé")
        if "middle_finger" in detected_gestures:
            print("spotify opening")
            try:
                tv.launch_app("spotify-beehive")
                time.sleep(1)
            except:
                print("allumez la télé")
    if "three2" in detected_gestures:
        if "peace" in detected_gestures:
            print("home")
            try:
                tv.home()
                time.sleep(1)
            except:
                print("allumez la télé")
        if "like" in detected_gestures:
            print("ok")
            try:
                tv.ok()
                time.sleep(1)
            except:
                print("allumez la télé")
        if "dislike" in detected_gestures:
            print("back")
            try:
                tv.back()
                time.sleep(1)
            except:
                print("allumez la télé")
        if "grabbing" in detected_gestures:
            print("palm")
            direction = get_grabbing_direction(
                result
            )

            print("Direction :", direction)

            try:

                if direction == "UP":
                    tv.up()

                elif direction == "DOWN":
                    tv.down()

                elif direction == "LEFT":
                    tv.left()

                elif direction == "RIGHT":
                    tv.right()
                time.sleep(1)
            except:
                print("allumez la télé")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()