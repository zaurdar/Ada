import cv2

for camera_id in range(5):
    print(f"\nTest camera {camera_id}")

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("  -> indisponible")
        continue

    # Laisse quelques frames à la caméra pour démarrer
    for _ in range(30):
        ret, frame = cap.read()

    if not ret:
        print("  -> aucune frame")
        cap.release()
        continue

    print("  -> OK", frame.shape)

    cv2.imshow(f"Camera {camera_id}", frame)
    print("Appuie sur une touche pour continuer")
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()