import cv2
import mediapipe as mp

# === Inisialisasi MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # 0 = kamera laptop, bisa diganti dengan index kamera lain

with mp_hands.Hands(
    max_num_hands=2,           # deteksi maksimal 2 tangan
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi BGR ke RGB (MediaPipe butuh RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Gambar titik dan koneksi tangan
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )

        cv2.imshow("Deteksi Gerakan Tangan - Titik Sendi", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

hand stack