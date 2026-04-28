import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    letter = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark

            fingers_up = 0

            # Thumb
            if lm[4].x < lm[3].x:
                fingers_up += 1

            # Other fingers
            for tip in finger_tips:
                if lm[tip].y < lm[tip - 2].y:
                    fingers_up += 1

            # A = fist (0 fingers)
            if fingers_up == 0:
                letter = "A"

            # B = open hand (5 fingers)
            elif fingers_up == 5:
                letter = "B"

            # C = 3 fingers (simple assumption)
            elif fingers_up == 3:
                letter = "C"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Sign: {letter}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ABC Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
