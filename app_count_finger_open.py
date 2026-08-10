import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)
finger_tip = [8 ,12 ,16 ,20]
while True:
    finger_count = 0
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS

            )
            lm = hand_landmarks.landmark

            if lm[4].x<lm[2].x:
                finger_count+=1

            for tip in finger_tip:
                if lm[tip].y<lm[tip-2].y:
                    finger_count+=1
            cv2.putText(
                frame,
                f"fingers:{finger_count}",
                (20 ,50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0,255,0),
                2

            )
    cv2.imshow("hand landmarks",frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

