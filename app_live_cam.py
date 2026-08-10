import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    rgb = cv2.cvtcolor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow('webcam',frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

