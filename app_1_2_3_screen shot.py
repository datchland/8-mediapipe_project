import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks, label):
    fingers = []

    # Thumb (compare x, since thumb moves sideways)
    if label == "Right":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

    # Other 4 fingers (compare y, tip above pip joint = up)
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

sequence = []
last_count = -1
last_change_time = time.time()
screenshot_taken = False
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    h, w, _ = frame.shape
    count = -1

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0]
        label = handedness.classification[0].label

        count = count_fingers(hand_landmarks, label)

        cv2.putText(
            frame,
            f"Fingers: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

       
        if count != last_count:
            last_count = count
            last_change_time = time.time()
        elif time.time() - last_change_time > 0.5:  
            if not sequence or sequence[-1] != count:
                sequence.append(count)
                last_change_time = time.time() + 999  

    else:
        last_count = -1

    
    sequence = sequence[-3:]

    
    if sequence == [1, 2, 3] and not screenshot_taken:
        screenshot_count += 1
        filename = f"screenshot_{screenshot_count}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        screenshot_taken = True
        sequence = []

    if count != 3:
        screenshot_taken = False

    cv2.imshow("finger count screenshot", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()