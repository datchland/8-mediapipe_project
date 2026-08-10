import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

LEFT_EYEBROW_TOP = 105
LEFT_EYE_TOP = 159
RIGHT_EYEBROW_TOP = 334
RIGHT_EYE_TOP = 386

LEFT_EYE_BOTTOM = 145
RIGHT_EYE_BOTTOM = 374

FACE_LEFT_EDGE = 234
FACE_RIGHT_EDGE = 454


def euclidean(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_face_width(landmarks):
    """Used to normalize all distances so it works regardless of how
    close/far the face is from the camera."""
    return euclidean(landmarks[FACE_LEFT_EDGE], landmarks[FACE_RIGHT_EDGE])


def detect_emotion(landmarks):
    face_width = get_face_width(landmarks)


    mouth_width = euclidean(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]) / face_width
    mouth_open = euclidean(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]) / face_width

    
    left_brow = euclidean(landmarks[LEFT_EYEBROW_TOP], landmarks[LEFT_EYE_TOP]) / face_width
    right_brow = euclidean(landmarks[RIGHT_EYEBROW_TOP], landmarks[RIGHT_EYE_TOP]) / face_width
    brow_dist = (left_brow + right_brow) / 2


    left_eye = euclidean(landmarks[LEFT_EYE_TOP], landmarks[LEFT_EYE_BOTTOM]) / face_width
    right_eye = euclidean(landmarks[RIGHT_EYE_TOP], landmarks[RIGHT_EYE_BOTTOM]) / face_width
    eye_open = (left_eye + right_eye) / 2

    metrics = {
        "mouth_width": mouth_width,
        "mouth_open": mouth_open,
        "brow_dist": brow_dist,
        "eye_open": eye_open,
    }

    if mouth_width > 0.42 and mouth_open < 0.06:
        return "Happy", (0, 255, 0), metrics
    elif mouth_open > 0.09 and eye_open > 0.045:
        return "Surprised", (0, 255, 255), metrics
    elif brow_dist < 0.028 and mouth_width < 0.36:
        return "Angry", (0, 0, 255), metrics
    elif mouth_width < 0.34 and mouth_open < 0.02 and brow_dist > 0.028:
        return "Sad", (255, 0, 0), metrics
    else:
        return "Neutral", (200, 200, 200), metrics


def main():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        show_debug = False  

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    emotion, color, metrics = detect_emotion(landmarks)

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

                    cv2.putText(
                        frame, emotion, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
                    )

                    if show_debug:
                        y = 90
                        for name, value in metrics.items():
                            cv2.putText(
                                frame, f"{name}: {value:.3f}", (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                            )
                            y += 30

            cv2.putText(
                frame, "Press D to toggle numbers, Q to quit", (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
            )

            cv2.imshow("Face Emotion Detector - press Q to quit", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
