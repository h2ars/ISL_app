

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

with open("./saved_models/01_74.task", "rb") as f:
    model_content = f.read()

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=model_content),
    running_mode=VisionRunningMode.IMAGE,
)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)

with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:
    with GestureRecognizer.create_from_options(options) as recognizer:
        # Read Initial Frame
        if cap.isOpened():
            rval, frame = cap.read()
        else:
            rval = False

        # While can read frames
        while rval:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True

            # Get prediction
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            gesture_recognition_result = recognizer.recognize(mp_image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)

            if len(gesture_recognition_result.gestures) > 0:
                top_matches = [(
                    match.score,
                    match.category_name
                ) for match in gesture_recognition_result.gestures[0] if match.category_name != ""]

                if len(top_matches) > 0:  # and top_matches[0][0] >= 0.3:
                    cv2.putText(
                        image,
                        f"Result: {top_matches[0][1]} ({top_matches[0][0]:.2f})",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 255),
                        3,
                    )

            cv2.imshow("MediaPipe Hands", image)  # cv2.flip(image, 1
            rval, frame = cap.read()

            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break

cap.release()
cap.release()
cap.release()
cap.release()
