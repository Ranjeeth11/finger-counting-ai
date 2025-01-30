import cv2
import mediapipe as mp
import pyttsx3

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define finger landmarks
FINGER_TIPS = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
THUMB_TIP = 5  # Thumb tip
THUMB_IP = 4  # Thumb IP joint
THUMB_BASE = 2  # Thumb MCP base joint

# Open the camera
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, hand_label):
    fingers = []

    # Check fingers (Index, Middle, Ring, Pinky)
    for tip in FINGER_TIPS:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)  # Finger is up
        else:
            fingers.append(0)  # Finger is down

    # Improved Thumb Detection
    if hand_label == "Right":
        if (hand_landmarks.landmark[THUMB_TIP].x > hand_landmarks.landmark[THUMB_BASE].x and
            hand_landmarks.landmark[THUMB_TIP].x > hand_landmarks.landmark[THUMB_IP].x):
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)
    else:  # Left hand
        if (hand_landmarks.landmark[THUMB_TIP].x < hand_landmarks.landmark[THUMB_BASE].x and
            hand_landmarks.landmark[THUMB_TIP].x < hand_landmarks.landmark[THUMB_IP].x):
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)

    return sum(fingers)  # Return total number of fingers up

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for a natural user experience
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get hand label (Right or Left)
            hand_label = results.multi_handedness[idx].classification[0].label

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers
            finger_count = count_fingers(hand_landmarks, hand_label)

            # Display detected number
            cv2.putText(frame, f"Fingers: {finger_count}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Speak the number
            engine.say(str(finger_count))
            engine.runAndWait()

    # Show the camera feed
    cv2.imshow("Finger Counter", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()