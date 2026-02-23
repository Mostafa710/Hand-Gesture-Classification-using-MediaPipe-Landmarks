import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import cv2

# 1 - recenter the landmarks (x, y) to make the origin the wrist point
# 2 - divide all the landmarks by the mid-finger tip position
# NOTE: z location doesn’t need to be processed as it is already processed

def preprocess_landmarks(landmarks):
    """
    Recenters landmarks to the wrist (0) and scales by the middle finger tip (12).
    """
    # Extract x and y coordinates (assuming columns are named x0, y0, z0, x1, y1, z1...)
    x_cols = [f'x{i}' for i in range(1, 22)]
    y_cols = [f'y{i}' for i in range(1, 22)]
    
    x_coords = landmarks[x_cols].values.astype(float)
    y_coords = landmarks[y_cols].values.astype(float)
    
    # Recenter: Make the wrist (index 0) the origin (0,0)
    x_wrist, y_wrist = x_coords[:, 0].reshape(-1, 1), y_coords[:, 0].reshape(-1, 1)
    x_centered = x_coords - x_wrist
    y_centered = y_coords - y_wrist
    
    # Scale: Divide by the middle finger tip (index 12) position
    scale_factor = np.sqrt(x_centered[:, 12]**2 + y_centered[:, 12]**2).reshape(-1, 1)
    
    # Avoid division by zero if landmark 12 is at the origin
    # if scale_factor != 0:
    #     x_scaled = x_centered / scale_factor
    #     y_scaled = y_centered / scale_factor
    # else:
    #     x_scaled, y_scaled = x_centered, y_centered

    # The previous code is commented, since the scaler factor will never be zero
    # Beacuse you centers the landmarks around the wrist point, which will be the origin.
    # Hince, the middle finger tip will never be at the origin.

    x_scaled = x_centered / scale_factor
    y_scaled = y_centered / scale_factor

    landmarks[x_cols] = x_scaled
    landmarks[y_cols] = y_scaled
        
    return landmarks

# 1. Load the SVC model
model = joblib.load('./Models/gesture_svc.joblib')

# 2. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# max_num_hands=1 simplifies the prediction logic for now
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.5)

# 3. Start Camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 4. Extract Landmark Coordinates (63 features)
            # Format: [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert to DataFrame for preprocessing
            cols = [f'{c}{i}' for i in range(1, 22) for c in ['x', 'y', 'z']]
            landmarks = pd.DataFrame([landmarks], columns=cols)

            # preprocess landmarks
            landmarks = preprocess_landmarks(landmarks)

            # 5. Model Prediction
            prediction = model.predict(landmarks)
            
            # Convert numeric ID back to the string name (e.g., 5 -> "peace")
            # ----------- The next line should be used only if you use XGBoost model -----------
            # label = encoder.inverse_transform(prediction)[0]

            # 6. Display the Label on the frame
            # Placing it near the wrist (landmark 0) or top left
            cv2.putText(frame, f"Gesture: {prediction}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the final frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()