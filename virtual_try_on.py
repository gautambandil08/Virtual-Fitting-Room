import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load the virtual clothing items
cloth_path = os.path.join('static', 'boys_11.png')

# Check if the coat image exists
if not os.path.exists(cloth_path):
    raise FileNotFoundError(f"The file {cloth_path} does not exist.")
virtual_clothing = cv2.imread(cloth_path, cv2.IMREAD_UNCHANGED)
if virtual_clothing is None:
    raise ValueError(f"Failed to load the image {cloth_path}")
if virtual_clothing.shape[2] != 4:
    raise ValueError(f"The image {cloth_path} does not have an alpha channel")

print(f"Loaded coat image with shape: {virtual_clothing.shape}")

def calculate_body_parameters(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    neck_x = (left_shoulder.x + right_shoulder.x) / 2
    neck_y = (left_shoulder.y + right_shoulder.y) / 2 + (nose.y - (left_shoulder.y + right_shoulder.y) / 2) / 2
    body_angle = np.arctan2(right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x)

    return neck_x, neck_y, body_angle, left_shoulder, right_shoulder, left_hip, right_hip

def resize_clothing(image, clothing, landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    shoulder_width = np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) - 
                                    np.array([right_shoulder.x, right_shoulder.y])) * image.shape[1]
    torso_height = np.linalg.norm(np.array([(left_shoulder.x + right_shoulder.x)/2, (left_shoulder.y + right_shoulder.y)/2]) - 
                                  np.array([(left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2])) * image.shape[0]

    width_scale = shoulder_width / clothing.shape[1] * 2.2
    height_scale = torso_height / clothing.shape[0] * 1.6

    new_size = (int(clothing.shape[1] * width_scale), int(clothing.shape[0] * height_scale))
    return cv2.resize(clothing, new_size)

def overlay_clothing(image, clothing, position):
    h, w = clothing.shape[:2]
    x, y = position
    
    if x >= image.shape[1] or y >= image.shape[0] or x + w <= 0 or y + h <= 0:
        print("Clothing position is outside the image bounds")
        return image

    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(image.shape[1], x + w)
    y_end = min(image.shape[0], y + h)

    clothing_x_start = x_start - x
    clothing_y_start = y_start - y
    clothing_x_end = clothing_x_start + (x_end - x_start)
    clothing_y_end = clothing_y_start + (y_end - y_start)

    alpha_s = clothing[clothing_y_start:clothing_y_end, clothing_x_start:clothing_x_end, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        image[y_start:y_end, x_start:x_end, c] = (alpha_s * clothing[clothing_y_start:clothing_y_end, clothing_x_start:clothing_x_end, c] +
                                                  alpha_l * image[y_start:y_end, x_start:x_end, c])

    return image

# Start capturing video
cap = cv2.VideoCapture(1)  # Try 0 if 1 doesn't work

with mp_pose.Pose(static_image_mode=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process Pose
        pose_results = pose.process(image_rgb)

        if pose_results.pose_landmarks:
            # Resize clothing based on detected landmarks
            resized_clothing = resize_clothing(image, virtual_clothing, pose_results.pose_landmarks.landmark)

            # Calculate position for the clothing
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            shoulder_x = int((left_shoulder.x + right_shoulder.x) / 2 * image.shape[1])
            shoulder_y = int((left_shoulder.y + right_shoulder.y) / 2 * image.shape[0])

            offset_y = 200
            clothing_position = (shoulder_x - resized_clothing.shape[1] // 2,
                                 shoulder_y - resized_clothing.shape[0] // 2 + offset_y)

            # Overlay the clothing on the image
            image = overlay_clothing(image, resized_clothing, clothing_position)

            # Draw pose landmarks for debugging
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Virtual Try-On', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()