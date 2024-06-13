import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import os
from dotenv import load_dotenv
import time
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'))

st.title('Yoga Pose Estimation')


# Load yoga poses steps from the text file
def load_yoga_poses(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n\n')
    yoga_poses = {}
    for pose in data:
        lines = pose.split('\n')
        pose_name = lines[0]
        steps = lines[1:]
        yoga_poses[pose_name] = steps
    return yoga_poses


yoga_poses = load_yoga_poses('yoga_guide.txt')

# Create a dropdown for selecting the yoga pose
selected_pose = st.sidebar.selectbox("Choose a Yoga Pose", list(yoga_poses.keys()))

# Display the selected yoga pose steps
st.sidebar.header(f"Steps for {selected_pose}")
for step in yoga_poses[selected_pose]:
    st.sidebar.write(step)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def detectPose(frame, pose):
    output_frame = frame.copy()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    height, width, _ = frame.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_frame, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

    return output_frame, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


def classifyPose(landmarks, output_frame, display=False):
    label = 'Unknown Pose'
    color = (0, 0, 255)

    try:
        left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

        right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

        left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        keypoints_angle = {"left_elbow_angle": left_elbow_angle, "right_elbow_angle": right_elbow_angle,
                           "left_shoulder_angle": left_shoulder_angle, "right_shoulder_angle": right_shoulder_angle,
                           "left_knee_angle": left_knee_angle, "right_knee_angle": right_knee_angle,
                           "left_hip_angle": left_hip_angle, "right_hip_angle": right_hip_angle}


    except IndexError as e:
        print(f"Error: {e}")
        return output_frame, label

    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 200:
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                if left_knee_angle > 190 and left_knee_angle < 260 or right_knee_angle > 90 and right_knee_angle < 140:
                    label = 'Warrior II Pose'
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                label = 'T Pose'
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
            label = 'Tree Pose'
    if (left_knee_angle > 10 and left_knee_angle < 25 and right_knee_angle > 10 and right_knee_angle < 25) or (
            left_knee_angle > 300 and left_knee_angle < 380 and right_knee_angle > 300 and right_knee_angle < 380):
        if (left_hip_angle > 260 and left_hip_angle < 350 and right_hip_angle > 260 and right_hip_angle < 295) or (
                right_hip_angle > 50 and right_hip_angle < 75 and left_hip_angle > 90 and left_hip_angle < 120):
            label = 'Vajrasana'

    if label != 'Unknown Pose':
        color = (0, 255, 0)
    cv2.putText(output_frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 5)

    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_frame[:, :, ::-1])
        plt.title("Output Frame")
        plt.axis('off')
    else:
        return output_frame, label, keypoints_angle


def get_corrections_from_gpt4(pose_name, keypoints_angle):
    prompt = f"You are an expert in yoga pose corrections. The user is trying to perform the {pose_name}.\n\n"
    prompt += "Here are the angles of their keypoints:\n"
    for keypoint, angle in keypoints_angle.items():
        prompt += f"{keypoint}: {angle:.2f} degrees\n"
    prompt += "\nPlease provide detailed corrections for this pose based on the given angles."

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": f"You are an expert in yoga pose corrections. The user is trying to perform the {pose_name}.\n\n"},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


input_method = st.sidebar.selectbox("Choose input method", ("Upload Image", "Use Webcam"))

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        output_frame, landmarks = detectPose(image, pose)
        if landmarks:
            output_frame, label, vector_keypoints = classifyPose(landmarks, output_frame, display=False)
            st.write(f"Detected Yoga Pose: {label}")
        st.image(output_frame, channels="BGR")
else:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    run = st.checkbox('Run Webcam')

    start_time = time.time()
    pose_start_time = None
    target_duration = 2  # seconds
    max_duration = 30  # seconds
    vector_store = {}

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break

        output_frame, landmarks = detectPose(frame, pose)
        if landmarks and (time.time() - start_time < max_duration):
            output_frame, label, keypoint_angle = classifyPose(landmarks, output_frame, display=False)
            if label == selected_pose:
                if pose_start_time is None:
                    pose_start_time = time.time()
                elif time.time() - pose_start_time >= target_duration:
                    st.write(f"Congratulations! You've held the {label} for {target_duration} seconds.")
                    run = False
            else:
                vector_store.update(keypoint_angle)
                pose_start_time = None
        else:
            st.write("You are not able to achieve to yoga pose")
            if vector_store:
                corrections = get_corrections_from_gpt4(selected_pose, vector_store)
                st.write(f"Corrections from GPT-4: {corrections}")
            break

        stframe.image(output_frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f"vector::{vector_store}")
    cap.release()
    cv2.destroyAllWindows()
