import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=2,  # Heavy model
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Input and output video paths
input_video_path = '../videos_test_sequences/TS1.mp4'
output_video_path = '../videos_test_sequences/TS1_Pred.mp4'

# Open the video file
cap = cv2.VideoCapture(input_video_path)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing video: {input_video_path}")
print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and detect pose
    results = pose.process(image)

    # Convert back to BGR for rendering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # Write the frame to the output video
    out.write(image)

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processed {frame_count}/{total_frames} frames")


cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()

print(f"Output video saved as: {output_video_path}")