import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
import ultralytics
os.chdir('P:\\Collision-Alert-System\\prediction_service\\yolov7\\')
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
#from sounds import play_alert_sound
print(os.getcwd())
# Change to the yolov7 directory

# Load the YOLOv7 model
weights_path = '../best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path, map_location=device)
model.eval()

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
if uploaded_file is not None:
    video_path = uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(video_path)

    # Define polygon coordinates
    polygon_points = np.array([(380, 100), (380, 590), (480, 100), (480, 590)], np.int32)

    # Create a temporary file to save processed video
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

    stframe = st.empty()  # Create a placeholder for the video display

    with st.spinner('Processing video...'):
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Pre-process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame_rgb, (640, 480))
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).to(device).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            # Run inference
            with torch.no_grad():
                pred = model(img_tensor)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.5, 0.45)

            # Check bounding boxes against the polygon
            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                        inside_polygon = any(cv2.pointPolygonTest(polygon_points, corner, False) >= 0 for corner in corners)

                        if inside_polygon:
                            cv2.putText(frame, "Alert!", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            #play_alert_sound()
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Write the processed frame to the output video
            out.write(frame)

            # Display the processed frame in real-time
            stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    out.release()

    # Display the processed video at the end
    st.video(output_path)
