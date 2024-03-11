import torch
import numpy as np
import cv2
from PIL import Image

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='path/to/weights.pt', force_reload=True)

# Define the image path
image_path = 'path/to/image.jpg'

# Load the image and convert it to a PIL image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)

# Run inference on the image
results = model(image)

# Display the results
for result in results.xyxy[0]:
    if result[-1] == 0:
        continue
    x1, y1, x2, y2, conf, class_id = result
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.putText(image, f'face {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Display the image
cv2.imshow('image', np.array(image))
cv2.waitKey(0)

# Define the video path
video_path = 'path/to/video.mp4'

# Load the video
cap = cv2.VideoCapture(video_path)

# Define the output video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define the output video file name
out_path = 'output.mp4'

# Get the video frame rate and dimensions
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the output video writer
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Process the video frame by frame
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Run inference on the frame
    results = model(frame)

    # Display the results
    for result in results.xyxy[0]:
        if result[-1] == 0:
            continue
        x1, y1, x2, y2, conf, class_id = result
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, f'face {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Write the frame to the output video
    out.write(np.array(frame))

    # Display the frame
    cv2.imshow('frame', np.array(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()