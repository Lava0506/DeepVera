import cv2
import os

# 1. Setup the files
video_path = 'test_video.mp4'
output_folder = 'extracted_frames'

# ==========================================
# THIS IS THE MAGIC LINE YOU ASKED ABOUT!
# It checks if 'extracted_frames' exists. If not, it creates it automatically.
# ==========================================
os.makedirs(output_folder, exist_ok=True)

# 2. Load the video into OpenCV
print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

# 3. Setup our counters
frame_count = 0
saved_count = 0
frame_skip = 5  # We only save every 5th frame so we don't end up with thousands of pictures

# 4. Loop through the video frame by frame
while cap.isOpened():
    success, frame = cap.read()
    
    # If there are no more frames, stop the loop
    if not success:
        break
        
    # If it's the 5th, 10th, 15th frame, etc... save it!
    if frame_count % frame_skip == 0:
        file_name = f"{output_folder}/frame_{saved_count}.jpg"
        cv2.imwrite(file_name, frame)
        saved_count += 1
        
    frame_count += 1

# 5. Clean up
cap.release()
print(f"--- SUCCESS ---")
print(f"Extracted and saved {saved_count} frames to the '{output_folder}' folder.")