import cv2
import os


video_path = 'test_video.mp4'
output_folder = 'extracted_frames'

os.makedirs(output_folder, exist_ok=True)


print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)


frame_count = 0
saved_count = 0
frame_skip = 5  


while cap.isOpened():
    success, frame = cap.read()
    
   
    if not success:
        break
        
   
    if frame_count % frame_skip == 0:
        file_name = f"{output_folder}/frame_{saved_count}.jpg"
        cv2.imwrite(file_name, frame)
        saved_count += 1
        
    frame_count += 1


cap.release()
print(f"SUCCESSSSSSS:)")
print(f"Extracted and saved {saved_count} frames to the '{output_folder}' folder.")