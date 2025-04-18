import cv2
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm

# Init MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Define emotion classes (make sure they match your folders)
EMOTIONS = ['disgust']

# Initialize dataframe to store features
data = []
# Add 'image_name' to columns
columns = ['image_name'] + [f'x{i}' for i in range(468)] + [f'y{i}' for i in range(468)] + ['emotion']

# Set your dataset path
DATASET_PATH = 'FER 2013\\train'

for emotion in EMOTIONS:
    folder_path = os.path.join(DATASET_PATH, emotion)
    for img_file in tqdm(os.listdir(folder_path), desc=f'Processing {emotion}'):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]

            # Add image filename to the data
            data.append([img_file] + x_coords + y_coords + [emotion])
        else:
            print(f'No face detected in: {img_path}')

# Save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv('landmarks_fer2013.csv', index=False)
print("✅ Feature extraction complete! Data saved to landmarks_fer2013.csv")