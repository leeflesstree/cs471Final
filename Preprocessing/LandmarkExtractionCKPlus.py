import os

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

def process_dataset(base_path, emotions):
    """
    Process images from the specified path and extract landmarks
    
    Args:
        base_path (str): Path to the CK+ dataset folder
        emotions (list): List of emotion categories to process
    """
    # Init MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize dataframe to store features
    data = []
    columns = ['image_name'] + [f'x{i}' for i in range(468)] + [f'y{i}' for i in range(468)] + ['emotion']

    base_path = Path(base_path)
    
    for emotion in emotions:
        folder_path = base_path / emotion
        if not folder_path.exists():
            print(f"⚠Warning: Folder not found for emotion {emotion}")
            continue

        for img_file in tqdm(list(folder_path.glob('*.png')), desc=f'Processing {emotion}'):
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read image {img_file}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                x_coords = [lm.x for lm in landmarks.landmark]
                y_coords = [lm.y for lm in landmarks.landmark]

                data.append([img_file.name] + x_coords + y_coords + [emotion])
            else:
                print(f'\nNo face detected in: {img_file}')

    # Create DataFrame
    if data:
        df = pd.DataFrame(data, columns=columns)
        return df
    else:
        print("No data was processed!")
        return None

def main():
    # Define emotion classes for CK+
    EMOTIONS = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

    # Get the project root directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent

    # Base path for CK+ dataset
    BASE_DIR = project_root / "CK+"

    # Output files
    os.makedirs("CleanData2/LandmarkData", exist_ok=True)
    TRAIN_CSV = "CleanData2/LandmarkData/landmarks_train.csv"
    TEST_CSV = "CleanData2/LandmarkData/landmarks_test.csv"

    # Process all data
    print("Processing CK+ dataset...")
    df = process_dataset(BASE_DIR, EMOTIONS)
    
    if df is not None:
        # Split into train and test sets (80/20)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
        
        # Save to CSV
        train_df.to_csv(TRAIN_CSV, index=False)
        test_df.to_csv(TEST_CSV, index=False)
        
        print(f"Feature extraction complete!")
        print(f"Training set saved to {TRAIN_CSV} with {len(train_df)} samples")
        print(f"Test set saved to {TEST_CSV} with {len(test_df)} samples")

if __name__ == "__main__":
    main()