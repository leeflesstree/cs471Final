import os

import pandas as pd
import numpy as np
from tqdm import tqdm

# === Feature functions ===

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_eye_features(landmarks):
    left = {
        "outer": landmarks[33],
        "inner": landmarks[133],
        "top": landmarks[159],
        "bottom": landmarks[145]
    }
    right = {
        "outer": landmarks[362],
        "inner": landmarks[263],
        "top": landmarks[386],
        "bottom": landmarks[374]
    }

    features = [
        euclidean(left["outer"], left["inner"]),
        euclidean(left["top"], left["bottom"]),
        euclidean(right["outer"], right["inner"]),
        euclidean(right["top"], right["bottom"]),
        euclidean(left["top"], left["bottom"]) / (euclidean(left["outer"], left["inner"]) + 1e-6),
        euclidean(right["top"], right["bottom"]) / (euclidean(right["outer"], right["inner"]) + 1e-6)
    ]
    return features

def get_eyebrow_features(landmarks):
    left_dist = euclidean(landmarks[105], landmarks[159])
    right_dist = euclidean(landmarks[334], landmarks[386])
    inter_brow = euclidean(landmarks[70], landmarks[300])
    return [left_dist, right_dist, inter_brow]

def get_mouth_features(landmarks):
    features = [
        euclidean(landmarks[61], landmarks[291]),
        euclidean(landmarks[13], landmarks[14]),
        euclidean(landmarks[78], landmarks[308]),
        euclidean(landmarks[82], landmarks[87]),
        euclidean(landmarks[312], landmarks[317]),
    ]
    features.append(features[1] / (features[0] + 1e-6))
    return features

def get_nose_features(landmarks):
    nose_tip = landmarks[1]
    upper_lip = landmarks[13]
    nose_length = euclidean(landmarks[168], landmarks[1])
    nose_to_lip = euclidean(nose_tip, upper_lip)
    return [nose_length, nose_to_lip]

def get_jaw_features(landmarks):
    left_jaw = landmarks[234]
    right_jaw = landmarks[454]
    chin = landmarks[152]
    nose_base = landmarks[1]
    return [
        euclidean(left_jaw, right_jaw),
        euclidean(chin, nose_base)
    ]

def get_face_angles(landmarks):
    # Eye angle using both eyes (outer corners)
    left_eye = landmarks[33]   # Left outer corner
    right_eye = landmarks[362] # Right outer corner
    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]
    angle_eyes = np.arctan2(eye_dy, eye_dx)

    # Mouth angle
    left_mouth = landmarks[61]   # Left corner
    right_mouth = landmarks[291] # Right corner
    mouth_dx = right_mouth[0] - left_mouth[0]
    mouth_dy = right_mouth[1] - left_mouth[1]
    angle_mouth = np.arctan2(mouth_dy, mouth_dx)

    #Convert to degrees
    angle_eyes = np.degrees(angle_eyes)
    angle_mouth = np.degrees(angle_mouth)

    return [angle_eyes, angle_mouth]

def get_symmetry_features(landmarks):
    center_x = landmarks[1][0]
    pairs = [(33, 263), (61, 291), (159, 386), (105, 334)]
    symmetry = []
    for l, r in pairs:
        l_x = landmarks[l][0]
        r_x = landmarks[r][0]
        symmetry.append(abs((l_x - center_x) - (center_x - r_x)))
    return symmetry

def extract_features(landmarks):
    normalizer = euclidean(landmarks[33], landmarks[263]) + 1e-6
    features = (
            [f / normalizer for f in get_eye_features(landmarks)] +
            [f / normalizer for f in get_eyebrow_features(landmarks)] +
            [f / normalizer for f in get_mouth_features(landmarks)] +
            [f / normalizer for f in get_nose_features(landmarks)] +
            [f / normalizer for f in get_jaw_features(landmarks)] +
            get_face_angles(landmarks) +  # angles are already scale-invariant
            get_symmetry_features(landmarks)
    )
    return features


def main():

    # Prepare data structures with descriptive feature names
    feature_columns = ['image_name'] + [
        # Eye features (6)
        'left_eye_width',
        'left_eye_height',
        'right_eye_width',
        'right_eye_height',
        'left_eye_aspect_ratio',
        'right_eye_aspect_ratio',
        
        # Eyebrow features (3)
        'left_eyebrow_eye_distance',
        'right_eyebrow_eye_distance',
        'inter_eyebrow_distance',
        
        # Mouth features (6)
        'mouth_width',
        'mouth_height',
        'mouth_inner_width',
        'left_lip_height',
        'right_lip_height',
        'mouth_aspect_ratio',
        
        # Nose features (2)
        'nose_length',
        'nose_to_lip_distance',
        
        # Jaw features (2)
        'jaw_width',
        'chin_to_nose_distance',
        
        # Face angles (2)
        'eye_line_angle',
        'mouth_line_angle',
        
        # Symmetry features (4)
        'eye_symmetry',
        'mouth_symmetry',
        'eyebrow_symmetry',
        'facial_symmetry'
    ] + ['emotion']

    # Read the landmarks CSV files
    train_df = pd.read_csv('CleanData2/LandmarkData/landmarks_train.csv')
    test_df = pd.read_csv('CleanData2/LandmarkData/landmarks_test.csv')

    for df, output_file in [(train_df, 'CleanData2/FeatureData/features_train.csv'),
                            (test_df, 'CleanData2/FeatureData/features_test.csv')]:

        # Process each row
        feature_data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Extracting features for {output_file}'):
            # Convert landmarks to required format
            landmarks = []
            for i in range(468):
                landmarks.append([row[f'x{i}'], row[f'y{i}']])

            # Extract features
            features = extract_features(landmarks)

            # Store results
            feature_data.append([row['image_name']] + features + [row['emotion']])

        # Save features to CSV
        features_df = pd.DataFrame(feature_data, columns=feature_columns)

        # Create directory if it doesn't exist
        os.makedirs('CleanData2/FeatureData', exist_ok=True)

        features_df.to_csv(output_file, index=False)
        print(f"Features extracted and saved to {output_file}")


def validate_features(features, feature_columns=None):
    expected_length = 25
    if len(features) != expected_length:
        raise ValueError(
            f"Expected {expected_length} features but got {len(features)}. "
            "Feature list should contain:\n" + 
            "\n".join(f"- {col}" for col in feature_columns[1:-1])  # Skip image_name and emotion
        )

if __name__ == "__main__":
    main()