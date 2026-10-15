# === Imports ===
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# === Paths ===
current_file = Path(__file__)
project_root = current_file.parent.parent
BASE_DIR = project_root / "Preprocessing/CleanData3/FeatureData"
TRAIN_DIR = BASE_DIR / "features_train.csv"
VAL_DIR = BASE_DIR / "features_val.csv"
TEST_DIR = BASE_DIR / "features_test.csv"


# === Helper Functions ===
def clean_tuple_strings(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: float(str(x).strip('()').split(',')[0]))
    return df


def full_evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, description=""):
    print(f"\n{'='*80}")
    print(f"=== MODEL EVALUATION: {description} ===")
    print(f"{'='*80}")

    # Training set
    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"\n{'-'*40}")
    print(f"TRAINING SET EVALUATION")
    print(f"{'-'*40}")
    print(f"Accuracy: {train_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred_train, target_names=label_encoder.classes_, zero_division=0))

    plt.figure(figsize=(10, 8))
    cm_train = confusion_matrix(y_train, y_pred_train)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix (Training Set) - {description}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # Validation set
    y_pred_val = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    print(f"\n{'-'*40}")
    print(f"VALIDATION SET EVALUATION")
    print(f"{'-'*40}")
    print(f"Accuracy: {val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val, target_names=label_encoder.classes_, zero_division=0))

    plt.figure(figsize=(10, 8))
    cm_val = confusion_matrix(y_val, y_pred_val)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix (Validation Set) - {description}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # Test set
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"\n{'-'*40}")
    print(f"TEST SET EVALUATION")
    print(f"{'-'*40}")
    print(f"Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, zero_division=0))

    plt.figure(figsize=(10, 8))
    cm_test = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix (Test Set) - {description}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # Summary
    print(f"\n{'-'*40}")
    print(f"SUMMARY OF RESULTS FOR {description}")
    print(f"{'-'*40}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return train_acc, val_acc, test_acc


# === Data Loading and Preprocessing ===
print(f"\n{'='*80}")
print(f"=== DATA LOADING AND PREPROCESSING ===")
print(f"{'='*80}")

# --- Load Data ---
print(f"\n{'-'*40}")
print(f"LOADING DATA")
print(f"{'-'*40}")
print(f"Loading data from:")
print(f"  Train: {TRAIN_DIR}")
print(f"  Validation: {VAL_DIR}")
print(f"  Test: {TEST_DIR}")

train_df = pd.read_csv(TRAIN_DIR)
val_df = pd.read_csv(VAL_DIR)
test_df = pd.read_csv(TEST_DIR)

print(f"Data loaded successfully.")
print(f"  Train samples: {len(train_df)}")
print(f"  Validation samples: {len(val_df)}")
print(f"  Test samples: {len(test_df)}")

# --- Feature Extraction ---
print(f"\n{'-'*40}")
print(f"EXTRACTING FEATURES AND LABELS")
print(f"{'-'*40}")
X_train_raw = clean_tuple_strings(train_df.drop(columns=["image_name", "emotion"]))
X_val_raw = clean_tuple_strings(val_df.drop(columns=["image_name", "emotion"]))
X_test_raw = clean_tuple_strings(test_df.drop(columns=["image_name", "emotion"]))
y_train_raw = train_df["emotion"]
y_val_raw = val_df["emotion"]
y_test_raw = test_df["emotion"]
print(f"Features extracted successfully.")
print(f"  Number of features: {X_train_raw.shape[1]}")

# --- Label Encoding ---
print(f"\n{'-'*40}")
print(f"ENCODING LABELS")
print(f"{'-'*40}")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_raw)
y_val_encoded = label_encoder.transform(y_val_raw)
y_test_encoded = label_encoder.transform(y_test_raw)
print(f"Labels encoded successfully.")
print(f"  Classes: {label_encoder.classes_}")
print(f"  Class distribution in training set:")
for i, cls in enumerate(label_encoder.classes_):
    count = (y_train_encoded == i).sum()
    print(f"    {cls}: {count} samples ({count/len(y_train_encoded)*100:.1f}%)")

# --- Feature Scaling ---
print(f"\n{'-'*40}")
print(f"SCALING FEATURES")
print(f"{'-'*40}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)
print(f"Features scaled successfully using StandardScaler.")

# --- SMOTE Resampling ---
print(f"\n{'-'*40}")
print(f"APPLYING SMOTE RESAMPLING")
print(f"{'-'*40}")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)
print(f"SMOTE resampling applied successfully.")
print(f"  Original training samples: {len(X_train_scaled)}")
print(f"  Resampled training samples: {len(X_train_resampled)}")
print(f"  Class distribution after resampling:")
for i, cls in enumerate(label_encoder.classes_):
    count = (y_train_resampled == i).sum()
    print(f"    {cls}: {count} samples ({count/len(y_train_resampled)*100:.1f}%)")

# --- Feature Selection ---
print(f"\n{'-'*40}")
print(f"SELECTING FEATURES")
print(f"{'-'*40}")
selector = SelectKBest(score_func=f_classif, k='all')
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)
print(f"Feature selection applied successfully using SelectKBest with f_classif.")
print(f"  All features retained for analysis (k='all').")


# === Evaluating Different Preprocessing Variants ===
print(f"\n{'='*80}")
print(f"=== EVALUATING DIFFERENT PREPROCESSING VARIANTS ===")
print(f"{'='*80}")
print("Evaluating SVM model performance with different preprocessing steps...")
print("For each variant, we'll evaluate on train, validation, and test sets.")

def run_preprocessing_variant(X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, description):
    print(f"\n{'-'*40}")
    print(f"PREPROCESSING VARIANT: {description}")
    print(f"{'-'*40}")
    print(f"Training SVM model with {description}...")
    model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    full_evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, description)


# 1. Raw features
print(f"\n{'='*80}")
print(f"=== VARIANT 1: RAW FEATURES ===")
print(f"{'='*80}")
run_preprocessing_variant(X_train_raw.values, y_train_encoded, X_val_raw.values, y_val_encoded, X_test_raw.values,
                          y_test_encoded, label_encoder, "Raw Features")

# 2. Scaled features
print(f"\n{'='*80}")
print(f"=== VARIANT 2: SCALED FEATURES ===")
print(f"{'='*80}")
run_preprocessing_variant(X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded, X_test_scaled, y_test_encoded,
                          label_encoder, "Scaled Features")

# 3. SMOTE resampled
print(f"\n{'='*80}")
print(f"=== VARIANT 3: SMOTE RESAMPLED ===")
print(f"{'='*80}")
run_preprocessing_variant(X_train_resampled, y_train_resampled, X_val_scaled, y_val_encoded, X_test_scaled,
                          y_test_encoded, label_encoder, "SMOTE Resampled")

# 4. SMOTE + Feature Selected
print(f"\n{'='*80}")
print(f"=== VARIANT 4: SMOTE + FEATURE SELECTED ===")
print(f"{'='*80}")
run_preprocessing_variant(X_train_selected, y_train_resampled, X_val_selected, y_val_encoded, X_test_selected,
                          y_test_encoded, label_encoder, "SMOTE + Feature Selected")


# === Hyperparameter Optimization with GridSearchCV ===
print(f"\n{'='*80}")
print(f"=== HYPERPARAMETER OPTIMIZATION WITH GRIDSEARCHCV ===")
print(f"{'='*80}")

from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Define parameter grid
param_grid = [
    {
        'kernel': ['poly'],
        'C': [50, 100, 200, 500],
        'degree': [2, 3],
        'gamma': ['scale', 0.0001, 0.001, 0.01],
        'coef0': [0, 0.1, 0.5, 1]
    },
    {
        'kernel': ['rbf'],
        'C': [10, 50, 100, 200],
        'gamma': ['scale', 0.0001, 0.001, 0.01]
    }
]


# Define SVC with balanced class weight
svc = SVC(class_weight='balanced', random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# GridSearchCV (note: we use all CPU cores with n_jobs=-1)
grid_search = GridSearchCV(
    svc,
    param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    refit=True
)

print("Starting GridSearchCV... This may take some time.")
grid_search.fit(X_train_selected, y_train_resampled)

# Print best parameters and best score
print(f"\n{'-'*40}")
print(f"GRIDSEARCHCV RESULTS")
print(f"{'-'*40}")
print(f"Best Validation Accuracy (CV average): {grid_search.best_score_:.4f}")
print("\nBest Parameters:")
for key, value in grid_search.best_params_.items():
    print(f"  {key}: {value}")

# Evaluate the final model (GridSearchCV already refit on full training data)
final_model = grid_search.best_estimator_

# Evaluate the final model on all datasets
print(f"\n{'-'*40}")
print(f"EVALUATING FINAL MODEL")
print(f"{'-'*40}")
train_acc, val_acc, test_acc = full_evaluation(
    final_model, X_train_selected, y_train_resampled,
    X_val_selected, y_val_encoded,
    X_test_selected, y_test_encoded,
    label_encoder, "GridSearchCV Best Model"
)


# === Feature Importance Analysis ===
print(f"\n{'='*80}")
print(f"=== FEATURE IMPORTANCE ANALYSIS ===")
print(f"{'='*80}")
print("Analyzing feature importance for the best model...")
print(f"Model kernel: {final_model.kernel}")
print("Data used: SMOTE + Feature Selected (fully preprocessed)")
feature_names = X_train_raw.columns.tolist()

# For linear kernel, we can directly use coefficients
if final_model.kernel == 'linear':
    # Get coefficients directly from final_model
    if hasattr(final_model, 'coef_'):
        coefs = final_model.coef_
        # For multi-class, average the absolute coefficients across classes
        if coefs.ndim > 1:
            importance = np.abs(coefs).mean(axis=0)
        else:
            importance = np.abs(coefs)

        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Feature Importance (Linear SVM Coefficients)')
        plt.tight_layout()
        plt.show()

        print("Top 10 Most Important Features:")
        print(feature_importance.head(10))
    else:
        print("No coefficients available for feature importance analysis.")
else:
    # For non-linear kernels, use permutation importance
    from sklearn.inspection import permutation_importance

    perm_importance = permutation_importance(final_model, X_val_selected, y_val_encoded,
                                             n_repeats=10, random_state=42)

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance (Permutation Importance)')
    plt.tight_layout()
    plt.show()

    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))

    # Also show SelectKBest scores for comparison
    print("\nSelectKBest F-scores for comparison:")
    selectk_importance = pd.DataFrame({
        'Feature': feature_names,
        'F-score': selector.scores_
    }).sort_values('F-score', ascending=False)
    print(selectk_importance.head(10))

# === Pie Chart of Resampled Training Data ===
print(f"\n{'='*80}")
print(f"=== PIE CHART: RESAMPLED TRAINING DATA DISTRIBUTION ===")
print(f"{'='*80}")