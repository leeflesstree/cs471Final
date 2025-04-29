# === Imports ===
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import optuna

# === Constants and Paths ===
current_file = Path(__file__)
project_root = current_file.parent.parent
BASE_DIR = project_root / "Preprocessing/CleanData2/FeatureData"
TRAIN_DIR = BASE_DIR / "features_train.csv"
TEST_DIR = BASE_DIR / "features_test.csv"

# === Helper Functions ===
def clean_tuple_strings(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: float(str(x).strip('()').split(',')[0]))
    return df

def train_and_evaluate(X_train, y_train, X_test, y_test, description=""):
    model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"\n=== {description} ===")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))
    return train_acc, test_acc

def optuna_objective(trial, X_train, y_train):
    params = {
        'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
        'gamma': trial.suggest_float('gamma', 1e-5, 1e2, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
    }
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)

    model = SVC(class_weight='balanced', random_state=42, **params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# === Data Preparation ===
# Load datasets
train_df = pd.read_csv(TRAIN_DIR)
test_df = pd.read_csv(TEST_DIR)

# Clean features
X_train_raw = clean_tuple_strings(train_df.drop(columns=["image_name", "emotion"]))
X_test_raw = clean_tuple_strings(test_df.drop(columns=["image_name", "emotion"]))
y_train_raw = train_df["emotion"]
y_test_raw = test_df["emotion"]
print("Features loaded and cleaned successfully.")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_raw)
y_test_encoded = label_encoder.transform(y_test_raw)
print("Target labels encoded successfully.")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
print("Features scaled successfully.")

# Resample with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)
print("Features SMOTED successfully.")

# Feature selection
selector = SelectKBest(score_func=f_classif, k='all')
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)
print("Features selected successfully.")

# === Multiple Tests ===
results = {}

# 1. Base - raw data
results["Raw Features + Raw Labels"] = train_and_evaluate(X_train_raw.values, y_train_raw.values, X_test_raw.values, y_test_raw.values, "Raw Features + Raw Labels")

# 2. Base - encoded labels
results["Raw Features + Encoded Labels"] = train_and_evaluate(X_train_raw.values, y_train_encoded, X_test_raw.values, y_test_encoded, "Raw Features + Encoded Labels")

# 3. Scaled features + encoded labels
results["Scaled Features + Encoded Labels"] = train_and_evaluate(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, "Scaled Features + Encoded Labels")

# 4. SMOTE resampled
results["SMOTE Resampled + Scaled + Encoded Labels"] = train_and_evaluate(X_train_resampled, y_train_resampled, X_test_scaled, y_test_encoded, "SMOTE Resampled + Scaled + Encoded Labels")

# 5. SMOTE + Feature Selected
results["SMOTE + Feature Selected"] = train_and_evaluate(X_train_selected, y_train_resampled, X_test_selected, y_test_encoded, "SMOTE + Feature Selected")

# === Summary of Results ===
print("\n=== Summary of Accuracies (Train and Test) ===")
for key, (train_acc, test_acc) in results.items():
    print(f"{key}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# === Optuna Hyperparameter Tuning ===
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: optuna_objective(trial, X_train_selected, y_train_resampled), n_trials=50)

print("\nOptuna Best Trial:")
print(f"  Value: {study.best_trial.value:.4f}")
print("  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# Train best model
best_params = study.best_trial.params
final_model = SVC(class_weight='balanced', random_state=42, **best_params)
final_model.fit(X_train_selected, y_train_resampled)

# Evaluate on test set
y_pred_final = final_model.predict(X_test_selected)
print("\nClassification Report (Optuna Optimized Model):")
print(classification_report(y_test_encoded, y_pred_final))
print(f"Optuna Optimized Test Accuracy: {accuracy_score(y_test_encoded, y_pred_final):.4f}")

# Confusion matrix
cm_final = confusion_matrix(y_test_encoded, y_pred_final)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_final, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Optuna Optimized Model)")
plt.tight_layout()
plt.show()

# Visualize optimization history
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optuna Optimization History')
plt.tight_layout()
plt.show()

optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Optuna Parameter Importances')
plt.tight_layout()
plt.show()
