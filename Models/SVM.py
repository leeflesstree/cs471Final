import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pathlib import Path
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Get the project root directory
current_file = Path(__file__)
project_root = current_file.parent.parent

# Base paths
BASE_DIR = project_root / "Preprocessing/CleanData2/FeatureData"
TRAIN_DIR = BASE_DIR / "features_train.csv"
TEST_DIR = BASE_DIR / "features_test.csv"

# Load the data
train_df = pd.read_csv(TRAIN_DIR)
test_df = pd.read_csv(TEST_DIR)

# Clean the data by converting string tuples to float values
def clean_tuple_strings(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: float(str(x).strip('()').split(',')[0]))
    return df

# Clean the feature columns
X_train = clean_tuple_strings(train_df.drop(columns=["image_name", "emotion"]))
X_test = clean_tuple_strings(test_df.drop(columns=["image_name", "emotion"]))
y_train = train_df["emotion"]
y_test = test_df["emotion"]
print("Features loaded and cleaned successfully.")

# Encode target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
print("Target labels encoded successfully.")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")

# Use SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)
print("Features SMOTED successfully.")

# Apply feature selection after SMOTE
selector = SelectKBest(score_func=f_classif, k='all')
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector.transform(X_test_scaled)
print("Features selected successfully.")

# Get feature importance scores
feature_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'Score': selector.scores_
})

# Sort features by importance score
feature_scores = feature_scores.sort_values('Score', ascending=False)

# Plot top 20 most important features
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_scores.head(20), x='Score', y='Feature')
plt.title('Top 20 Most Important Features')
plt.xlabel('F-Score')
plt.ylabel('Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print top 20 features with their scores
print("\nTop 20 Most Important Features:")
print(feature_scores.head(20))
# Grid search for SVM parameters
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_selected, y_train_resampled)

print("Best parameters:", grid.best_params_)
print("Best cross-val accuracy:", grid.best_score_)
best_model = grid.best_estimator_

# Predict
y_pred = best_model.predict(X_test_selected)

# Evaluate
report = classification_report(
    y_test_encoded, y_pred, target_names=label_encoder.classes_, zero_division=0
)
accuracy = accuracy_score(y_test_encoded, y_pred)

print("Classification Report:\n", report)
print("Accuracy:", accuracy)

# Plot confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)
labels = label_encoder.classes_

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Learning curves
train_sizes, train_scores, valid_scores = learning_curve(
    estimator=best_model,
    X=X_train_scaled,
    y=y_train_encoded,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Calculate means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label="Training Accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, valid_mean, 'o-', label="Validation Accuracy")
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.2)
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (SVM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter search space
    params = {
        'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
        'gamma': trial.suggest_float('gamma', 1e-5, 1e2, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
    }
    
    # Optional: Add degree parameter for polynomial kernel
    if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    
    # Create SVM classifier with the suggested parameters
    classifier = SVC(random_state=42, class_weight='balanced', **params)
    
    # Perform cross-validation
    scores = cross_val_score(
        classifier,
        X_train_selected,
        y_train_resampled,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Return the mean accuracy
    return scores.mean()

# Create and run the study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # You can adjust the number of trials

# Print the results
print("Best trial:")
print("  Value: ", study.best_trial.value)
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# Train the final model with the best parameters
best_params = study.best_trial.params
final_model = SVC(random_state=42, class_weight='balanced', **best_params)
final_model.fit(X_train_selected, y_train_resampled)

y_pred1 = final_model.predict(X_train_selected)
# Print classification report
print("\nClassification Report Train Set with best parameters:")
print(classification_report(
    y_train_encoded,
    y_pred1,
    target_names=label_encoder.classes_,
    zero_division=0
))

# Evaluate on test set
y_pred = final_model.predict(X_test_selected)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\nTest Accuracy with best parameters: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(
    y_test_encoded,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

# Visualize the optimization history
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.show()

# Plot parameter importances
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Parameter Importances')
plt.show()

# Plot the parallel coordinate plot for the parameters
plt.figure(figsize=(12, 6))
optuna.visualization.matplotlib.plot_parallel_coordinate(study)
plt.title('Parallel Coordinate Plot')
plt.tight_layout()
plt.show()

# Create confusion matrix with the best model
cm = confusion_matrix(y_test_encoded, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Optuna Optimized Model)")
plt.tight_layout()
plt.show()