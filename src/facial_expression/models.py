"""Classical model selection with all learned preprocessing inside each fold."""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_search(name: str, seed: int = 42, quick: bool = False, grouped: bool = False):
    if name == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=seed)
        grid = {
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [3, 6, 10, None],
            "model__min_samples_leaf": [1, 3, 5],
            "smote": ["passthrough", SMOTE(random_state=seed)],
        }
    elif name == "svm":
        classifier = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
        grid = {
            "model__C": [0.1, 1, 10, 50, 100],
            "model__gamma": ["scale", 0.001, 0.01, 0.1],
            "smote": ["passthrough", SMOTE(random_state=seed)],
        }
    else:
        raise ValueError(f"Unknown model: {name}")
    if quick:
        grid = {key: values[:1] for key, values in grid.items()}
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("smote", "passthrough"),
        ("model", classifier),
    ])
    splitter = StratifiedGroupKFold if grouped else StratifiedKFold
    return GridSearchCV(
        pipeline, grid, scoring="f1_macro",
        cv=splitter(n_splits=2 if quick else 5, shuffle=True, random_state=seed),
        n_jobs=1, error_score="raise", return_train_score=False,
    )
