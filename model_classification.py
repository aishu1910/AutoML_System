from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
models = {
    "Logistic Regression": {"model": LogisticRegression(), "scale": True},
    "SVC": {"model": SVC(), "scale": True},
    "KNN": {"model": KNeighborsClassifier(), "scale": True},
    "Decision Tree": {"model": DecisionTreeClassifier(), "scale": False},
    "Random Forest": {"model": RandomForestClassifier(), "scale": False},
    "Gradient Boosting": {"model": GradientBoostingClassifier(), "scale": False},
    "XGBoost": {"model": XGBClassifier(), "scale": False}
}

def Evaluate(X_train, X_test, y_train, y_test):
    all_results = []

    for name, info in models.items():
        model = info["model"]
        needs_scaling = info["scale"]

        # Pipeline
        if needs_scaling:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
        else:
            pipeline = Pipeline([
                ("model", model)
            ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

        all_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

    return all_results

def tune_model(model_name, model, X_train, y_train, scale):

    # pipeline
    if scale:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
    else:
        pipeline = Pipeline([
            ("model", model)
        ])

    param_dist = {
        "Logistic Regression": {
            "model__C": np.logspace(-3, 3, 50),
            "model__penalty": ["l2"]
        },

        "SVC": {
            "model__C": np.logspace(-2, 2, 50),
            "model__kernel": ["rbf", "linear"]
        },

        "KNN": {
            "model__n_neighbors": np.arange(3, 20),
            "model__weights": ["uniform", "distance"]
        },

        "Decision Tree": {
            "model__max_depth": [None] + list(np.arange(3, 20)),
            "model__min_samples_split": np.arange(2, 10)
        },

        "Random Forest": {
            "model__n_estimators": np.arange(100, 500),
            "model__max_depth": [None] + list(np.arange(5, 20))
        },

        "Gradient Boosting": {
            "model__n_estimators": np.arange(100, 500),
            "model__learning_rate": np.linspace(0.01, 0.2, 50)
        },

        "XGBoost": {
            "model__n_estimators": np.arange(100, 500),
            "model__learning_rate": np.linspace(0.01, 0.2, 50),
            "model__max_depth": np.arange(3, 10)
        }
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist[model_name],
        n_iter=30,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_score_