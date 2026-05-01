from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
models = {
    "Ridge": {"model": Ridge(), "scale": True},
    "Lasso": {"model": Lasso(), "scale": True},
    "SVR": {"model": SVR(), "scale": True},
    "KNN": {"model": KNeighborsRegressor(), "scale": True},
    "Random Forest": {"model": RandomForestRegressor(), "scale": False},
    "Gradient Boosting": {"model": GradientBoostingRegressor(), "scale": False},
    "XGBoost": {"model": XGBRegressor(), "scale": False}
}

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def Evaluate(X_train, X_test, y_train, y_test):
    all_results = []

    for name, info in models.items():
        model = info["model"]
        needs_scaling = info["scale"]

        # Build pipeline
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

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        all_results.append({
            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
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

    # parameter distributions
    param_dist = {
        "Linear Regression": {},  # no tuning needed

        "Ridge": {
            "model__alpha": np.logspace(-3, 3, 50)
        },

        "Lasso": {
            "model__alpha": np.logspace(-3, 3, 50)
        },

        "SVR": {
            "model__C": np.logspace(-2, 2, 50),
            "model__epsilon": np.linspace(0.01, 1, 50),
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

    # skip tuning if no params
    if model_name == "Linear Regression":
        pipeline.fit(X_train, y_train)
        return pipeline, {}, pipeline.score(X_train, y_train)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist[model_name],
        n_iter=30,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_score_