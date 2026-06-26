from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost using GridSearchCV.
    """

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    param_grid = {

        "n_estimators": [100, 200, 300],

        "max_depth": [3, 5, 7],

        "learning_rate": [0.01, 0.05, 0.1],

        "subsample": [0.8, 1.0],

        "colsample_bytree": [0.8, 1.0]

    }

    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        cv=5,

        scoring="roc_auc",

        n_jobs=-1,

        verbose=2

    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters")

    print(grid.best_params_)

    print("\nBest ROC AUC")

    print(grid.best_score_)

    return grid.best_estimator_