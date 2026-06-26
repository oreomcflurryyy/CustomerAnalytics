import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.config import OUTPUT_DIR

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained machine learning model.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC AUC": roc_auc_score(y_test, probabilities)
    }

    return results

def save_results(results):
    """
    Save model comparison results.
    """

    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========\n")

    print(results_df)

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    return results_df