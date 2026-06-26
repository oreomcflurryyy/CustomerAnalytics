import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from src.config import OUTPUT_DIR


def save_evaluation_plots(
    model,
    X_test,
    y_test,
    model_name
):
    """
    Save evaluation plots.
    """

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{model_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ROC Curve
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{model_name}_roc_curve.png",
        dpi=300
    )

    plt.close()

    # Precision Recall
    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{model_name}_pr_curve.png",
        dpi=300
    )

    plt.close()

    print(f"Saved evaluation plots for {model_name}")