import shap
import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_DIR


def shap_summary_plot(model, X_train_processed, feature_names):
    """
    Generate a SHAP summary plot.
    """

    print("\nGenerating SHAP Summary Plot...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_train_processed)

    X_train_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_train_df,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✓ Saved SHAP Summary")


def shap_bar_plot(model, X_train_processed, feature_names):
    """
    Generate a SHAP feature importance bar plot.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_train_processed)

    X_train_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_train_df,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✓ Saved SHAP Bar Plot")