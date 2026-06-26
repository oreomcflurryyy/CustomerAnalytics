import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_DIR


def plot_feature_importance(
    model,
    feature_names,
    top_n=20
):
    """
    Plot the most important features.
    """

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    print("\nTop Feature Importances\n")
    print(importance_df.head(top_n))

    plt.figure(figsize=(10,8))

    plt.barh(
        importance_df.head(top_n)["Feature"][::-1],
        importance_df.head(top_n)["Importance"][::-1]
    )

    plt.xlabel("Importance")

    plt.title("Top Feature Importance")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=300
    )

    plt.close()

    return importance_df