import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_DIR

def churn_distribution(df):
    """
    Plot the overall customer churn distribution.
    """

    churn_counts = df["Churn"].value_counts()

    churn_percentages = (
        df["Churn"]
        .value_counts(normalize=True)
        * 100
    )

    print("\nCustomer Churn Distribution\n")
    print(churn_counts)

    print("\nPercentages\n")
    print(churn_percentages.round(2))

    plt.figure(figsize=(6,5))

    plt.bar(
        churn_counts.index,
        churn_counts.values
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")

    for i, value in enumerate(churn_counts.values):
        plt.text(
            i,
            value + 50,
            str(value),
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

def plot_categorical_feature(df, feature):
    """
    Analyze a categorical feature against churn.
    """

    table = pd.crosstab(
        df[feature],
        df["Churn"],
        normalize="index"
    ) * 100

    print("\n" + "=" * 70)
    print(f"{feature.upper()}")
    print("=" * 70)

    print(table.round(2))

    # Highest churn category
    highest = table["Yes"].idxmax()
    highest_rate = table["Yes"].max()

    print(
        f"\nHighest churn: {highest} ({highest_rate:.2f}%)"
    )

    table.plot(
        kind="bar",
        stacked=True,
        figsize=(8,5)
    )

    plt.title(f"{feature} vs Churn")

    plt.ylabel("Percentage")

    plt.tight_layout()

    filename = (
        feature.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    plt.savefig(
        OUTPUT_DIR / f"{filename}_vs_churn.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

def analyze_all_categorical_features(df):
    """
    Analyze all categorical features against churn.
    """

    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    for feature in categorical_features:
        plot_categorical_feature(df, feature)