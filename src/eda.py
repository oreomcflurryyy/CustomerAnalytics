import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt


def churn_distribution(df):
    """
    Plot the distribution of customer churn.
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

    print(churn_percentages)

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

    plt.show()