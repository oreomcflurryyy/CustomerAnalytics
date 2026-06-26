from src.preprocessing import (
    load_data,
    inspect_data,
    check_data_quality,
    clean_data
)
from src.eda import churn_distribution

df = load_data()

inspect_data(df)

check_data_quality(df)

df = clean_data(df)

print(df.shape)

churn_distribution(df)