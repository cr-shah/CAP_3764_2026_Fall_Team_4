import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test(path="../data/processed/dataco_clean.csv"):
    # Load the processed data
    processed_data = pd.read_csv(path, parse_dates=["order date (DateOrders)"])

    # Date features
    processed_data["order_month"]   = processed_data["order date (DateOrders)"].dt.month
    processed_data["order_weekday"] = processed_data["order date (DateOrders)"].dt.dayofweek
    processed_data["order_hour"]    = processed_data["order date (DateOrders)"].dt.hour

    # Sort by time
    processed_data = processed_data.sort_values("order date (DateOrders)")

    # Features and target
    X = processed_data.drop(columns=["Late_delivery_risk", "order date (DateOrders)"])
    y = processed_data["Late_delivery_risk"]

    # Time-based split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Drop high-count text columns
    drop_text = ["Customer Country", "Customer State", "Customer City",
                 "Order Country", "Order State", "Order City", "Product Name",
                 "Customer Zipcode"]
    X_train = X_train.drop(columns=drop_text)
    X_test  = X_test.drop(columns=drop_text)

    # One-hot encode, then make test columns match train
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test  = pd.get_dummies(X_test, drop_first=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test, y_train, y_test