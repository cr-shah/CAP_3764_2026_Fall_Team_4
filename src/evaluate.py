from src.prepare_data import get_train_test

X_train, X_test, y_train, y_test = get_train_test()

print("Train:", X_train.shape)
print("Test:", X_test.shape)
print(y_train.value_counts())
print(X_train.columns.tolist())