from model_utils import load_dataset, train_models
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = load_dataset()

# Train models
lr_model, rf_model, X_test, y_test = train_models(df)

# Predictions
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Evaluation
print("=== Linear Regression ===")
print("R2 Score:", r2_score(y_test, lr_preds))
print("MAE:", mean_absolute_error(y_test, lr_preds))

print("\n=== Random Forest ===")
print("R2 Score:", r2_score(y_test, rf_preds))
print("MAE:", mean_absolute_error(y_test, rf_preds))

