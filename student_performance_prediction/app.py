import streamlit as st
import pandas as pd
from model_utils import load_dataset, train_models
from sklearn.metrics import r2_score
import numpy as np

# Load and train models
df = load_dataset()
lr_model, rf_model, X_test, y_test, scaler = train_models(df)

st.title("Student Final Marks Prediction (Scaled)")

st.write("Enter attendance percentage, Mid-1 and Mid-2 marks to predict final marks (out of 100).")

attend = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
mid1 = st.number_input("Mid-1 Marks (out of 35)", min_value=0, max_value=35, value=20)
mid2 = st.number_input("Mid-2 Marks (out of 35)", min_value=0, max_value=35, value=22)

if st.button("Predict Final Marks"):
    # Prepare input
    input_df = pd.DataFrame([[attend, mid1, mid2]], columns=['Attend', 'Mid1', 'Mid2'])

    # Predictions (scaled)
    lr_scaled = lr_model.predict(input_df)[0]
    rf_scaled = rf_model.predict(input_df)[0]

    # Convert back to 0-100 range
    lr_pred = scaler.inverse_transform([[lr_scaled]])[0][0]
    rf_pred = scaler.inverse_transform([[rf_scaled]])[0][0]

    # Clip to safe range
    lr_pred = np.clip(lr_pred, 0, 100)
    rf_pred = np.clip(rf_pred, 0, 100)

    st.subheader("Predicted Final Marks")
    st.write(f"**Linear Regression:** {lr_pred:.2f}")
    st.write(f"**Random Forest:** {rf_pred:.2f}")

st.write("---")
st.write("**Model Accuracy on Test Set:**")
st.write(f"Linear Regression R²: {r2_score(y_test, lr_model.predict(X_test)):.2f}")
st.write(f"Random Forest R²: {r2_score(y_test, rf_model.predict(X_test)):.2f}")
