# STUDENTS.MARKS-PREDICTION.
## **Student Final Marks Prediction**
This project predicts a student's **Final Marks (0–100)** based on:
* **Attendance (%)**
* **Mid-1 Marks (out of 35)**
* **Mid-2 Marks (out of 35)**
It uses **Linear Regression (Polynomial + Ridge)** and **Random Forest** models trained on a scaled target to ensure predictions remain within **0–100**.
### **Features**
* **Training**
  * Target scaling (0–1) to constrain outputs
  * Polynomial features with Ridge regularization for linear model
  * Random Forest Regressor for non-linear mapping
* **Evaluation**
  * R² Score and MAE displayed
* **Interactive Prediction**
  * Streamlit web app: input values → get predictions from both models
* **Model Persistence**
  * Saves models and target scaler using `joblib`
### **Project Structure**

student_performance_prediction/
│
├── app.py                   # Streamlit app for interactive predictions
├── model_utils.py           # Utilities for loading data & training
├── train_and_test_improved.py  # Advanced training script with scaling & tuning
├── student_data.csv         # Dataset (Attendance, Mid1, Mid2, FinalMarks)
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

### **Requirements**
streamlit
pandas
scikit-learn
joblib
```
Install them:
**Dataset Format**
CSV must have the following columns:
Attend,Mid1,Mid2,FinalMarks
95,32,34,94
88,30,28,88
70,20,22,70
```
* **Attend**: Attendance percentage (0–100)
* **Mid1, Mid2**: Marks out of 35 each
* **FinalMarks**: Final marks (0–100)
### **How to Run the App**

1. Make sure `student_data.csv` is in the project folder.
2. Run Streamlit:

```bash
streamlit run app.py
```

3. Enter:

   * Attendance (%)
   * Mid-1 marks (out of 35)
   * Mid-2 marks (out of 35)
4. Click **Predict Final Marks** → shows predictions from **Linear Regression** and **Random Forest**.

---

### **Improved Training Script**

Run:

```bash
python train_and_test_improved.py
```

This script:

* Scales target (FinalMarks) to 0–1
* Trains Polynomial Ridge Regression and Random Forest
* Clips predictions to 0–1 before inverse-scaling back to 0–100
* Evaluates both models
* Tests three predefined cases:

  ```
  Case 1: [90, 30, 32]
  Case 2: [55, 15, 20]
  Case 3: [80, 28, 30]
  ```
* Saves best models and target scaler:
  lin_scaled_best.joblib
  rf_scaled_best.joblib
  target_scaler.joblib

### **Example Predictions (Improved Models)**
For test cases:
[90,30,32] → Linear: 100.00, RF: 99.96
[55,15,20] → Linear: 62.29, RF: 62.15
[80,28,30] → Linear: 96.13, RF: 99.10

### **Key Enhancements**
* No predictions above 100 (due to scaling and clipping)
* Better generalization via regularization and feature engineering
* Easy integration with Streamlit for live testing.
