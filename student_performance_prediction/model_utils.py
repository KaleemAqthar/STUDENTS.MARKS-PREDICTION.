import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path="student_data.csv"):
    """
    Loads the student dataset from the given CSV path.
    Must contain: Attend, Mid1, Mid2, FinalMarks
    """
    return pd.read_csv(path)

def train_models(df):
    """
    Trains models on scaled target (0-1) to keep predictions within 0-100.
    Returns trained models, test set, and scaler for inverse transform.
    """
    X = df[['Attend', 'Mid1', 'Mid2']]
    y = df['FinalMarks']

    # Scale target to 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42
    )

    # Train models
    lr_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)

    return lr_model, rf_model, X_test, y_test, scaler
