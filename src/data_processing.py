# train.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- Paths ---
CSV_PATH = "data/raw/Telco-Customer-Churn.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "best_rf.pkl")

# --- Preprocessing function ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares Telco Customer Churn data for modeling.
    """
    df = df.drop_duplicates()

    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges_missing'] = df['TotalCharges'].isna().astype(int)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())


    # Yes/No columns → 1/0
    yes_no_cols = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'Churn'
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Encode gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # One-hot encode remaining categorical columns
    cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True)

    # Numeric columns
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df

# --- Load data ---
df = pd.read_csv(CSV_PATH)
df = preprocess_data(df)

# --- Split features and target ---
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

# Optional: scale numeric features
num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TotalCharges_missing']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Train Random Forest ---
clf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced', random_state=42
)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred_prob = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, clf.predict(X_test)))

# --- Save model ---
joblib.dump(clf, MODEL_PATH)
print(f"Saved trained model to {MODEL_PATH}")
