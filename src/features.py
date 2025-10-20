#src/features.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer




def build_feature_pipeline(df: pd.DataFrame):
    df = df.copy()
    # Feature engineering columns
    df['TotalCharges_missing'] = df['TotalCharges'].isna().astype(int)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['has_internet'] = (df['InternetService'] != 'No').astype(int)


    # Define feature sets
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalCharges_missing', 'has_internet']
    categorical_features = [c for c in df.select_dtypes(include='object').columns if c not in ('customerID',)]


    # Build transformers
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])



    preprocessor = ColumnTransformer(
    transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
    ], remainder='drop', sparse_threshold=0
    )


    return preprocessor, numeric_features, categorical_features


def prepare_X_y(df):
    preprocessor, num_cols, cat_cols = build_feature_pipeline(df)
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    return X, y, preprocessor