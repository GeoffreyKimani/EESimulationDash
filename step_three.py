import os
import pandas as pd
from constants import DATA_DIR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_features_for_crop(crop):
    df = None
    if crop == 'maize':
        maize_file = os.path.join(DATA_DIR, 'maize_df_features.csv')
        df = pd.read_csv(maize_file)
    else:
        potato_file = os.path.join(DATA_DIR, 'potato_df_features.csv')
        df = pd.read_csv(potato_file)
    return df

def preprocess_features(features_df_json):
    df = pd.read_json(features_df_json, orient='split')
    df.fillna(method='ffill', inplace=True)

    # Define categorical and numerical features based on the DataFrame
    categorical_features = [col for col in df.columns if df[col].dtype == 'object']
    numerical_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

    # Define the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Fit and transform the data, converting to a dense numpy array
    transformed = preprocessor.fit_transform(df)
    if hasattr(transformed, 'toarray'):
        # If the result is a sparse matrix, convert it to a dense array
        X_preprocessed = transformed.toarray()
    else:
        # The result is already a dense array
        X_preprocessed = transformed

    print(X_preprocessed.shape)
    feature_names = preprocessor.get_feature_names_out()
    print(feature_names)

    # Assuming the next step can handle dense arrays
    return X_preprocessed


def scale_y(features_df_json, y_pred=None):
    df = pd.read_json(features_df_json, orient='split')
    target = 'yield_kg_ph'

    # Optional: Scale the target variable
    y = df[target].values.reshape(-1, 1)
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y).flatten()  # Use flatten to convert back to 1D array if needed

    if y_pred is not None:
        y_pred_inverse = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        print("y_pred inverse transformed")
        return y_pred_inverse
    else:
        print("y scaled")
        return y_scaled
