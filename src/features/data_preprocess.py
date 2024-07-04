import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

def load_data(input_path: Path):
    logging.info(f"Loading data from {input_path}")
    return pd.read_csv(input_path)

def preprocess_data(df: pd.DataFrame):
    logging.info("Starting data preprocessing...")
    
    # Drop FIPS, State, and County columns
    logging.info("Dropping FIPS, State, and County columns...")
    df = df.drop(columns=['FIPS', 'State', 'County'], axis=1)
    
    # Convert percentage strings to floats
    logging.info("Converting percentage strings to floats...")
    percentage_columns = [col for col in df.columns if 'PCT' in col or 'Percentage' in col]
    for col in percentage_columns:
        df[col] = df[col].astype(str).str.rstrip('%').astype('float') / 100.0
    
    # Separate features and target
    TARGET = 'Obesity_Percentage_2017'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    # Create preprocessing pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Fit and transform the features
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Convert back to DataFrame
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=X.columns)
    
    # Save the preprocessor
    model_path = Path(__file__).parent.parent.parent / 'models' / 'transformers'
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, model_path / 'preprocessor.joblib')
    
    # Combine preprocessed features with target
    df_preprocessed = pd.concat([X_preprocessed, y], axis=1)
    
    return df_preprocessed

def save_preprocessed_data(df: pd.DataFrame, output_path: Path):
    logging.info(f"Saving preprocessed data to {output_path}")
    df.to_csv(output_path, index=False)

def main():
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        input_path = root_path/'data'/'Processed'
        output_path = root_path/'data'/'Preprocessed'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find the most recent input file
        input_files = list(input_path.glob('food_atlas_data_*.csv'))
        if not input_files:
            raise FileNotFoundError("No input files found")
        latest_file = max(input_files, key=lambda x: x.stat().st_mtime)
        
        df = load_data(latest_file)
        df_preprocessed = preprocess_data(df)
        
        filename = f'preprocessed_food_atlas_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        save_preprocessed_data(df_preprocessed, output_path / filename)
        
        logging.info(f"Number of records after preprocessing: {len(df_preprocessed)}")
        
        logging.info("\nPreprocessed data summary:")
        logging.info(df_preprocessed.describe())
        
        logging.info("Step 2: Completed Data Preprocessing and Cleaning")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()