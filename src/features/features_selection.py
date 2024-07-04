import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from src.logger import logging
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import joblib

def load_data(input_path: Path):
    logging.info(f"Loading data from {input_path}")
    return pd.read_csv(input_path)

def perform_feature_selection(df: pd.DataFrame, k=10):
    logging.info(f"Performing feature selection, selecting top {k} features...")
    
    TARGET = 'Obesity_Percentage_2017'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Perform feature selection
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Save the selector
    model_path = Path(__file__).parent.parent.parent / 'models' / 'feature_selector'
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(selector, model_path / 'feature_selector.joblib')
    
    logging.info(f"Selected features: {selected_features}")
    
    # Create new dataframe with selected features and target
    df_selected = pd.concat([X[selected_features], y], axis=1)
    
    return df_selected

def save_selected_data(df: pd.DataFrame, output_path: Path):
    logging.info(f"Saving data with selected features to {output_path}")
    df.to_csv(output_path, index=False)

def main():
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        input_path = root_path/'data'/'Preprocessed'
        output_path = root_path/'data'/'FeatureSelected'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find the most recent preprocessed file
        input_files = list(input_path.glob('preprocessed_food_atlas_data_*.csv'))
        if not input_files:
            raise FileNotFoundError("No preprocessed files found")
        latest_file = max(input_files, key=lambda x: x.stat().st_mtime)
        
        df = load_data(latest_file)
        df_selected = perform_feature_selection(df)
        
        filename = f'feature_selected_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        save_selected_data(df_selected, output_path / filename)
        
        logging.info(f"Number of records after feature selection: {len(df_selected)}")
        logging.info(f"Number of features selected: {df_selected.shape[1] - 1}")  # -1 for target column
        
        logging.info("Step 4: Completed Feature Selection")
        
    except Exception as e:
        logging.error(f"An error occurred during feature selection: {str(e)}")

if __name__ == "__main__":
    main()