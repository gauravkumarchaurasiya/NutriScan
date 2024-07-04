import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from src.logger import logging

def load_data(input_path: Path):
    logging.info(f"Loading data from {input_path}")
    return pd.read_csv(input_path)

def perform_eda(df: pd.DataFrame,eda_output_folder:Path):
    logging.info("Performing Exploratory Data Analysis...")
    
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Obesity_Percentage_2017'], kde=True)
    plt.title('Distribution of Obesity Percentage 2017')
    plt.savefig(eda_output_folder/'obesity_distribution.png')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(eda_output_folder/'correlation_heatmap.png')
    plt.close()

    # Top 5 correlated features with target
    corr_with_target = df.corr()['Obesity_Percentage_2017'].sort_values(ascending=False)
    top_5_corr = corr_with_target[1:6]  # Exclude the target itself
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_5_corr.index, y=top_5_corr.values)
    plt.title('Top 5 Correlated Features with Obesity Percentage 2017')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(eda_output_folder/'top_5_correlations.png')
    plt.close()

    # Summary statistics
    summary_stats = df.describe()
    summary_stats.to_csv(eda_output_folder/'summary_statistics.csv')

    logging.info("EDA completed. Outputs saved in 'eda_outputs' directory.")

def main():
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        input_path = root_path/'data'/'Preprocessed'
        
        # Create EDA outputs directory
        eda_output_path = root_path/'reports'/'eda_outputs'
        eda_output_path.mkdir(parents=True, exist_ok=True)
        
        # Find the most recent preprocessed file
        input_files = list(input_path.glob('preprocessed_food_atlas_data_*.csv'))
        if not input_files:
            raise FileNotFoundError("No preprocessed files found")
        latest_file = max(input_files, key=lambda x: x.stat().st_mtime)
        
        df = load_data(latest_file)
        perform_eda(df,eda_output_path)
        
        logging.info("Step 3: Completed Exploratory Data Analysis")
        
    except Exception as e:
        logging.error(f"An error occurred during EDA: {str(e)}")

if __name__ == "__main__":
    main()