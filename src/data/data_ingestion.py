import requests
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
from io import BytesIO
from src.logger import logging

# USDA Food Environment Atlas data URL
DATA_URL = "https://www.ers.usda.gov/webdocs/DataFiles/80526/FoodEnvironmentAtlas.xls?v=3592.9"

def fetch_food_atlas_data():
    logging.info("Fetching Food Environment Atlas data...")
    response = requests.get(DATA_URL)
    if response.status_code == 200:
        return pd.read_excel(BytesIO(response.content), sheet_name="HEALTH")
    else:
        raise Exception(f"Data download failed with status code: {response.status_code}")

def structure_food_atlas_data(df):
    logging.info("Structuring data...")
    
    selected_columns = [
        'FIPS', 'State', 'County', 
        'PCT_DIABETES_ADULTS08', 'PCT_DIABETES_ADULTS13', 'PCT_OBESE_ADULTS12',
         'PCT_OBESE_ADULTS17',
        'RECFAC11', 'RECFAC16', 'RECFACPTH11', 'RECFACPTH16',
        'PCH_RECFAC_11_16', 'PCH_RECFACPTH_11_16'
    ]
    
    df_subset = df[selected_columns]
    
    df_subset = df_subset.rename(columns={
        'PCT_DIABETES_ADULTS13': 'Diabetes_Percentage_2013',
        'PCT_OBESE_ADULTS17': 'Obesity_Percentage_2017',
        'PCT_OBESE_ADULTS12': 'Obesity_Percentage_2012',
        'PCT_DIABETES_ADULTS08': 'Diabetes_Percentage_2008',
        'RECFAC11': 'Recreation_Facilities_2011',
        'RECFAC16': 'Recreation_Facilities_2016',
        'RECFACPTH11': 'Recreation_Facilities_per_1000_2011',
        'RECFACPTH16': 'Recreation_Facilities_per_1000_2016',
        'PCH_RECFAC_11_16': 'Recreation_Facilities_Percent_Change_2011_16',
        'PCH_RECFACPTH_11_16': 'Recreation_Facilities_Per_1000_Pop_Percent_Change_2011_16'
    })
    
    return df_subset

def save_locally(df: pd.DataFrame, output_path: Path):
    logging.info(f"Saving data locally...")
    df.to_csv(output_path, index=False)
    logging.info(f"Data saved successfully at {output_path}")

def main():
    try:
        current_path = Path(__file__)
        root_path = current_path.parent.parent.parent
        output_path = root_path/'data' / 'Processed'
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = fetch_food_atlas_data()
        df_processed = structure_food_atlas_data(df)
        
        filename = f'food_atlas_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        save_locally(df_processed, output_path / filename)
        
        logging.info(f"Number of records: {len(df_processed)}")
        logging.info("\nFirst few rows of the data:")
        logging.info(df_processed.head())
        
        logging.info("\nData summary:")
        logging.info(df_processed.describe())
        
        logging.info("Step 1: Completed Data Extraction and Structuring")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()