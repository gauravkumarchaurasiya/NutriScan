Nutriscan
==============================

NutriScan is a comprehensive data pipeline project that analyzes the intricate relationship between food accessibility, retail food environment, and health outcomes across U.S. counties. Leveraging the USDA's Food Environment Atlas dataset, this project aims to provide insights into how the food landscape correlates with critical health indicators such as diabetes and obesity rates.

Project Organization
------------

   .
├── app.py
├── Dockerfile
├── dvc.yaml
├── README.md
├── requirements.txt
├── templates/
│   └── index.html
├── models/
│   ├── best_model/
│   ├── feature_selector/
│   └── transformers/
├── reports/
│   ├── eda_reports/
│   └── figures/
├── data/
│   ├── raw/
│   │   ├── api_data.csv
│   │   └── csv_data.csv
│   └── processed/
│       └── processed_data.csv
└── src/
    ├── logger.py
    ├── data_ingestion.py
    ├── feature/
    │   ├── data_preprocessing.py
    │   ├── feature_selection.py
    ├── visualization/
    │   └── visualize.py
    └── model/
        └── train.py


