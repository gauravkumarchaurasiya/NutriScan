from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the preprocessor, feature selector, and model
preprocessor_path = Path("models/transformers/preprocessor.joblib")
feature_selector_path = Path("models/feature_selector/feature_selector.joblib")
model_path = Path("models/best_model/RandomForestRegressor_model.joblib")

preprocessor = joblib.load(preprocessor_path)
feature_selector = joblib.load(feature_selector_path)
model = joblib.load(model_path)

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})

@app.post("/predict", response_class=HTMLResponse)
def form_post(request: Request,
              Diabetes_Percentage_2008: float = Form(...),
              Diabetes_Percentage_2013: float = Form(...),
              Obesity_Percentage_2012: float = Form(...),
              Recreation_Facilities_2011: float = Form(...),
              Recreation_Facilities_2016: float = Form(...),
              Recreation_Facilities_per_1000_2011: float = Form(...),
              Recreation_Facilities_per_1000_2016: float = Form(...),
              Recreation_Facilities_Percent_Change_2011_16: float = Form(...),
              Recreation_Facilities_Per_1000_Pop_Percent_Change_2011_16: float = Form(...)):
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Diabetes_Percentage_2008': [Diabetes_Percentage_2008],
        'Diabetes_Percentage_2013': [Diabetes_Percentage_2013],
        'Obesity_Percentage_2012': [Obesity_Percentage_2012],
        'Recreation_Facilities_2011': [Recreation_Facilities_2011],
        'Recreation_Facilities_2016': [Recreation_Facilities_2016],
        'Recreation_Facilities_per_1000_2011': [Recreation_Facilities_per_1000_2011],
        'Recreation_Facilities_per_1000_2016': [Recreation_Facilities_per_1000_2016],
        'Recreation_Facilities_Percent_Change_2011_16': [Recreation_Facilities_Percent_Change_2011_16],
        'Recreation_Facilities_Per_1000_Pop_Percent_Change_2011_16': [Recreation_Facilities_Per_1000_Pop_Percent_Change_2011_16]
    })

    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)
    input_data_preprocessed = pd.DataFrame(input_data_preprocessed, columns=input_data.columns)
    
    # Select features
    selected_features = feature_selector.transform(input_data_preprocessed)

    # Make prediction
    prediction = model.predict(selected_features)

    return templates.TemplateResponse('index.html', context={'request': request, 'prediction': prediction[0]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
