import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
import time

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model = None
        self.preprocessor = None
        self._loaded = False

    def load_models(self):
        """Load models once and cache in memory"""
        if self._loaded:
            return
            
        try:
            print(f" Loading model from {self.model_path}...")
            start = time.time()
            self.model = load_object(file_path=self.model_path)
            print(f"Model loaded in {time.time() - start:.2f}s")
            
            print(f" Loading preprocessor from {self.preprocessor_path}...")
            start = time.time()
            self.preprocessor = load_object(file_path=self.preprocessor_path)
            print(f" Preprocessor loaded in {time.time() - start:.2f}s")
            
            self._loaded = True
            print(" All models ready for inference!")
            
        except Exception as e:
            print(f" Error loading models: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # Safety check
            if not self._loaded or self.model is None or self.preprocessor is None:
                print(" Models not loaded, loading now...")
                self.load_models()
            
            # Fast inference
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            
            if hasattr(self.model, "predict_proba"):
                prob_leaving = self.model.predict_proba(data_scaled)[0, 1]
                
                if 0.45 < prob_leaving < 0.55: 
                    confidence = "Low"
                elif 0.35 < prob_leaving < 0.65: 
                    confidence = "Moderate"
                else: 
                    confidence = "High"
                    
                return int(preds[0]), prob_leaving, confidence
            return int(preds[0]), 0.5, "Unknown"
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_influencing_factors(self, df):
        try:
            negatives = []
            positives = []
            
            if float(df['JobSatisfaction'].iloc[0]) <= 2: 
                negatives.append("Low Job Satisfaction")
            
            if df['OverTime'].iloc[0] == 'Yes': 
                negatives.append("Excessive Overtime")
            
            if float(df['MonthlyIncome'].iloc[0]) < 30000: 
                negatives.append("Low Salary Bracket")
            
            if float(df['YearsSinceLastPromotion'].iloc[0]) > 3: 
                negatives.append("Career Stagnation")
            
            if float(df['WorkLifeBalance'].iloc[0]) <= 2: 
                negatives.append("Poor Work-Life Balance")
            
            if float(df['DistanceFromHome'].iloc[0]) > 20:
                negatives.append("Long Commute")
            
            if float(df['TotalWorkingYears'].iloc[0]) > 10: 
                positives.append("High Experience")
            
            if (float(df['YearsAtCompany'].iloc[0]) > 5 and 
                float(df['JobSatisfaction'].iloc[0]) >= 3):
                positives.append("Company Loyalty")
            
            if df['MaritalStatus'].iloc[0] == 'Married': 
                positives.append("Family Stability")
            
            if (float(df['YearsSinceLastPromotion'].iloc[0]) <= 1 and 
                float(df['JobSatisfaction'].iloc[0]) >= 3):
                positives.append("Recent Career Growth")
            
            if float(df['MonthlyIncome'].iloc[0]) > 50000:
                positives.append("Competitive Salary")

            return {'negatives': negatives, 'positives': positives}
        except Exception:
            return {'negatives': [], 'positives': []}


class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_data_frame(self):
        try:
            df = pd.DataFrame([self.data])
            
            satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']
            df['TotalSatisfaction'] = df[satisfaction_cols].mean(axis=1)
            
            df['YearsPerPromotion'] = (df['YearsSinceLastPromotion'] + 1) / (df['YearsAtCompany'] + 1)
            
            overtime_num = df['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0)
            df['OvertimeStress'] = overtime_num / (df['WorkLifeBalance'] + 0.5)
            
            return df
        except Exception as e:
            raise CustomException(e, sys)