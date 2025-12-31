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
            # শুধু স্টার্টআপের সময় এই মেসেজ আসবে, প্রেডিকশনের সময় না
            print("Loading models into memory...")
            self.model = load_object(file_path=self.model_path)
            self.preprocessor = load_object(file_path=self.preprocessor_path)
            self._loaded = True
            print("Models loaded successfully!")
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # Safety check
            if not self._loaded or self.model is None or self.preprocessor is None:
                self.load_models()
            
            # 1. Preprocessing
            data_scaled = self.preprocessor.transform(features)

            # 2. Prediction (Optimized & Clean)
            if hasattr(self.model, "predict_proba"):
                # শুধু একবার মডেল কল করছি (Time saving)
                prob_arr = self.model.predict_proba(data_scaled)
                prob_leaving = prob_arr[0, 1]
                
                prediction = 1 if prob_leaving >= 0.5 else 0
                
                # Confidence Calculation
                if 0.45 < prob_leaving < 0.55: 
                    confidence = "Low"
                elif 0.35 < prob_leaving < 0.65: 
                    confidence = "Moderate"
                else: 
                    confidence = "High"
                
                return prediction, prob_leaving, confidence
            
            else:
                # Fallback for models without predict_proba
                preds = self.model.predict(data_scaled)
                return int(preds[0]), 0.5, "Unknown"
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_influencing_factors(self, df):
        try:
            negatives = []
            positives = []
            
            # Helper functions to avoid crashes
            def get_val(col):
                return float(df[col].iloc[0]) if col in df.columns else 0

            def get_str(col):
                return df[col].iloc[0] if col in df.columns else ""

            # Rules Engine
            if get_val('JobSatisfaction') <= 2: 
                negatives.append("Low Job Satisfaction")
            
            if get_str('OverTime') == 'Yes': 
                negatives.append("Excessive Overtime")
            
            if get_val('MonthlyIncome') < 30000: 
                negatives.append("Low Salary Bracket")
            
            if get_val('YearsSinceLastPromotion') > 3: 
                negatives.append("Career Stagnation")
            
            if get_val('WorkLifeBalance') <= 2: 
                negatives.append("Poor Work-Life Balance")
            
            if get_val('DistanceFromHome') > 20:
                negatives.append("Long Commute")
            
            if get_val('TotalWorkingYears') > 10: 
                positives.append("High Experience")
            
            if (get_val('YearsAtCompany') > 5 and get_val('JobSatisfaction') >= 3):
                positives.append("Company Loyalty")
            
            if get_str('MaritalStatus') == 'Married': 
                positives.append("Family Stability")
            
            if (get_val('YearsSinceLastPromotion') <= 1 and get_val('JobSatisfaction') >= 3):
                positives.append("Recent Career Growth")
            
            if get_val('MonthlyIncome') > 50000:
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
            
            # Feature Engineering with Safety Checks
            satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']
            
            if all(col in df.columns for col in satisfaction_cols):
                df['TotalSatisfaction'] = df[satisfaction_cols].astype(float).mean(axis=1)
            else:
                df['TotalSatisfaction'] = 0 

            if 'YearsSinceLastPromotion' in df.columns and 'YearsAtCompany' in df.columns:
                denom = df['YearsAtCompany'].astype(float) + 1
                df['YearsPerPromotion'] = (df['YearsSinceLastPromotion'].astype(float) + 1) / denom
            else:
                df['YearsPerPromotion'] = 0

            if 'OverTime' in df.columns and 'WorkLifeBalance' in df.columns:
                overtime_num = df['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0)
                df['OvertimeStress'] = overtime_num / (df['WorkLifeBalance'].astype(float) + 0.5)
            else:
                df['OvertimeStress'] = 0
            
            return df
        except Exception as e:
            raise CustomException(e, sys)