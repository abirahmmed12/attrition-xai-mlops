import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from datetime import datetime


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    feature_store_data_path = os.path.join("feature_store", "data")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_feature_engineering_object(self):
        """
        Returns the functions responsible for feature engineering and outlier handling.
        """
        try:
            
            def add_new_features(df):
                df = df.copy()
                logging.info("Adding Research-Grade Composite Features...")
                
                
                if 'YearsSinceLastPromotion' in df.columns and 'YearsAtCompany' in df.columns:
                    df['YearsPerPromotion'] = (df['YearsSinceLastPromotion'] + 1) / (df['YearsAtCompany'] + 1)
                
                
                satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']
                existing_satisfaction_cols = [col for col in satisfaction_cols if col in df.columns]
                if existing_satisfaction_cols:
                    df['TotalSatisfaction'] = df[existing_satisfaction_cols].mean(axis=1)

                
                if 'Overtime' in df.columns and 'WorkLifeBalance' in df.columns:
                    overtime_num = df['Overtime'].map({'Yes': 1, 'No': 0})
                    overtime_num = overtime_num.fillna(0)
                    df['OvertimeStress'] = overtime_num / (df['WorkLifeBalance'] + 0.5)
                
                return df

            
            def handle_outliers(df):
                outlier_cols = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
                df = df.copy()
                for col in outlier_cols:
                    if col in df.columns:
                       
                        if pd.api.types.is_numeric_dtype(df[col]):
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            upper_limit = Q3 + 1.5 * IQR
                            lower_limit = Q1 - 1.5 * IQR
                            
                            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
                            df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
                return df
            
            return add_new_features, handle_outliers

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        """
        Creates the preprocessing pipeline dynamically based on AVAILABLE columns.
        """
        try:
           
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

           
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal_encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical columns for scaling: {numerical_columns}")
            logging.info(f"Categorical columns for encoding: {categorical_columns}")

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def push_to_feature_store(self, df, prefix):
        try:
            os.makedirs(self.data_transformation_config.feature_store_data_path, exist_ok=True)
            
            
            if 'employee_id' not in df.columns:
                 if 'EmployeeNumber' in df.columns:
                     df['employee_id'] = df['EmployeeNumber']
                 else:
                     df['employee_id'] = df.index + 1000
            
            if 'event_timestamp' not in df.columns:
                df['event_timestamp'] = pd.to_datetime(datetime.now())

            file_name = f"{prefix}_attrition_features.parquet"
            path = os.path.join(self.data_transformation_config.feature_store_data_path, file_name)
            
            df.to_parquet(path, index=False)
            logging.info(f"Successfully saved {prefix} data for Feast at {path}")
            
        except Exception as e:
            logging.error(f"Error saving to Feature Store: {str(e)}")

    def initiate_data_transformation(self, train_path, test_path):
        try:
           
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            
            feature_eng_func, outlier_func = self.get_feature_engineering_object()

            
            logging.info("Applying Feature Engineering...")
            train_df = feature_eng_func(train_df)
            test_df = feature_eng_func(test_df)
            
            logging.info("Handling Outliers...")
            train_df = outlier_func(train_df)
            test_df = outlier_func(test_df)

            
            logging.info("Pushing enriched data to Feast Feature Store...")
            self.push_to_feature_store(train_df, "train")
            self.push_to_feature_store(test_df, "test")

            
            target_column_name = "Attrition"
            
            
            potential_numerical_columns = [
                'Age', 'DailyRate', 'DistanceFromHome', 'EnvironmentSatisfaction',
                'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 
                'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany', 
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                'YearsPerPromotion', 'TotalSatisfaction', 'OvertimeStress'
            ]
            
            
            potential_categorical_columns = [
                'BusinessTravel', 'Department', 'EducationField', 'Gender',
                'JobRole', 'MaritalStatus', 'OverTime', 'Education'
            ]

            
            numerical_columns = [col for col in potential_numerical_columns if col in train_df.columns]
            categorical_columns = [col for col in potential_categorical_columns if col in train_df.columns]

            
            preprocessor_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            
            cols_to_drop = [target_column_name, 'employee_id', 'event_timestamp', 'EmployeeCount', 'StandardHours', 'Over18']
            train_drop = [c for c in cols_to_drop if c in train_df.columns]
            test_drop = [c for c in cols_to_drop if c in test_df.columns]
            
            input_feature_train_df = train_df.drop(columns=train_drop, axis=1)
            target_feature_train_df = train_df[target_column_name].map({'Yes': 1, 'No': 0})

            input_feature_test_df = test_df.drop(columns=test_drop, axis=1)
            target_feature_test_df = test_df[target_column_name].map({'Yes': 1, 'No': 0})

            logging.info("Applying preprocessing object (Scaling & Encoding)...")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
   
    print(f"Train Array Shape: {train_arr.shape}")