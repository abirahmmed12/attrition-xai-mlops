import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info(" Starting Training Pipeline...")

            # 1. Data Ingestion
            logging.info("Step 1: Data Ingestion initiated")
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()
            print(f" Data Ingestion Completed.")

            # 2. Data Transformation
            logging.info("Step 2: Data Transformation initiated")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            print(" Data Transformation Completed.")

            # 3. Model Training
            logging.info("Step 3: Model Training initiated")
            model_trainer = ModelTrainer()
            
            # Trainer returns a dictionary or score, handled below
            result = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            # Fix: Don't try to format a dictionary as a float
            print(" Model Training & Comparison Completed Successfully.")
            logging.info(" Training Pipeline Completed Successfully!")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        print(f" Error Occurred: {e}")
        