import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
   
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            
            train_file_path = os.path.join('data-source', 'final_augmented_train_data.csv')
            test_file_path = os.path.join('data-source', 'test_data_holdout.csv')

            logging.info(f"Reading Train Data from: {train_file_path}")
            logging.info(f"Reading Test Data from: {test_file_path}")

            df_train = pd.read_csv(train_file_path)
            df_test = pd.read_csv(test_file_path)
            
            logging.info(f"Train Data Shape: {df_train.shape}")
            logging.info(f"Test Data Shape: {df_test.shape}")

           
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            
            df_train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            
            df_train.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data Ingestion completed! Files copied to artifacts.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()