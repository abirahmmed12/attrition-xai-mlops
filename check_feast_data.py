import pandas as pd
import os

# ফাইল পাথ
file_path = "feature_store/data/train_attrition_features.parquet"

if os.path.exists(file_path):
    # Parquet ফাইল রিড করা
    df = pd.read_parquet(file_path)
    
    print("✅ Feast Data Found!")
    print(f"Total Rows: {len(df)}")
    print("\nColumns inside Feature Store data:")
    print(df.columns.tolist())
    
    print("\nSample Data (First 2 rows):")
    print(df[['employee_id', 'event_timestamp', 'YearsPerPromotion', 'Attrition']].head(2))
else:
    print("❌ Feast data file not found! Check path.")