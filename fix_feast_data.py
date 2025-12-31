import pandas as pd
import os

# ১. সোর্স এবং টার্গেট পাথ সেট করা (একদম নির্দিষ্ট করে)
base_path = "/home/abir-ahmmed/projects/employee-attrition-mlops"
csv_path = os.path.join(base_path, "data-source/final_augmented_train_data.csv")
target_folder = os.path.join(base_path, "feature_store/data")
target_file = os.path.join(target_folder, "attrition.parquet")

print(f"Reading CSV from: {csv_path}")

# ২. ডেটা লোড এবং প্রসেস করা
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # Feast-এর জন্য জরুরি কলাম
    df['employee_id'] = range(1, len(df) + 1)
    df['event_timestamp'] = pd.to_datetime("2024-01-01")
    
    # ৩. কলাম ফিল্টার করা (যাতে কোনো গার্বেজ না থাকে)
    required_cols = [
        'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'MaritalStatus', 
        'Gender', 'JobSatisfaction', 'EnvironmentSatisfaction', 
        'RelationshipSatisfaction', 'WorkLifeBalance', 'Education', 
        'Department', 'OverTime', 'DistanceFromHome', 'Attrition',
        'employee_id', 'event_timestamp'
    ]
    # শুধু যে কলামগুলো আসলে আছে সেগুলোই নেব
    existing_cols = [c for c in required_cols if c in df.columns]
    df = df[existing_cols]

    # ৪. ফোল্ডার তৈরি করে ফাইল সেভ করা
    os.makedirs(target_folder, exist_ok=True)
    df.to_parquet(target_file)
    
    print(f"✅ SUCCESS! File created at: {target_file}")
else:
    print(f"❌ ERROR: CSV file not found at {csv_path}")