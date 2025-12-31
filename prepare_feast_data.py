import pandas as pd
import os

# ১. ডেটা লোড করা
# আপনার ফাইলটি যেখানে আছে তার পাথ দিন
df = pd.read_csv("data-source/final_augmented_train_data.csv")

# ২. আপনার দেওয়া কলামগুলো সিলেক্ট করা (অপ্রয়োজনীয় কলাম বাদ দেওয়া)
required_columns = [
    'Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'MaritalStatus', 
    'Gender', 'JobSatisfaction', 'EnvironmentSatisfaction', 
    'RelationshipSatisfaction', 'WorkLifeBalance', 'Education', 
    'Department', 'OverTime', 'DistanceFromHome', 'Attrition'
]

# শুধু এই কলামগুলোই নেব (যদি ফাইলে থাকে)
existing_cols = [col for col in required_columns if col in df.columns]
df = df[existing_cols]

# ৩. Feast-এর জন্য 'employee_id' তৈরি করা (যেহেতু আপনার লিস্টে ID ছিল না)
# আমরা জাস্ট ১, ২, ৩... এভাবে আইডি দিয়ে দিচ্ছি
df['employee_id'] = range(1, len(df) + 1)

# ৪. টাইমস্ট্যাম্প যোগ করা (আবশ্যিক)
df['event_timestamp'] = pd.to_datetime("2024-01-01")

# ৫. Parquet ফরম্যাটে সেভ করা
save_path = "feature_store/data"
os.makedirs(save_path, exist_ok=True)
df.to_parquet(f"{save_path}/attrition.parquet")

print(f"✅ Feast data prepared successfully at: {save_path}/attrition.parquet")
print(f"✅ Columns: {df.columns.tolist()}")