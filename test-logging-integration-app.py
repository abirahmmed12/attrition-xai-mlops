import pandas as pd

# artifacts ফোল্ডারের ট্রেনিং ডেটা লোড করুন
df = pd.read_csv('artifacts/train.csv')

# অ্যাট্রিশন কলামের ভ্যালু কাউন্ট দেখুন (ধরে নিচ্ছি কলামের নাম 'Attrition')
print(df['Attrition'].value_counts())

# পারসেন্টেজ দেখতে চাইলে
print(df['Attrition'].value_counts(normalize=True) * 100)