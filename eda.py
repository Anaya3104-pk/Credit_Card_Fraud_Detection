import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("fraudTrain.csv")

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)


df.drop_duplicates(inplace=True)


df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])


print("\nMissing Values:\n", df.isnull().sum())


num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())


df['hour'] = df['trans_date_trans_time'].dt.hour


print("\nData Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 1. FRAUD DISTRIBUTION
sns.countplot(x='is_fraud', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
total = len(df)
for p in plt.gca().patches:
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2,
             height,
             f'{(height/total)*100:.2f}%',
             ha='center')

plt.show()

# 2. TRANSACTION AMOUNT DISTRIBUTION
sns.histplot(df['amt'], bins=50, kde=True)
plt.xlim(0, 2000)  
plt.title("Transaction Amount Distribution")
plt.show()


# 3. FRAUD VS AMOUNT
plt.figure()
sns.boxplot(x='is_fraud', y='amt', data=df)
plt.title("Fraud vs Amount")
plt.show()

Q1 = df['amt'].quantile(0.25)
Q3 = df['amt'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[(df['amt'] >= lower_bound) & (df['amt'] <= upper_bound)]

sns.boxplot(x='is_fraud', y='amt', data=df, showfliers=False)
plt.title("Fraud vs Transaction Amount")
plt.show()


# 4. FRAUD BY CATEGORY
fraud_df = df[df['is_fraud'] == 1]

plt.figure(figsize=(10,5))
fraud_df['category'].value_counts().plot(kind='bar', color='red')
plt.title("Fraud Transactions by Category")
plt.xticks(rotation=45)
plt.show()


# 5. FRAUD BY HOUR
fraud_df = df[df['is_fraud'] == 1]

sns.histplot(fraud_df['hour'], bins=24, color='red')
plt.title("Fraud Transactions by Hour")
plt.show()

# 6. CORRELATION HEATMAP
cols_to_drop = ['Unnamed: 0', 'cc_num']
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

plt.figure(figsize=(10,6))
sns.heatmap(df_clean.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()