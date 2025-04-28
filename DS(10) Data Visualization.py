#for iris dataset, 
#1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
#2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
#3. Create a boxplot for each feature in the dataset.
#4. Compare distributions and identify outliers.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset into a DataFrame
df = pd.read_csv(r"C:\Users\Siya\.kaggle\kaggle.json\Iris.csv")

print("🔍 Dataset Preview:")
print(df.head())

if 'Id' in df.columns:
    df = df.drop(columns='Id')

# ------------------------------------------
# 1. List Features and Their Types
# ------------------------------------------
print("\n📋 Features and Data Types:")
print(df.dtypes)

# Identifying type as numeric/nominal
print("\n📌 Feature Type Inference:")
for column in df.columns:
    if df[column].dtype == 'object':
        print(f"{column}: Nominal")
    else:
        print(f"{column}: Numeric")

# ------------------------------------------
# 2. Histograms for Feature Distributions
# ------------------------------------------
print("\n📊 Histograms for Each Feature:")
df.hist(figsize=(10, 8), edgecolor='black')
plt.suptitle("Feature Distributions - Histograms", fontsize=16)
plt.tight_layout()
plt.show()

# ------------------------------------------
# 3. Boxplots for Each Feature
# ------------------------------------------
print("\n📦 Boxplots for Each Feature:")
plt.figure(figsize=(10, 8))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

# ------------------------------------------
# 4. Outlier Detection & Distribution Comparison
# ------------------------------------------
print("\n🚨 Outlier Detection Summary:")
for col in df.columns[:-1]:  # Skip 'species'
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

