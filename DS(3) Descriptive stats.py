#Descriptive Statistics - Measures of Central Tendency and variability

import pandas as pd
data = pd.read_csv("C:/Users/Siya/.kaggle/kaggle.json/Iris.csv")
print(data.head())
grouped_summary= data.groupby('Species').agg(['mean', 'median', 'min', 'max', 'std'])
print("\nSummary statistics grouped by 'species':\n")
print(grouped_summary)

sepal_length_grouped= data.groupby('Species')['SepalLengthCm'].apply(list).to_dict()
print("\nList of sepal length values per species :")
for Species, values in sepal_length_grouped.items():
    print(f"{Species}: {values}")

species_list = data['Species'].unique()
for Species in species_list:
    print(f"\n---Statistics for {Species}---")
    stats=data[data['Species']==Species].describe(percentiles=[.25, .5, .75])
    print(stats)
