#Create an “Academic performance” dataset of students and perform the following operations using Python.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Create a sample dataset
np.random.seed(0)
data = pd.DataFrame({
    'student_id': range(1, 11),
    'math_score': [95, 88, np.nan, 45, 77, 91, 95, 85, np.nan, 300],  # includes missing and outliers
    'english_score': [85, 80, 78, 90, 87, np.nan, 65, 70, np.nan, 99],
    'attendance_rate': [0.95, 0.90, 0.87, 0.55, 0.93, 1.05, 0.98, 0.82, 0.77, -0.1]  # includes inconsistency
})


print("Null values : ")
print(data.isnull().sum())
data['math_score'].fillna(data['math_score'].mean(), inplace=True)
data['english_score'].fillna(data['english_score'].mean(), inplace=True)
print("After replacing null values with mean : ")
print(data.isnull().sum())

data.loc[(data['attendance_rate']<0) | (data['attendance_rate']>1), 'attendance_rate']= np.nan
print("Inconsistency in attendance rate :")
print(data.isnull().sum())
data['attendance_rate'].fillna(data['attendance_rate'].median(), inplace=True)
print("Replacing inconsistency with median : ")
print(data.isnull().sum())

def cap_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.where(col>upper,upper,np.where(col<lower,lower,col))
data['math_score'] = cap_outliers(data['math_score'])
data['attendance_rate'] = cap_outliers(data['attendance_rate'])
