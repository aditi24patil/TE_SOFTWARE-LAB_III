#1) Data Wrangling, I
#Perform the following operations using Python on any open source dataset (e.g., data.csv)
#1. Import all the required Python Libraries.
#2. Locate open source data from the web (e.g., https://www.kaggle.com). Provide a clear description of the data and its source (i.e., URL of the web site).
#3. Load the Dataset into pandas dataframe.
#4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe() function to get some initial statistics. Provide variable descriptions. Types of variables etc.
#Check the dimensions of the data frame.
#5. Data Formatting and Data Normalization: Summarize the types of variables by checking the data types (i.e., character, numeric, integer, factor, and logical) 
#of the variables in the data set. If variables are not in the correct data type, apply proper type conversions.
#6. Turn categorical variables into quantitative variables in Python.
**CHANGE DATASET PATH**

import pandas as pd
import matplotlib.pylab as plt
import numpy as num

data=pd.read_csv("C:/Users/Siya/.kaggle/kaggle.json/AutoData.csv")
print("First 5 rows are :")
data.head() #first 5 rows 
print("Last 5 rows are :")
data.tail(5) #last 5 rows 
print("Variable description of data : ")
data.describe() #variable description of data
data.info() #info about columns 
data.isnull() #finding missing values
data.isnull().sum()
data.notnull()
data.notnull().sum()
data.dtypes
data.shape
data.size
data=data.astype({"make":'category',"curbweight":'float64'})
data.info()
data['doornumber'].replace(['two', 'four'],[2, 4], inplace=True)
data[:20]
