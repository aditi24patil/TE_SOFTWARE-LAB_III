#Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about 
#the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we can find any patterns in the data.
#Write a code to check how the price of the ticket (column name: 'fare') for each passenger is distributed by plotting a histogram.

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Preview the dataset
print(titanic.head())
print("\nDataset Shape:", titanic.shape)

sns.countplot(x='sex', hue='survived', data=titanic)
plt.title('Survival Count by Gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

sns.countplot(x='pclass', hue='survived', data=titanic)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

sns.histplot(data=titanic, x='age', hue='survived', kde=True, multiple='stack')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.show()

# Drop missing fares just in case
titanic_fare = titanic['fare'].dropna()

# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(titanic_fare, bins=30, kde=True)
plt.title('Distribution of Ticket Fare')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.show()
