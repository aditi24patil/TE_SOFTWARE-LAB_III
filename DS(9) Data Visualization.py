#Use the inbuilt dataset 'titanic'. Plot a box plot for distribution of age with respect to each gender along with the information about whether they survived or
#not. (Column names : 'sex' and 'age')
#Write observations on the inference from the above statistics

import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Drop rows with missing age values
titanic = titanic.dropna(subset=['age'])

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)
plt.title('Age Distribution by Gender and Survival Status')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
