#Implement Simple Naïve Bayes classification algorithm using Python/R on iris.csv dataset

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\Siya\.kaggle\kaggle.json\Iris.csv")

# Preview the dataset
print("Dataset Preview:")
print(df.head())

# Check column names
print("\nColumns:", df.columns)

# Assuming the dataset has 4 features and 1 target column named 'species'
X = df.iloc[:, :-1]  # all rows, all columns except last
y = df.iloc[:, -1]   # all rows, last column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nConfusion Matrix:\n", cm)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification report for Precision, Recall, etc.
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# For binary classification metrics (only valid if we had 2 classes):
# Example: assuming you want metrics for a specific class like 'Iris-setosa'
target_class = model.classes_[0]  # choose a class, e.g., 'Iris-setosa'
y_true_binary = (y_test == target_class).astype(int)
y_pred_binary = (y_pred == target_class).astype(int)

cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
TN, FP, FN, TP = cm_binary.ravel()

accuracy = accuracy_score(y_true_binary, y_pred_binary)
error_rate = 1 - accuracy
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)

print(f"\nFor class '{target_class}':")
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
