import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# File paths after extracting the zip file
train_data_path = 'titanic/train.csv'
test_data_path = 'titanic/test.csv'
gender_submission_path = 'titanic/gender_submission.csv'

# Check if files exist
if not os.path.exists(train_data_path):
    print(f"Error: Train data file not found at {train_data_path}")
    exit()

if not os.path.exists(test_data_path):
    print(f"Error: Test data file not found at {test_data_path}")
    exit()

if not os.path.exists(gender_submission_path):
    print(f"Error: Gender submission file not found at {gender_submission_path}")
    exit()

# Load the data
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
gender_submission = pd.read_csv(gender_submission_path)

# Step 1: Data Analysis and Missing Value Handling
print("\nMissing values in the train data:")
print(train_data.isnull().sum())

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

print("\nMissing values in the test data:")
print(test_data.isnull().sum())

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# Drop irrelevant columns
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encode categorical variables
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Prepare training data
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure columns in test data match training data
if set(test_data.columns) != set(X_train.columns):
    raise ValueError("Test data columns do not match training data columns.")

# Make predictions on the test data
test_predictions = model.predict(test_data)

# Check if gender submission file matches test data size
if len(gender_submission) != len(test_data):
    raise ValueError("Gender submission file size does not match the test data size.")

# Calculate accuracy
gender_actual = gender_submission['Survived']
accuracy = accuracy_score(gender_actual, test_predictions)
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")

# Confusion matrix
conf_matrix = confusion_matrix(gender_actual, test_predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
