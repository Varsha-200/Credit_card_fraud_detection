# BALANCING OF DATASET FOR CREDIT CARD FRAUD DETECTION
``` python
pip install imbalanced-learn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```
## Loading of dataset
``` python
credit_card_data = pd.read_csv('creditcard.csv')
## Separate features and target variable
X = credit_card_data.drop('Class', axis=1)
y = credit_card_data['Class']
```
## Split data into train and test sets
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
# Identify NaN values in y_train
``` python
nan_indices = y_train.index[y_train.isnull()]
smote=SMOTE()
```
# Remove rows with NaN values
``` python
X_train_cleaned = X_train.drop(index=nan_indices)
y_train_cleaned = y_train.drop(index=nan_indices)
```
# Apply SMOTE on cleaned data
``` python
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_cleaned, y_train_cleaned)
```
# Train a classifier on the balanced data
``` python
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_balanced, y_train_balanced)
```
# Predict on the test set
``` python
y_pred = rf_classifier.predict(X_test)
```
# Evaluate the model
``` python
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
