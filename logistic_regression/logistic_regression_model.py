from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.utils import shuffle
import joblib


# Assuming 'train_df' and 'test_df' are already loaded and cleaned as in your previous steps


# Load datasets
train_path = "../amazon_review_polarity_csv/train.csv"
test_path = "../amazon_review_polarity_csv/test.csv"

train_df = pd.read_csv(train_path, header=None, names=['polarity', 'title', 'text'])
test_df = pd.read_csv(test_path, header=None, names=['polarity', 'title', 'text'])

# Shuffle data
train_df = shuffle(train_df).reset_index(drop=True)
test_df = shuffle(test_df).reset_index(drop=True)

# Data Cleaning: Fill NaNs and ensure string type
train_df['title'] = train_df['title'].fillna('').astype(str)
train_df['text'] = train_df['text'].fillna('').astype(str)

test_df['title'] = test_df['title'].fillna('').astype(str)
test_df['text'] = test_df['text'].fillna('').astype(str)

# Combine title and text
train_df['combined'] = train_df['title'] + " " + train_df['text']
test_df['combined'] = test_df['title'] + " " + test_df['text']

# Final check to ensure 'combined' is string and no NaNs
train_df['combined'] = train_df['combined'].fillna('').astype(str)
test_df['combined'] = test_df['combined'].fillna('').astype(str)

# Verify the cleaning
print("\nAfter cleaning:")
print("Data types in 'combined':")
print(train_df['combined'].apply(type).value_counts())

print("\nNumber of NaNs in 'combined':")
print(train_df['combined'].isna().sum())

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=20000, lowercase=True, stop_words='english')

# Fit on training data and transform both training and testing data
X_train_bow = vectorizer.fit_transform(train_df['combined'])
X_test_bow = vectorizer.transform(test_df['combined'])

# Save the fitted CountVectorizer to a file
joblib.dump(vectorizer, 'count_vectorizer.joblib')

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['polarity'])
y_test = label_encoder.transform(test_df['polarity'])

# Split training data into training and validation sets
X_train_bow_split, X_val_bow, y_train_split, y_val = train_test_split(
    X_train_bow, y_train, test_size=0.1, random_state=42
)

# Initialize and train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_bow_split, y_train_split)

# Predict on validation and test sets
y_val_pred = model.predict(X_val_bow)
y_test_pred = model.predict(X_test_bow)

# Evaluate accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

import joblib

# Save the model to a file
joblib.dump(model, 'logistic_regression_model.joblib')


# You can use the loaded model to make predictions
