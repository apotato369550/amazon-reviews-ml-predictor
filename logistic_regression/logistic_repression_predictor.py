import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved Logistic Regression model
model = joblib.load('logistic_regression_model.joblib')

# Load the same CountVectorizer that was used during training
vectorizer = joblib.load('count_vectorizer.joblib')  # Make sure to save and load the vectorizer too!

# Function to predict review sentiment
def predict_review(title, text):
    # Combine title and text just like in the training phase
    combined = title + " " + text
    
    # Transform the combined review into a BoW vector
    review_bow = vectorizer.transform([combined])
    
    # Make prediction using the loaded model
    prediction = model.predict(review_bow)
    
    # Decode the prediction (assume label_encoder used 0 for negative and 1 for positive)
    polarity = 'Positive' if prediction[0] == 1 else 'Negative'
    return polarity

# Example usage:
sample_title = "Amazing product"
sample_text = "This is the best product I have ever used!"
print(predict_review(sample_title, sample_text))  # Output: Positive

sample_title_neg = "Terrible quality"
sample_text_neg = "The product broke within two days of use. Very disappointed."
print(predict_review(sample_title_neg, sample_text_neg))  # Output: Negative
