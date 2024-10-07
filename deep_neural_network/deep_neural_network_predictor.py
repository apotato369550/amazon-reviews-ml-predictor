import numpy as np
import joblib
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences

# Define the same parameters as during training
vocab_size = 20000
max_length = 200

# Load the saved tokenizer
tokenizer = joblib.load('tokenizer.joblib')

# Load the trained model
model = tf.keras.models.load_model('amazon_review_classifier.h5')

# Function to preprocess and predict sentiment of new reviews
def predict_review(title, text):
    # Combine the title and text
    combined = title + " " + text
    
    # Tokenize the new review
    sequence = tokenizer.texts_to_sequences([combined])
    
    # Pad the sequence to the same length used during training
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Make a prediction using the loaded model
    prediction = model.predict(padded)
    
    # Interpret the model's output
    polarity = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return polarity

# Example reviews for prediction
sample_title = "Great product"
sample_text = "I really enjoyed using this product. It exceeded my expectations!"

sample_title_neg = "Bad experience"
sample_text_neg = "The product broke after two days of use. Very disappointed."

# Predict the sentiment of the example reviews
print(predict_review(sample_title, sample_text))  # Output: Positive
print(predict_review(sample_title_neg, sample_text_neg))  # Output: Negative
