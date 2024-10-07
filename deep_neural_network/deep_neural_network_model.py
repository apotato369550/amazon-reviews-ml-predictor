import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib

# Importing from keras_preprocessing
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import layers
from keras import models
from keras import callbacks

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

# Encode labels
label_encoder = LabelEncoder()
train_df['polarity'] = label_encoder.fit_transform(train_df['polarity'])
test_df['polarity'] = label_encoder.transform(test_df['polarity'])

# Features and labels
X_train = train_df['combined'].values
y_train = train_df['polarity'].values
X_test = test_df['combined'].values
y_test = test_df['polarity'].values

# Tokenization and padding
vocab_size = 20000
max_length = 200
embedding_dim = 128

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

joblib.dump(tokenizer, 'tokenizer.joblib')

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Model architecture
input_layer = layers.Input(shape=(max_length,))
embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)(input_layer)
lstm = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedding)
dense = layers.Dense(64, activation='relu')(lstm)
dropout = layers.Dropout(0.5)(dense)
output = layers.Dense(1, activation='sigmoid')(dropout)

model = models.Model(inputs=input_layer, outputs=output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Train-validation split
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_padded, y_train, test_size=0.1, random_state=42)

# Early stopping
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_final, y_train_final,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Prediction function
def predict_review(title, text):
    combined = title + " " + text
    sequence = tokenizer.texts_to_sequences([combined])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded)
    polarity = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return polarity

# Example predictions
sample_title = "Great product"
sample_text = "I really enjoyed using this product. It exceeded my expectations!"
print(predict_review(sample_title, sample_text))  # Output: Positive

sample_title_neg = "Bad experience"
sample_text_neg = "The product broke after two days of use. Very disappointed."
print(predict_review(sample_title_neg, sample_text_neg))  # Output: Negative

# Save the model
model.save('amazon_review_classifier.h5')
