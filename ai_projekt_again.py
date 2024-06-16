import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Step 1: Prepare the data
# Assuming you have a file `train.txt` with sentences for training

# Step 2: Preprocess the data
# Load training data
with open('train.txt', 'r') as file:
    sentences = file.readlines()

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Convert sentences to sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')

# Create labels (for simplicity, assume binary classification)
labels = np.array([1, 0, 1, 1])  # Adjust according to your data

# Step 3: Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=padded_sequences.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
model.fit(padded_sequences, labels, epochs=10)

# Step 5: Recognize and log words
# Load a new text file
with open('lorem_ipsum.txt', 'r') as file:
    new_sentences = file.readlines()

# Convert new sentences to sequences
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded_sequences = pad_sequences(new_sequences, padding='post', maxlen=padded_sequences.shape[1])

# Make predictions
predictions = model.predict(new_padded_sequences)

# Log recognized words
recognized_words = []
for i, sequence in enumerate(new_sequences):
    for word_index in sequence:
        word = tokenizer.index_word[word_index]
        if predictions[i] > 0.5:  # Assuming a threshold of 0.5 for binary classification
            recognized_words.append(word)

# Write recognized words to output file
with open('output.txt', 'w') as file:
    for word in recognized_words:
        file.write(f"{word}\n")

print("Recognition and logging complete.")
