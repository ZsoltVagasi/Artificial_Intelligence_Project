import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def preprocess_data(sentences, num_words=10000, oov_token="<OOV>"):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post')
    return tokenizer, padded_sequences

def build_model(vocab_size, input_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 16, input_length=input_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, padded_sequences, labels, epochs=50):
    model.fit(padded_sequences, labels, epochs=epochs)

def recognize_words(model, tokenizer, new_sentences, maxlen):
    new_sequences = tokenizer.texts_to_sequences(new_sentences)
    new_padded_sequences = pad_sequences(new_sequences, padding='post', maxlen=maxlen)
    predictions = model.predict(new_padded_sequences)
    recognized_words = []
    for i, sequence in enumerate(new_sequences):
        if predictions[i] > 0.5:  # Assuming a threshold of 0.5 for binary classification
            recognized_words.extend([tokenizer.index_word[idx] for idx in sequence if idx in tokenizer.index_word])
    return recognized_words

def log_recognized_words(recognized_words, output_file):
    with open(output_file, 'w') as file:
        for word in recognized_words:
            if word != '<OOV>':
                file.write(f"{word}\n")

# Main script
if __name__ == "__main__":
    # Load and preprocess training data
    train_sentences = load_data('train.txt')
    tokenizer, padded_sequences = preprocess_data(train_sentences)
    labels = np.array([1] * len(train_sentences))  # Adjust according to your data

    # Build and train the model
    model = build_model(vocab_size=1000, input_length=padded_sequences.shape[1])
    train_model(model, padded_sequences, labels)

    # Load new sentences for recognition
    new_sentences = load_data('lorem_ipsum.txt')

    # Recognize words and log them
    recognized_words = recognize_words(model, tokenizer, new_sentences, maxlen=padded_sequences.shape[1])
    log_recognized_words(recognized_words, 'output.txt')

    print("Recognition and logging complete.")