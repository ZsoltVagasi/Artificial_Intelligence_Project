import my_speech_recognition as speech_recognition
import my_word_recognition as word_recognition
import numpy as np

if __name__ == "__main__":
    # Load and preprocess training data
    train_sentences = word_recognition.load_data('train.txt')
    tokenizer, padded_sequences = word_recognition.preprocess_data(train_sentences)
    labels = np.array([1] * len(train_sentences))  # Adjust according to your data

    # Build and train the model
    model = word_recognition.build_model(vocab_size=1000, input_length=padded_sequences.shape[1])
    word_recognition.train_model(model, padded_sequences, labels)

    # Record speech using the written tool

    speech_recognition.speech_to_text_from_mic_to_file('speech_output.txt','hu-HU')

    # Load new sentences for recognition
    new_sentences = word_recognition.load_data('speech_output.txt')

    # Recognize words and log them
    recognized_words = word_recognition.recognize_words(model, tokenizer, new_sentences, maxlen=padded_sequences.shape[1])
    word_recognition.log_recognized_words(recognized_words, 'output.txt')

    print("Recognition and logging complete.")