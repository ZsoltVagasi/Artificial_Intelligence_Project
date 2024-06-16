import speech_recognition as sr

def speech_to_text_from_file(file_path, language='en-US'):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language=language)
        return text

def speech_to_text_from_mic(language='en-US'):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language=language)
            print(text)
            return text
        except:
            print("Sorry could not recognize your voice")
            return None
        
def speech_to_text_from_mic_to_file(output_file, language='en-US'):
    recorded_text = speech_to_text_from_mic(language)
    with open(output_file, 'w') as file:
        file.write(f"{recorded_text}\n")

# print(speech_to_text_from_mic('hu-HU'))  # For Hungarian