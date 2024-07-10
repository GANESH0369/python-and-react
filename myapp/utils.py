# audio_converter/utils.py
import speech_recognition as sr
# from pdf2docx import Converter


def convert_audio_to_text(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"



# def convert_pdf_to_docx(pdf_bytes, output_file):
#     converter = Converter(pdf_bytes)
#     converter.convert(output_file)
#     converter.close()