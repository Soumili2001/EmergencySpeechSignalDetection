###### IMPORTS ###################
import threading
import time
import sounddevice as sd
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3

#### SETTING UP TEXT TO SPEECH ###
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    engine.endLoop()


fs = 22050
seconds = 2

model = load_model("C:\\Users\\sarso\\Desktop\\HELP PROJECT\\saved_model\\WWD1.h5")
print('Predicting.......')


def listener():
    while True:
        print("Speak Now:")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        mfcc = librosa.feature.mfcc(y=myrecording.ravel(), sr=fs, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        prediction_thread(mfcc_processed)
        time.sleep(0.001)


def voice_thread():
    listen_thread = threading.Thread(target=listener, name="ListeningFunction")
    listen_thread.start()


def prediction(y):
    prediction = model.predict(np.expand_dims(y, axis=0))
    if prediction[:, 1] > 0.99:
        print(f"Help detected!!")
        if engine._inLoop:
            engine.endLoop()

        speak("Hello, What can I do for you?")
        print("Accuracy:",prediction[:,1])
    else:
        print(f"Help not Detected")    

    time.sleep(0.1)

def prediction_thread(y):
    pred_thread = threading.Thread(target=prediction, name="PredictFunction", args=(y,))
    pred_thread.start()

voice_thread()