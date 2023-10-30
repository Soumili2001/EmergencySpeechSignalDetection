import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model

fs=44100
seconds=2
filename="test.wav"
class_names=["Help detected","Help not Detected"]

model=load_model("saved_model/WWD1.h5")

print('Predicting.......')
i=0
while True:
    print("Speak Now:")
    myrecording=sd.rec(int(seconds*fs),samplerate=fs, channels=2)
    sd.wait()
    write(filename,fs,myrecording)

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed=np.mean(mfcc.T,axis=0)

    prediction=model.predict(np.expand_dims(mfcc_processed,axis=0))
    if prediction[:,1]>0.99:
        print(f"Help detected!!")
        print("Accuracy:",prediction[:,1])
        i+=1

    else:
        print(f"Help not Detected")
        print("Accuracy:",prediction[:,0])
