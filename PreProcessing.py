import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sample="audio_data/help.wav"
data, sample_rate=librosa.load(sample)

plt.title("Wave form")
librosa.display.waveplot(data, sr=sample_rate)
plt.show()

mfccs=librosa.feature.mfcc(y=data, sr=sample_rate,n_mfcc=40)

print("Shape of mfcc:", mfccs.shape)

plt.title("MFCC")

librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.show()

all_data=[]

data_path_dict={
    0: ["background_sounds/" + file_path for file_path in os.listdir("background_sounds/")],
    1: ["audio_data/" + file_path for file_path in os.listdir("audio_data/")],
    }

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        data, sample_rate=librosa.load(single_file)
        mfccs=librosa.feature.mfcc(y=data, sr=sample_rate,n_mfcc=40)
        mfcc_processed=np.mean(mfccs.T,axis=0)
        all_data.append([mfcc_processed, class_label])
    print(f"INFO: Successfully preprocessed class Label {class_label}")

df=pd.DataFrame(all_data, columns=["feature", "class_label"])
df.to_pickle("final_audio_data_csv/audio_data1.csv")
