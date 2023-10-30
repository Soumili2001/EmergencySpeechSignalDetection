import sounddevice as sd
from scipy.io.wavfile import write

def record_audio_and_save(save_path,n_times=50):
    input("To start audio recording press Enter: ")
    for i in range(n_times):
        fs=44100
        seconds=2
        myrecording = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i+100) +".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl+C ({i+1}/{n_times})")

def record_background_save(save_path, n_times=100):
    input("To start your background sounds press Enter: ")
    for i in range(n_times):
        fs=44100
        seconds=2
        myrecording = sd.rec(int(seconds*fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) +".wav", fs, myrecording)
        input(f"Currently on {i+1}/{n_times}")

print("Recording the wake word: \n")
record_audio_and_save("audio_data/")


#print("Recording the background sounds: \n")

#record_background_save("background_sounds/")
    