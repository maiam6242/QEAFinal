import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

class SoundProcessing:
    
    def __init__(self):
        pass

    def record_sound(self):
        fs = 44100  # Sample rate
        seconds = 3  # Duration of recording

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, myrecording)  # Save as WAV file 

if __name__ == "__main__":
    sp = SoundProcessing()
    sp.record_sound()