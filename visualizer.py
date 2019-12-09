import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt
from pydub import AudioSegment
from scipy import signal, fft
from scipy.io.wavfile import write, read

class SoundProcessing:
    
    def __init__(self):
        pass

    def record_sound(self):
        fs = 44100  # Sample rate
        seconds = 90  # Duration of recording

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, myrecording)  # Save as WAV file 


    def break_up_sound(self):
        audio_chunk_list = []
        audio_file = "lamb.wav"
        audio = AudioSegment.from_wav(audio_file)
        print(type(audio))
        list_of_timestamps = [1, 2, 3, 4, 5 ,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] #and so on in *seconds*

        start = 0
        for  idx, t in enumerate(list_of_timestamps):
            #break loop if at last element of list
            if idx == len(list_of_timestamps):
                break

            end = t * 1000 #pydub works in millisec
            print("split at [{}:{}] ms".format((start/1000), (end/1000)))
            audio_chunk = audio[start:end]
            audio_chunk.export("audio_chunk_{}.wav".format(end), format="wav")

            start = end #pydub works in millisec     
            print(type(audio_chunk))  
            print(audio_chunk.max_dBFS)
            print(audio_chunk.max_possible_amplitude)
            audio_chunk_list.append(audio_chunk) 
        return audio_chunk_list

class MusicClip:

    def __init__(self, clip):
        self.music_clip = clip
        
    def filter(self):
        
        num_array = AudioSegment.get_array_of_samples(self.music_clip)
        # print(num_array)
        print(np.shape(num_array))
        
        # plt.plot((np.array(num_array)))
        # # plt.plot(np.real(signal.hilbert(np.array(num_array))))
        # plt.show()
        
        # print(np.real(signal.hilbert(np.array(num_array))))
        # filtered = signal.hilbert(np.array(num_array))
        # plt.plot(filtered, label = 'hilbert filtering ')
        # plt.show(block = False)
        num_array = np.array(num_array)

        # act_num_array =[]
        # for row in np.real(signal.hilbert(np.array(num_array[1]))):
        #     print(row)
            # act_num_array.append(row[1])

        max_sound = np.asarray(num_array).max()
        print(np.asarray(num_array).max())
        print(max_sound)
        print((np.asarray(num_array).max() == max_sound))
        print(type(num_array))
        # print(num_array.si)
        win = signal.hann(np.size(num_array/2))
        # plt.plot(win)
        # plt.show()
        win = win*(max_sound)
        # plt.plot(win)
        # plt.show()
        print(np.shape(num_array))
        # win = signal.hann(20)
        print(type(win))
        plt.plot(num_array)
        # # plt.show()
        plt.plot(win)
        plt.show(block = False)
        
        
        filtered = signal.fftconvolve(num_array, win, mode = 'same')
        filtered = filtered/1000000
        conv = signal.convolve(num_array, win, mode = 'same')
        plt.plot((np.array(num_array)), label = 'Signal')
        plt.show(block = False)
        plt.plot(win, label = 'hann window')
        plt.show(block = False)
        plt.plot(filtered, label = 'fft convolve with hann filter')
        plt.show(block = False)
        plt.plot(conv/1000000, label = 'normal convolution')
        plt.show()
        

        filt = write('output.wav', 44100, filtered)

        # filt = AudioSegment.from_numpy_array(filtered)
        print(filt)
        
        
    def run_fft(self):
        data = AudioSegment.get_array_of_samples(self.music_clip)
        fft_d = fft(data)
        fft_norm = fft_d/len(data)
        fft_norm = fft_norm[range(int(len(data)/2))]
        real_d = np.real(fft_d)
        
        tp = len(data)/1000
        vals = np.arange(int(len(data)/2))
        freq = vals/tp
        # hist_bins, hist_vals = self.music_clip.fft()
        # hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)
        # plt.plot(hist_bins / 1000, hist_vals_real_normed)
        # plt.xlabel("kHz")
        # plt.ylabel("dB")
        # plt.plot(hist_vals_real_normed)
        plt.plot(freq, abs(fft_norm))
        print(max(np.real(fft_d)))
        
        # plt.plot(np.imag(fft_d))
        plt.show()
        pass

    # def identify_note(self):


if __name__ == "__main__":
    sp = SoundProcessing()
    # sp.filter()
    clips = sp.break_up_sound()

    for clip in clips:
        print(clip)
        clip_object = MusicClip(clip)
        clip_object.filter()
        clip_object.run_fft()
        # break