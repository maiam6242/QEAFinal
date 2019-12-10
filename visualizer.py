import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt
import pylab as pl
from pydub import AudioSegment
from pydub.playback import play
from scipy import signal, fft, fftpack
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
        '''
        We ended up doing our filtering in matlab, but this method shows some of the many filters and ways in which we tried to filter in python. Ultimately, we decided that MATLAB would allow us more flexibilty to test different types of filters more quickly.
        '''
        num_array = AudioSegment.get_array_of_samples(self.music_clip)
        # print(num_array)
        print(np.shape(num_array))
        
        #hilbert transforms
        # plt.plot((np.array(num_array)))
        # # plt.plot(np.real(signal.hilbert(np.array(num_array))))
        # plt.show()
        
        # print(np.real(signal.hilbert(np.array(num_array))))
        # filtered = signal.hilbert(np.array(num_array))
        # plt.plot(filtered, label = 'hilbert filtering ')
        # plt.show(block = False)
        num_array = np.array(num_array)
        num_array = num_array/1000
        # act_num_array =[]
        # for row in np.real(signal.hilbert(np.array(num_array[1]))):
        #     print(row)
            # act_num_array.append(row[1])

        max_sound = np.asarray(num_array).max()
        print(np.asarray(num_array).max())
        print(max_sound)
        print((np.asarray(num_array).max() == max_sound))
        print(type(num_array))

        # hann window
        # print(num_array.si)
        # win = signal.hann(np.size(num_array/2), False)
        # # plt.plot(win)
        # # plt.show()
        # win = win*(max_sound)
        # # plt.plot(win)
        # # plt.show()
        # print(np.shape(num_array))
        # # win = signal.hann(20)
        # print(type(win))
        # # plt.plot(num_array)
        # # # # plt.show()
        # # plt.plot(win)
        # plt.show(block = False)
        
        # time = np.linspace(0,1,np.size(num_array))

        # high pass filters with fourier transforms
        # W = fftpack.fftfreq(np.size(num_array))
        # print(W)
        # f_signal = fftpack.rfft(num_array)

        # cut_f_signal = f_signal.copy()
        # cut_f_signal[(W>200000000000000000000)] = 0
        # cut_signal = fftpack.irfft(cut_f_signal)

        # plt.subplot(221)
        # plt.plot(time,num_array)
        # plt.subplot(222)
        # plt.plot(W,f_signal)
        # plt.xlim(0,1)
        # plt.subplot(223)
        # plt.plot(W,cut_f_signal)
        # plt.xlim(0,1)
        # plt.subplot(224)
        # plt.plot(time,cut_signal)
        # plt.show()

        # plt.plot(fftpack.rfft(num_array))

        # plt.show()

        fs = 44100
        t = np.arange(np.size(num_array))
        fc = 30  # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(7, w, 'high')
        output = signal.filtfilt(b, a, num_array)
        plt.plot(t, output, label='filtered')
        plt.legend()
        plt.show()
        
        # convoltional ffts to filter
        # filtered = np.fft.fftshift(signal.fftconvolve(num_array, win, mode = 'same'))
        # filtered = filtered/1000000
        # fw = signal.firwin(17,.11, pass_zero='highpass')
        # f3 = signal.sosfilt(fw, num_array)
        # plt.plot(f3)
        # plt.show()

        # butterworth filter
        # butter1, butter2 = signal.butter(15, .99, 'hp', output='ba', analog = True)
        # filtered2 = signal.filtfilt(butter1, butter2, num_array)
        # # plt.plot(num_array)
        # plt.plot(filtered2)
        
        # plt.show()

        # convolutional filter
        # conv = signal.convolve(num_array, win, mode = 'same')
        # plt.plot((np.array(num_array)), label = 'Signal')
        # plt.show(block = False)
        # plt.plot(win, label = 'hann window')
        # plt.show(block = False)
        # plt.plot(butter, label = 'fft convolve with hann filter')
        # # plt.show(block = False)
        # # plt.plot(conv/1000000, label = 'normal convolution')
        # plt.show()
        

        filt = write('output_fr.wav', 44100, filtered2)

        # filt = AudioSegment.from_numpy_array(filtered)
        print(filt)
        
        
    def run_fft(self):
        data = AudioSegment.get_array_of_samples(self.music_clip)
        play(self.music_clip)
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
        # plt.plot(freq, abs(fft_norm))
        print(max(np.real(fft_d))/100000)
        note = self.identify_note(max(np.real(fft_d))/100000)
        print(type(note))

        plt.title(note)
        plt.subplot(221)
        plt.plot(np.real(fft_norm))
        plt.title('fourier transform')
        plt.subplot(223)
        plt.plot(AudioSegment.get_array_of_samples(AudioSegment.from_wav('lamb.wav')))
        plt.title('raw entire signal')
        plt.subplot(224)
        plt.plot(AudioSegment.get_array_of_samples(AudioSegment.from_wav('convolution.wav')))
        plt.title('convolutional filter')
        plt.subplot(222)
        plt.plot(data)
        plt.title('raw discrete signal')
        plt.suptitle('Note: ' + note)
        
        # plt.legend()
        plt.show()
        pass

    def identify_note(self, num):
        notes = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30,  40.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53, 2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040.00, 7458.62, 7902.13]

        # Take in the max of the fft, then run through this
        fr_in = notes.index(min(notes,key=lambda x:abs(x-num)))
        print(min(notes,key=lambda x:abs(x-num)))

        notes_list = ['C0', 'C#0/Db0', 'D0', 'D#0/Eb0', 'E0', 'F0', 'F#0/Gb0', 'G0', 'G#0/Ab0', 'A0', 'A#0/Bb0', 'B0', 'C1', 'C#1/Db1', 'D1', 'D#1/Eb1', 'E1', 'F1', 'F#1/Gb1', 'G1', 'G#1/Ab1','A1', 'A#1/Bb1', 'B1', 'C2', 'C#2/Db2', 'D2', 'D#2/Eb2', 'E2', 'F2', 'F#2/Gb2', 'G2', 'G#2/Ab2', 'A2', 'A#2/Bb2', 'B2', 'C3', 'C#3/Db3', 'D3', 'D#3/Eb3', 'E3', 'F3', 'F#3/Gb3', 'G3','G#3/Ab3', 'A3', 'A#3/Bb3', 'B3', 'C4', 'C#4/Db4', 'D4', 'D#4/Eb4', 'E4', 'F4', 'F#4/Gb4', 'G4', 'G#4/Ab4', 'A4', 'A#4/Bb4', 'B4', 'C5', 'C#5/Db5', 'D5', 'D#5/Eb5', 'E5', 'F5', 'F#5/Gb5', 'G5', 'G#5/Ab5', 'A5', 'A#5/Bb5', 'B5', 'C6', 'C#6/Db6', 'D6', 'D#6/Eb6', 'E6', 'F6',  'F#6/Gb6', 'G6', 'G#6/Ab6', 'A6', 'A#6/Bb6', 'B6', 'C7', 'C#7/Db7', 'D7', 'D#7/Eb7', 'E7', 'F7', 'F#7/Gb7', 'G7', 'G#7/Ab7', 'A7', 'A#7/Bb7', 'B7', 'C8', 'C#8/Db8', 'D8', 'D#8/Eb8', 'E8', 'F8', 'F#8/Gb8', 'G8', 'G#8/Ab8', 'A8', 'A#8/Bb8', 'B8']
        print(str(notes_list[fr_in]))
        #ret val of notes thats closest, make another list that's note names, then find index of notes that right entry is and then find corresponding note
        return notes_list[fr_in]

if __name__ == "__main__":
    sp = SoundProcessing()
    # sp.filter()
    clips = sp.break_up_sound()

    for clip in clips:
        print(clip)
        clip_object = MusicClip(clip)
        # clip_object.filter()
        clip_object.run_fft()
        # break