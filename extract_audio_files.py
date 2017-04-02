import glob, os
import soundfile as sf
from math import sqrt
from scipy.io.wavfile import read as wavread
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd


from scipy.signal import butter, lfilter, freqz

FILE_FORMAT = "wav"
VIDEOS_LIST = "videos/P*/*/*.avi"

def extract_audio():
    """Extract the audio in mono format from the videos"""

    for video in glob.glob(VIDEOS_LIST):
        f = video[:-3] + FILE_FORMAT
        os.system("ffmpeg -v 0 -y -i "+video+" -f "+FILE_FORMAT+" -ar 10000 -vn "+f)

def read_audio():
    """Read the audio files extracted and plot them"""
    l = []
    for audio in glob.glob("videos/P*/*/*."+FILE_FORMAT):
        data, samplerate = sf.read(audio)

        # First, design the Buterworth filter
        N  = 3    # Filter order
        Wn = 0.1 # Cutoff frequency
        B, A = signal.butter(N, Wn, output='ba')
        smooth_data = signal.filtfilt(B,A, data)

        smooth_data = pd.rolling_mean(data, 10)
        plt.plot(data, 'b')

        plt.plot(smooth_data, 'r')
        # print(audio[:-4]+"_smooth.wav")
        # sf.write(audio[:-4]+"_smooth.wav", smooth_data, samplerate)
        # plt.savefig(audio[:-3]+"png")
        plt.show()
        # plt.clf()
        l.append(data.shape[0])

    av = sum(l)/len(l)
    s = 0

    for x in l:
        s += sqrt((x - av)**2)

    # print(max(l), min(l), av, s/len(l))
