import glob, os
import soundfile as sf
from math import sqrt

FILE_FORMAT = "flac"
def extract_audio():
    for video in glob.glob("videos/P*/*/*.avi"):
        f = video[:-3] + FILE_FORMAT
        # print(f)
        os.system("ffmpeg -v 0 -y -i "+video+" -f "+FILE_FORMAT+" -vn "+f)

def read_audio():
    l = []
    for audio in glob.glob("videos/P*/*/*."+FILE_FORMAT):
        data, samplerate = sf.read(audio)
        l.append(data.shape[0])

    av = sum(l)/len(l)
    s = 0

    for x in l:
        s += sqrt((x - av)**2)

    print(max(l), min(l), av, s/len(l))

extract_audio()
read_audio()
