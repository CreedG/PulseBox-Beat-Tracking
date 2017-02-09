from BeatFind import run_song
import sys
import os

path = "closed/"
def main(argv):
    #files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    #wavs = [x[:-4] for x in files if x[-4:] == ".wav"]
    #open_wavs = [x for x in wavs if x[:4] == "open"]
    #challenge_wavs = [x for x in wavs if x[:9] == "challenge"]
    #submitted_wavs = [x for x in wavs if x[:9] == "submitted"]
    #closed_wavs = [x for x in wavs if x[:6] == "closed"]
    wavs = "competition"
    print(run_song(wavs))

if __name__ == "__main__":
    main(sys.argv[1:])
