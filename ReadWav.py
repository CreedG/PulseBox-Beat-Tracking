#I3Creed - Starting point for our algorithm

import wave, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from time import sleep


#THE ALGORITHM

def beat_detect(wav):
    #Normally would use wav.readframes to read in some data and use the data
    #gathered so far to predict the next beat. Add that beat to the returned
    #array which will be plot against the known beats
    
    #Sample beat results (placeholder for algorithm)
    return [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
            26,27,28,29,30]

#END ALGORITHM


#SET UP THE PLOT

def main(argv):

    #Get the wave file set up (wave module does everything)
    if len(argv) != 1:
        print "You fuqqed up cuz... try again and specify the wav file as a program argument"
        return
    wav = wave.open(argv[0])
    print "Successfully read the wav file:",argv[0],wav.getparams()


    #Get the algorithm's beat times

    found_beats = beat_detect(wav)

    print("FOUND BEATS")
    print(found_beats)

    #Grab the known beat times from the text file with the same name

    known_beats = []

    txtfile_name = argv[0].split('.')[0]+".txt"

    with open(txtfile_name) as f:
        for line in f:
            known_beats.append(float(line))
            if 'str' in line:
                break

    print("CORRECT BEATS")
    print(known_beats)


    #Create the plots (3 stacked plots each showing 10 seconds)
    #This was annoying to set up but is best for showing more of the data
    #Don't worry about the details, it's just some matplotlib bullshit

    total_samples = wav.getnframes()

    display_div = 50
    
    sample_data = sample_arr = downsample_arr = x_axis_seconds = [0,0,0]
    plt.figure(1)
    for i in range(0,3):
        sample_data[i] = wav.readframes(total_samples/3)
        sample_arr[i] = np.fromstring(sample_data[i], dtype='int16')
        downsample_arr[i] = sample_arr[i][::display_div]
        plt.subplot(311+i)
        plt.plot(downsample_arr[i], color='#aaaaaa')
        ax = plt.gca()
        ax.xaxis.set_ticks([n*44100/display_div for n in range (0,11)])
        ticks = ax.get_xticks()/(44100/display_div)+i*10
        ax.set_xticklabels(ticks)
        plt.grid()



    #Add the beat indicators to the plots (known beats in red and beats found by
    #the algorithm in blue)

    for b in known_beats:
        if b <= 10:
            plt.subplot(311)
            xpos = 44100/display_div*b
        elif b <= 20:
            plt.subplot(312)
            xpos = 44100/display_div*(b-10)
        else:
            plt.subplot(313)
            xpos = 44100/display_div*(b-20)
        plt.plot([xpos,xpos],[-36000,39000], 'k-', lw=1.5, color='r')


    for b in found_beats:
        if b <= 10:
            plt.subplot(311)
            xpos = 44100/display_div*b
        elif b <= 20:
            plt.subplot(312)
            xpos = 44100/display_div*(b-10)
        else:
            plt.subplot(313)
            xpos = 44100/display_div*(b-20)
        plt.plot([xpos,xpos],[-39000,36000], 'k-', lw=1.5, color='b')


    plt.show()

    

if __name__ == "__main__":
    main(sys.argv[1:])

