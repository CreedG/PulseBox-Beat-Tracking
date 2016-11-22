import numpy as np


#This function will autocorrelate or cross correlate two vectors of peaks

#We need our own function because the data is made up of onset peaks at distinct times with variable intervals in between

#Therefore, the correlation can be (much) faster than correlating the entire signal because there are less data points
# (on the order of 100 points to correlate for each signal in a 30 second song

#It faster but also more difficult to deal with, since we must allow for a margin of error to end up with a discrete
#list of correlation strengths

#This will need to be optimized


#TODO someone please do this

#Basically just grouping data and weighted averages

'''Place the data into groups of possible "correct answers". A group can span values for a maximum
distance of max_group_size. confidence levels (z) are provided for each of the inputs. They basically
represent the voting power of that data element. The output should be the list of groups and their
combined confidence (1st place, 2nd place...) The group is given a value that is the average of the data that contributed to it'''

#e.g. we have 10 different guesses for period estimates
#They are, in milliseconds: 500, 656, 502, 698, 496, 356, 702, 501, 500, 654
#Confidence levels are:     .9   .6    .7  .5    .4   .4   .3   .2   .6   .6

#We should end up with groups with values around 500, 655, 700, 356
#The most values contributed to the 500 group, and it's combined confidence is highest so it wins
#Also list 655, 600, and 356 as 2nd 3rd and 4th place with their scores

#If we had data: 400, 500, 500
#Confidence       1,   .2,  .2
#Then the 400 group wins because of the confidence multiplier

#If we had data: 400, 420, 410
#Confidence       .4  .2   .4
#Then there is only one winning group if max_group_size is greater than 20. It's value is the weighted average 408


#data is list of possible things to choose from
#confidence is a vector of the same length as data describing a weighting for data at that index (0-1)
#max_group_size is the maximum difference two elements in the same group can have

'''The details of where to split groups might be tricky, I don't know how yet, just make sure clustered data is
included in the same group'''

def find_consensus(data,confidence,max_group_size):
    return






#IGNORE THIS, not currently working
def corr_peaks(a,b):

    x1vec = a[0]
    x2vec = b[0]
    y1vec = a[1]
    y2vec = b[1]

    print("aaa")
    print(len(x1vec))

    tolerance = 0.03

    shifts = []


    #STAGE 1, fill the shifts array to get the possible tempos

    for ix1, x1 in enumerate(x1vec):
        for ix2, x2 in enumerate(x2vec):

            print("comparing",x1,x2)

            dist = abs(x1-x2)
            if (dist < tolerance):
                break

            mult = y1vec[ix1]*y2vec[ix2]

            matched = False
            #add the shift component which may be slightly off from another OR a multiple, correct for these
            #target range is 80-160bpm which is 375 to 750ms so squash all multiples to fit
            for s in shifts:
                if (abs(dist - s[0]) < tolerance):
                    s[0] = (s[0]*s[1]+dist*mult)/(s[1]+mult)    #Shift it over
                    s[1] += mult   #Add the new component
                    matched = True
                    print("matched the bin",s[0],", new strength is",s[1])
                    break

            if (matched == False):
                print("creating a new bin for",dist )
                shifts.append([dist,mult])

            print("shifts is now",shifts)
            print()

    #Find period from the maximal correlation shift
    max = [0,0]
    for s in shifts:
        if s[1] > max[1]:
            max[1] = s[1]
            max[0] = s[0]

    T = max[0]


    #Two close contenders lowers the z (confidence) score

    #STAGE 2 get the actual beat times using the period... assuming it is correct.

    #Find phase from combing through the data at the period interval. Do this for each input to the correlation (or once if auto)

    #Use the n (3 for now) most recent peaks to test

    xlen = len(x1vec)

    best_phase_pow = 0
    best_phase = 0

    print(xlen)

    for start_idx in reversed(range(xlen-10,xlen)):
        #print("starting from",start_idx,"which is",x1vec[start_idx])
        total_pow = 0
        comb_idx = start_idx
        comb_sec = x1vec[start_idx]
        while(comb_idx >= 0 and comb_sec >= 0):
            #print("comb_sec =",comb_sec)
            comb_sec -= T
            while (x1vec[comb_idx] > comb_sec-.05 and comb_idx >= 0):
                if (abs(comb_sec-x1vec[comb_idx]) < .05):
                    #print("found match at comb_sec,x1vec[comb_idx]",comb_sec,x1vec[comb_idx])
                    total_pow += y1vec[comb_idx]
                    comb_sec = x1vec[comb_idx]  #Snap the value to get rid of accumulated period error (over a long time, may deviate)
                comb_idx -= 1
        #print("total pow for the phase is", total_pow)
        if (total_pow > best_phase_pow):
            best_phase_pow = total_pow
            best_phase = x1vec[start_idx]

    P = best_phase







    return T,P,0


if __name__ == "__main__":
    a = [[1,1.3,2.01,2.66,2.3,2.99,4.01,4.11,5,6],[5,3,7,10,2,2,4,6,5,6]]
    b = a

    t,p,z = corr_peaks(a,b)
    print(a)

    print(t,p,z)
