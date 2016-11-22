import numpy as np




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




