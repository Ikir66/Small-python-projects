import numpy as np


c = np.array([[13123,43243423,43423,33,3,2,22223,43434,324,33,24,223,33242434,4432423,23242342,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

#Function for calculating moving average for wanted time
def maVector (vector, matime, time):
    i = matime - 1
    maVector = np.zeros([1,time])
    while i < time:
        sum = 0
        j = i - matime + 1
        c = 0
        while c < matime:
            sum += vector[0,j]
            c += 1
            j += 1
        maVector[0,i] = sum/matime

        i += 1
    return maVector

print(maVector(c,5,33))
