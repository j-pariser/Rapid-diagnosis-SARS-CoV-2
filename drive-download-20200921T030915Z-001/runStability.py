import codeStability as s
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


# Settings
electrodeNames = ['mini3sI','mini3sII'] # Change frequency here
dataType = 'SWV'
frequency = '_10Hz' # Must include underscore before number
experimentConditions = [1,2,3,4,5,6,7,8,9,10]
concentrations = [1,2,3,4,5,6,7,8,9,10]
save = False

#1 = endpoint
# 2 = smallest slope
# 3 = portion of the first derivative
electrode1 = [[3,3],[0.50,0.65],[],[0.35,0.70]]
electrode2 = [[3,3], [0.50, 0.65], [], [0.50,0.45]]
electrode3 = [[3,3], [0.50, 0.60], [], [0.30,0.70]]
electrode4 = [[3,3], [0.55, 0.90], [], [0.30,0.70]]
settings = [electrode1,electrode2]
df = pd.DataFrame(settings, columns = ['Method1', 'Fraction1', 'Method2', 'Fraction2'], index = electrodeNames)

# Import directory
electrodesFolders = []
for electrode in electrodeNames: # Gets the file path of all electrodes of interest
    electrodesFolders.append(os.path.join(os.getcwd(), electrode))
rawPeaks = s.peaksMatrix(dataType, frequency,electrodesFolders,experimentConditions,df) # Calculates the peaks for every electrode
print(rawPeaks)
#
# newPeaks = []
# newPeaks.append(np.array(rawPeaks.iloc[:,0]))
# i =0
# while i <len(rawPeaks.columns)-1:
#     reference = np.array(rawPeaks.iloc[:,i])
#     current = np.array(rawPeaks.iloc[:,i+1])
#     newPeaks.append(current-reference)
#     print(i)
#     i += 1
#     print(current,reference)
# newPeaks = np.array(newPeaks).transpose()  # Switch rows and columns for proper dataframe size
# newPeaks = pd.DataFrame(newPeaks)
# print(newPeaks)
s.plotPeaks(rawPeaks,concentrations,electrodeNames,frequency)
normSignal = s.normalize(rawPeaks,0,experimentConditions)
averages = s.stats(normSignal)
# s.plotConcentration(averages, concentrations,electrodeNames,frequency)

# Plot Raw Peaks
# s.plotPeaks(rawPeaks,concentrations,electrodeNames,frequency)
# Normalization
referenceMeasurement = 0  # Indicate index of measurement number to normalize to [NOTE: Python indexing starts at 0]
normSignal = s.normalize(rawPeaks,referenceMeasurement,experimentConditions)
print(normSignal)
# averages = s.stats(normSignal)
# print(averages)

s.plotSignalStability(normSignal,electrodeNames)


# Concentration Curve (x-axis = S1 concentration, y-axis = (averaged) signal change)
# s.plotConcentration(averages, concentrations,electrodeNames,frequency)


# Frequency Map
# s.plotFreq(electrodesFolders,dataType,electrodeNames,df)

# Save to File
if save:
    file = open('Signal Change 75 Hz.txt', 'a')
    file.write('\nNormalized Change [%] \n')
    file.write(str(averages.iloc[0])+'\n')
    file.write('\nStandard Error \n')
    file.write(str(averages.iloc[1]) + '\n')
    # file.write('\n'+str([electrodeNames,dataType,frequency])+ '\n')
    # file.write('\nRaw Peak Height [nA] \n')
    # file.write(str(rawPeaks.iloc[0])+'\n')
    # file.write('\nNormalized Signal Change [%] \n')
    # file.write(str(normSignal.iloc[0])+'\n')
    file.close()

plt.show()