import concentrationCurveFunctions as s
import matplotlib.pyplot as plt
import os
import pandas as pd

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Settings '1_15','1_19','1_20','1_21'
electrodeNames = ['1_13','1_17','1_18']
dataType = 'SWV'
frequency = '_10Hz' # Must include underscore before number
experimentConditions = ['0 S1','0.05 fg/mL S1','0.1 fg/mL S1', '0.5 fg/mL S1', '1 fg/mL S1', '5 fg/mL S1', '10 fg/mL S1', '50 fg/mL S1', '100 fg/mL S1']
labels = [0,0.05, 0.1, 0.5, 1,5, 10, 50,100]
save = False

#1 = endpoint
# 2 = smallest slope
# 3 = portion of the first derivative
# a = aptamer/concentration 1
# b = MCh/concentration 2

# Standard Electrode Methods
# electrode1a = [[1,1], [0.40, 0.35], [], [0.50,0.50]]
# electrode1b = [[1,1], [0.40, 0.35], [], [0.50,0.50]]
# electrode2a = [[1,1], [0.40, 0.35], [], [0.50,0.50]]
# electrode2b = [[1,1], [0.40, 0.50], [], [0.50,0.50]]
# electrode3a = [[1,1], [0.40, 0.50], [], [0.50,0.50]]
# electrode3b = [[1,1], [0.40, 0.50], [], [0.50,0.50]]

# Wrinkled Electrode Methods
# electrode1a = [[1,3], [0.40, 0.60], [1,1], [0.50,0.50]]
# electrode1b = [[1,3], [0.40, 0.60], [1,1], [0.50,0.50]]
# electrode2a = [[1,3], [0.40, 0.60], [], [0.50,0.50]]
# electrode2b = [[1,3], [0.40, 0.60], [], [0.50,0.50]]
# electrode3a = [[1,3], [0.40, 0.60], [1,1], [0.50,0.50]]
# electrode3b = [[1,3], [0.40, 0.60], [1,1], [0.50,0.50]]
# electrode4a = [[1,3], [0.40, 0.60], [1,1], [0.50,0.50]]
# electrode4b = [[1,3], [0.40, 0.60], [1,1], [0.50,0.50]]

electrode1 = [[2,3],[0.40,0.60],[2,2],[0,0]]
electrode2 = [[2,3],[0.40,0.40],[1,1],[0,0]]
electrode3 = [[2,3],[0.40,0.50],[1,2],[0,0]]

# electrode1a,electrode1b,electrode2a,electrode2b,electrode3a,electrode3b,electrode4a,electrode4b
# settings = [electrode1a,electrode1b,electrode2a,electrode2b,electrode3a,electrode3b,electrode4a,electrode4b]
settings = [electrode1,electrode2,electrode3]
df = pd.DataFrame(settings, index = list(range(0,len(electrodeNames))), columns = ['Method1', 'Fraction1', 'Method2', 'Fraction2'])

# Import directory
electrodesFolders = []
for electrode in electrodeNames: # Gets the file path of all electrodes of interest
    electrodesFolders.append(os.path.join(os.getcwd(), electrode))

rawPeaks = s.peaksMatrix(dataType, frequency,electrodesFolders,experimentConditions,df) # Calculates the peaks for every electrode
print('Raw Peaks\n',rawPeaks)


# Plot Raw Peaks
# s.plotPeaks(rawPeaks,labels,electrodeNames,frequency)
#
# Normalization
referenceMeasurement = 0  # Indicate index of measurement number to normalize to [NOTE: Python indexing starts at 0]
normSignal = s.normalize(rawPeaks,referenceMeasurement,experimentConditions)
print('Normalized Change \n',normSignal)
#

averages = s.stats(normSignal)
print(averages)
print('\nStandard Error')
print(str(averages.iloc[1]) + '\n')

# Concentration Curve (x-axis = S1 concentration, y-axis = (averaged) signal change)
s.plotConcentration(averages, labels,electrodeNames,frequency)


# Frequency Map
# concentrations = [' 0 fgml',' 1 ngmL ']
# freqPeaks = s.plotFreq(electrodesFolders,dataType, concentrations,df)

# # Normalized, averaged frequency map
# s.plotFrequencyNavg(averages)


# Save to File
if save:
    file = open('082020 Peak Height Values 50 Hz.txt', 'a')
    file.write('\n'+str([electrodeNames,dataType,frequency])+ '\n')
    file.write('\nRaw Peak Height [nA] \n')
    file.write(str(rawPeaks)+'\n')
    file.write('\nNormalized Signal Change [%] \n')
    file.write(str(normSignal)+'\n')
    file.write('\nAveraged Normalized Signal [%] \n')
    file.write(str(averages.iloc[0]) + '\n')
    file.write('\nStandard Error \n')
    file.write(str(averages.iloc[1]) + '\n')
    file.close()

plt.show()
