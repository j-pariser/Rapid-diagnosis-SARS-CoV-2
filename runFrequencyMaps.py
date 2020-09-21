import frequencyMapFunctions as s
import matplotlib.pyplot as plt
import os
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Settings
electrodeNames = ['mini3sI','mini3sII','mini3sIII']
electrodeNames2 = ['mini3sIV', 'mini3sV', 'mini3sVI']
dataType = 'DPV'
frequency = '_50Hz' # Must include underscore before number
experimentConditions = ['0 S1','0.05 fg/mL S1', '0.1 fg/mL S1', '0.5 fg/mL S1', '1 fg/mL S1', '5 fg/mL S1', '10 fg/mL S1', '50 fg/mL S1', '100 fg/mL S1']
labels = [0, 0.05, 0.1, 0.5, 1,5, 10, 50,100]
save = False

# Mini Cell DPV
# Concentration = MCH (#1-3)
electrode1a1 = [[1,1],[0.55,0.60],[1,3],[0.40,0.50]]
electrode2a1 = [[1,3],[0.55,0],[1,1],[0.40,0.30]]
electrode3a1 = [[1,1],[0.55,0.50],[1,3],[0.40,0.50]]

# Concentration = CSF (#1-3)
electrode1b = [[1,1],[0.65,0.60],[],[0.40,0.40]]
electrode2b = [[1,3],[0.55,0.85],[1,1],[0.40,0.70]]
electrode3b = [[1,1],[0.55,0.90],[1,3],[0.40,0.60]]

# Concentration = MCH (#4-6)
electrode1a2 = [[1,3],[0.50,0.70],[1,3],[0.50,0.70]]
electrode2a2 = [[1,3],[0.50,0.50],[1,3],[0.50,0.60]]
electrode3a2 = [[1,3],[0.50,0.90],[1,1],[0.50,0.10]]

# Concentration = S1 (#4-6)
electrode1c = [[1,3],[0.50,0.70],[1,3],[0.50,0.70]]
electrode2c = [[1,3],[0.40,0.50],[1,3],[0.40,0.60]]
electrode3c = [[1,3],[0.50,0.60],[1,3],[0.40,0.70]]

# # Mini Cell SWV
# # Concentration = MCH (#1-3)
# electrode1a1 = [[1,3],[0.55,0.85],[1,3],[0.40,0.40]]
# electrode2a1 = [[3,3],[0.55,0.90],[1,1],[0.40,0.40]]
# electrode3a1 = [[3,3],[0.55,0.70],[1,1],[0.40,0.40]]
#
# # Concentration = CSF (#1-3)
# electrode1b = [[3,3],[0.65,0.80],[1,3],[0.40,0.40]]
# electrode2b = [[3,3],[0.55,0.90],[1,1],[0.40,0.40]]
# electrode3b = [[3,3],[0.55,0.90],[1,1],[0.40,0.40]]
#
# # Concentration = MCH (#4-6)
# electrode1a2 = [[3,3],[0.50,1.2],[3,3],[0.50,0.15]]
# electrode2a2 = [[3,3],[0.50,1.2],[3,3],[0.50,0.25]]
# electrode3a2 = [[3,3],[0.50,1.2],[3,3],[0.50,0.10]]
#
# # Concentration = S1 (#4-6)
# electrode1c = [[3,3],[0.50,1.0],[3,3],[0.50,0.15]]
# electrode2c = [[3,3],[0.40,1.2],[3,3],[0.40,0.10]]
# electrode3c = [[3,3],[0.50,1.2],[3,3],[0.40,0.10]]

#1 = endpoint
# 2 = smallest slope
# 3 = portion of the first derivative
# a = aptamer/concentration 1
# b = MCh/concentration 2


# Standard S1
# # Standard Electrode Methods - 0 fgml S1
# electrode1a1 = [[1, 1], [0.60, 0.60], [], [0.50,0.50]]
# electrode2a1 = [[1, 1], [0.40, 0.60], [], [0.50, 0.60]]
# electrode3a1 = [[1, 1], [0.40, 0.50], [], [0.50,0.50]]
#
# # Standard Electrode Methods - 100 fgml S1
# electrode1b = [[1, 1], [0.40, 0.60], [], [0.50, 0.60]]
# electrode2b = [[1, 1], [0.40, 0.60], [], [0.50 ,0.60]]
# electrode3b = [[1, 1], [0.40, 0.60], [], [0.50, 0.60]]
#
# # Standard Electrode Methods - 1 ngml S1
# electrode1c = [[1, 3], [0.40, 0.50], [1, 1], [0.50,0.50]]
# electrode2c = [[1, 1], [0.40, 0.60], [], [0.50 ,0.60]]
# electrode3c = [[2, 1], [0.70, 0.60], [], [0.50, 0.60]]

# Standard HSA
# Standard Electrode Methods - 0 fgml
# electrode1a2 = [[1, 1], [0.60, 0.55], [], [0.50,0.50]]
# electrode2a2 = [[1, 1], [0.40, 0.60], [], [0.50, 0.60]]
# electrode3a2 = [[1, 1], [0.40, 0.60], [], [0.50,0.50]]
#
# # Standard Electrode Methods - 100 fgml HSA
# electrode1d = [[1, 1], [0.40, 0.50], [], [0.50, 0.60]]
# electrode2d = [[1, 1], [0.40, 0.70], [], [0.50 ,0.60]]
# electrode3d = [[1, 1], [0.40, 0.60], [], [0.50, 0.60]]

# Wrinkled
# 1_15. 1_19, 1_20
# # Wrinkled Electrode Methods - 0 fgml
# electrode1a1 = [[1, 3], [0.60, 0.60], [1,1], [0.50,0.50]]
# electrode2a1 = [[1, 3], [0.40, 0.60], [], [0.50, 0.60]]
# electrode3a1 = [[1, 3], [0.40, 0.60], [1,1], [0.50,0.50]]
#
# # Wrinkled Electrode Methods - 100 fgml S1
# electrode1b = [[1, 2], [0.40, 0.50], [1,1], [0.50, 0.60]]
# electrode2b = [[1, 3], [0.40, 0.60], [], [0.50 ,0.60]]
# electrode3b = [[1,3], [0.40, 0.60], [1,1], [0.50, 0.60]]
#
# # Wrinkled Electrode Methods - 1 ngml S1
# electrode1c = [[1, 3], [0.40, 0.60], [1, 1], [0.50,0.50]]
# electrode2c = [[1, 3], [0.40, 0.60], [], [0.50 ,0.60]]
# electrode3c = [[1, 3], [0.70, 0.60], [1,1], [0.50, 0.60]]
#
# # 1_28, 1_29, 1_30
# # Wrinkled Electrode Methods - 0 fgml
# electrode1a2 = [[1, 3], [0.60, 0.55], [], [0.50,0.50]]
# electrode2a2 = [[1, 3], [0.40, 0.60], [1,1], [0.50, 0.60]]
# electrode3a2 = [[1, 3], [0.40, 0.60], [1,2], [0.50,0.50]]
#
# # Wrinkled Electrode Methods - 100 fgml HSA
# electrode1d = [[1, 2], [0.40, 0.50], [1,1], [0.50, 0.60]]
# electrode2d = [[1, 2], [0.40, 0.70], [], [0.50 ,0.60]]
# electrode3d = [[1, 2], [0.40, 0.60], [1,1], [0.50, 0.60]]


# 1_22,1_23,1_24 DPV
# 0 S1
# electrode1a1= [[1,3],[0.40,0.60],[1,2],[0.40,0.40]]
# electrode2a1 = [[1,2],[0.40,0.35],[1,3],[0.40,0.50]]
# electrode3a1  = [[1,3],[0.40,0.40],[1,2],[0.40,0.30]]

# 1000 fg S1
# electrode1b= [[1,3],[0.40,0.60],[1,2],[0.40,0.40]]
# electrode2b = [[1,3],[0.40,0.45],[1,3],[0.40,0.55]]
# electrode3b = [[1,1],[0.40,0.30],[1,3],[0.40,0.60]]

# 1_31, 1_34, 1_35 DPV
# 0 HSA
# electrode1a2= [[1,3],[0.40,0.40],[],[0.40,0.40]]
# electrode2a2 = [[1,3],[0.40,0.50],[],[0.40,0.40]]
# electrode3a2 = [[1,3],[0.40,0.35],[1,3],[0.40,0.50]]

# 1000 fg HSA
# electrode1c= [[1,1],[0.40,0.50],[1,3],[0.40,0.50]]
# electrode2c = [[1,3],[0.40,0.55],[1,3],[0.40,0.45]]
# electrode3c = [[1,1],[0.40,0.40],[1,3],[0.40,0.50]]

# 1_22,1_23,1_24 SWV
# 0 S1
# electrode1a1= [[2,2],[0.40,0.40],[],[]]
# electrode2a1 = [[2,3],[0.40,0.50],[2,1],[0.40,0.40]]
# electrode3a1  = [[2,2],[0.40,0.50],[],[0.40,0.40]]

# 1000 fg S1
# electrode1b= [[2,2],[0.40,0.40],[],[]]
# electrode2b = [[2,2],[0.40,0.40],[],[]]
# electrode3b = [[2,2],[0.40,0.30],[],[]]
#
# # 1_31, 1_34, 1_35 SWV
# # 0 HSA
# electrode1a2= [[2,3],[0.40,0.55],[2,2],[0.40,0.40]]
# electrode2a2 = [[2,3],[0.40,0.55],[2,1],[0.40,0.40]]
# electrode3a2 = [[1,3],[0.40,0.40],[1,1],[0.40,0.40]]
#
# # 1000 fg HSA
# electrode1c= [[2,3],[0.40,0.60],[2,1],[0.40,0.40]]
# electrode2c = [[2,3],[0.40,0.55],[2,3],[0.40,0.50]]
# electrode3c = [[1,1],[0.40,0.40],[],[]]

# s100 = [electrode1a1,electrode1b,electrode2a1,electrode2b,electrode3a1,electrode3b]
# df100 = pd.DataFrame(s100, index = list(range(0,len(electrodeNames)*2)), columns = ['Method1', 'Fraction1', 'Method2', 'Fraction2'])

s1 = [electrode1a1,electrode1b,electrode2a1,electrode2b,electrode3a1,electrode3b]
df1 = pd.DataFrame(s1, index = list(range(0,len(electrodeNames)*2)), columns = ['Method1', 'Fraction1', 'Method2', 'Fraction2'])

h100 = [electrode1a2, electrode1c, electrode2a2,electrode2c,electrode3a2,electrode3c]
dfh1 = pd.DataFrame(h100, index = list(range(0,len(electrodeNames2)*2)), columns = ['Method1', 'Fraction1', 'Method2', 'Fraction2'])

# Import directory
electrodesFolders = []
for electrode in electrodeNames: # Gets the file path of all electrodes of interest
    electrodesFolders.append(os.path.join(os.getcwd(), electrode))

electrodesFolders2 = []
for electrode in electrodeNames2: # Gets the file path of all electrodes of interest
    electrodesFolders2.append(os.path.join(os.getcwd(), electrode))

# rawPeaks = s.peaksMatrix(dataType, frequency,electrodesFolders,experimentConditions,df) # Calculates the peaks for every electrode
# print(rawPeaks)
#
# # Plot Raw Peaks
# s.plotPeaks(rawPeaks,labels,electrodeNames,frequency)
#
# # Normalization
# referenceMeasurement = 0  # Indicate index of measurement number to normalize to [NOTE: Python indexing starts at 0]
# normSignal = s.normalize(rawPeaks,referenceMeasurement,experimentConditions)
# print(normSignal)
#
# averages = s.stats(normSignal)
# print(averages)

# Concentration Curve (x-axis = S1 concentration, y-axis = (averaged) signal change)
# s.plotConcentration(averages, labels,electrodeNames,frequency)

# Frequency Map - CSF
concentrations1 = [' MCH ',' CSF ']
#freqPeaks = s.plotFreq(electrodesFolders,dataType, concentrations,df)
raw100, norm100, freq_list100 = s.normFreq(electrodesFolders, dataType, concentrations1, df1, electrodeNames,False)
s.plotFreqN(norm100, freq_list100,electrodeNames)
print('CSF Normalized Change [%] \n', norm100)

# Frequency Map - S1
concentrations2 = [' MCH ',' S1 ']
#freqPeaks = s.plotFreq(electrodesFolders,dataType, concentrations,df)
raw1, norm1, freq_list1 = s.normFreq(electrodesFolders2, dataType, concentrations2, dfh1, electrodeNames2,True)
s.plotFreqN(norm1, freq_list1,electrodeNames2)
print('S1 Normalized Change [%] \n', norm1)

# Frequency Map - 1000 fgml HSA
# concentrations3 = [' mch ',' 1000 fgmL HSA']
# #freqPeaks = s.plotFreq(electrodesFolders,dataType, concentrations,df)
# normHSA, freq_listHSA = s.normFreq(electrodesFolders, dataType, concentrations3, dfh1, electrodeNames2,False)
# print(normHSA)
# s.plotFreqN(normHSA, freq_listHSA,electrodeNames)
# #
s.plotFreqNOverlay(norm100, freq_list100,norm1, freq_list1)

# names3 = ['mini3sIII']
# folders3 = []
# for electrode in names3: # Gets the file path of all electrodes of interest
#    folders3.append(os.path.join(os.getcwd(), electrode))
# method1 = [[1,3],[0.40,0.40],[],[]]
# method2 = [[1,3],[0.40,0.40],[],[]]
# methods = [method1,method2]
# methods=pd.DataFrame(methods)
# s.rawOverlay(folders3,frequency,dataType,names3,methods)


# Save to File
if save:
    file = open('CSF and S1 Experiement Values.txt', 'a')
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
