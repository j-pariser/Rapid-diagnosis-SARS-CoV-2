import pandas as pd
import gamry_parser as parser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.ticker as ticker
import os
from scipy.signal import argrelmin
from scipy.signal import savgol_filter
import scipy
import re


"""
basicPeaks: the same as peaks() but specific for electrode 4 batch
"""
def basicPeaks(filesList, electrodeFolder):
    peakHeights = []
    n = 0
    while n < len(filesList):
        filename = filesList[n]
        gp = parser.GamryParser()
        entryPath = os.path.join(electrodeFolder, filename)
        gp.load(filename=entryPath)
        data = gp.get_curve_data()

        yVals = data['Idif']  # units A
        xVals = data['Vfwd']  # units = V
        # plt.figure()
        # plt.title('Raw %s Data' %(filename))
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.scatter(xVals,yVals)

        # Remove non significant data points
        newY = yVals[10:None]
        newX = xVals[10:None]
        # plt.figure()
        # plt.title('% s Data Without First 5 Points' %filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.plot(newX,newY)
        # plt.xlim([xVals[0],xVals[len(xVals)-1]])

        # Add indices to numpy series to allow for indexing later
        ind = list(range(0, len(newY)))
        newY.index = [ind]
        newX.index = [ind]

        # Smooth out raw data
        smoothY = scipy.signal.savgol_filter(newY, 25, 3)
        # plt.figure()
        # plt.title('Smoothed %s Data' % filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.plot(newX, smoothY)

        # # Find local "minimas" for baseline
        firstDeriv = np.gradient(smoothY)
        firstDeriv = scipy.signal.savgol_filter(firstDeriv, 25, 3)
        midpoint = round((len(firstDeriv) - 1) / 2)
        rightHalf = firstDeriv[midpoint:len(firstDeriv) - 1]
        leftHalf = firstDeriv[0:midpoint]
        # plt.figure()
        # plt.title('Derivative of DPV %s Data' % filename)
        # plt.xlabel('Vfwd [V]')
        # plt.ylabel('Idif [A]')
        # plt.plot(newX, firstDeriv)

        # Peak height calculation
        newPeakHeightInd = (np.abs(rightHalf-0)).argmin()
        newPeakHeightInd = newPeakHeightInd + len(leftHalf)
        newPeakHeight = smoothY[newPeakHeightInd]
        #plt.figure()
        #plt.title('%s Data with Baseline Peak' % filename)
        #plt.xlabel('Vfwd [V]')
        #plt.ylabel('Idif [A]')
        #plt.plot(newX, smoothY)
        #plt.plot(newX[newPeakHeightInd], smoothY[newPeakHeightInd], '*')
        #plt.annotate(newPeakHeight, (newX[newPeakHeightInd], newPeakHeight))
        nA = round(newPeakHeight * (10 ** 9), 3)
        # print('Peak height relative to the baseline is', nA, 'nA')
        peakHeights.append(nA)
        n+=1
    return peakHeights

def basicPeaksMatrix(dataType,electrodesFolders,experimentConditions):
    allPeaks= [] # Store the peaks for the each electrode
    for electrode in electrodesFolders: # for loop to go through all electrodes
        filesList = extractDataType(electrode, dataType)
        allPeaks.append(basicPeaks(filesList, electrode)) # Adds electrode peaks to matrix
    df_peaks = pd.DataFrame(allPeaks,columns=experimentConditions) # Converts matrix to dataframe where columns=conditions, rows=electrodes
    return df_peaks

"""
peaksMatrix: returns the array of peaks for multiple electrodes in the form of a data frame (AKA matrix)
inputs: dataType --> measurement type being analyzed (DPV, SWV, ACV)
        electrodesFolders --> list of electrode file paths
        experimentConditions --> list of conditions
        methodNum --> method for baseline calculation
output: df_peaks --> returns a dataframe of all electrode peaks
"""
def peaksMatrix(dataType, frequency, electrodesFolders, experimentConditions, df):
    allPeaks= [] # Store the peaks for the each electrode
    for i, electrode in enumerate(electrodesFolders): # for loop to go through all electrodes
        filesList = extractDataType(electrode, dataType,frequency)
        method1 = df.iloc[i, 0]
        fraction1 = df.iloc[i, 1]
        method2 = df.iloc[i, 2]
        fraction2 = df.iloc[i, 3]
        # print(method1)
        # print(method2)
        allPeaks.append(peaks(filesList, electrode, method1, fraction1, method2, fraction2)) # Adds electrode peaks to matrix
    df_peaks = pd.DataFrame(allPeaks, columns=experimentConditions) # Converts matrix to dataframe where columns=conditions, rows=electrodes
    return df_peaks

"""
extractDataType: 
inputs: electrodeFolder --> filepath of folder containing DTA files
        dataType --> measurement type being analyzed (DPV, SWV, ACV)
output: filesList --> returns a list of files of interest
"""
import natsort

def extractDataType(electrodeFolder, dataType, frequency):
    allFiles = os.listdir(electrodeFolder) # All the files in folder
    filesList = []
    for file in allFiles:  # Loop through each file in folder
        if file.endswith('.dta') or file.endswith('.DTA'):  # Only take DTA files
            if dataType in file and frequency in file: # checks if data type (DPV, etc.) is in file name
                if ('#') not in file:
                    if '0.01' not in file and '0_01' not in file:
                        filesList.append(file) # adds file name to list if it is
    for i, entry in enumerate(filesList):
        if 'aptmch' in entry:
            new = entry.replace('aptmch', '0.0 fgml')
            filesList[i] = new
            os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
        if '0_1' in entry:
            new = entry.replace('0_1', '0.10')
            filesList[i] = new
            os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
        # if '0.1' in entry:
        #     new = entry.replace('0.1', '0.10')
        #     filesList[i] = new
        #     os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
        # if '0 fgmL' in entry:
        #     new = entry.replace('0fgml', '0.0 fgml')
        #     filesList[i] = new
        #     os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
        if '0_01' in entry:
            new = entry.replace('0_01', '0.01 ')
            filesList[i] = new
            os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
        if '0_05' in entry:
            new = entry.replace('0_05', '0.05')
            filesList[i] = new
            os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
        if '0_5' in entry:
            new = entry.replace('0_5', '0.50')
            filesList[i] = new
            os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
        if '0.5 fgmL' in entry:
            new = entry.replace('0.5', '0.50')
            filesList[i] = new
            os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))

    filesList = natsort.natsorted(filesList)
    print(filesList)
    return filesList

# Generate file list of all frequencies of a single concentration
def extractConcentration(electrodeFolder, dataType,concentration):
    allFiles = os.listdir(electrodeFolder) # All the files in folder
    filesList = []
    for file in allFiles:  # Loop through each file in folder
        if file.endswith('.dta') or file.endswith('.DTA'):  # Only take DTA files
            if dataType in file and concentration in file: # checks if data type (DPV, etc.) is in file name
                #if re.search(r'\^' + concentration + r'\$', file):
                if '_500Hz'not in file and '#' not in file:
                    filesList.append(file) # adds file name to list if it is
    # for i, entry in enumerate(filesList):
    #     if 'aptMCH' in entry:
    #         new = entry.replace('aptMCH', '0fgml')
    #         filesList[i] = new
    #         os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
    filesList = natsort.natsorted(filesList)
    #print(filesList)
    return filesList

def frequencies(filesList, electrodesFolder,dataType):
    if dataType == 'SWV':
        gp = parser.GamryParser()
        freq_list = []
        for i, entry in enumerate(filesList):
            entryPath = os.path.join(electrodesFolder, entry)
            gp.load(filename=entryPath)
            header = gp.get_header()
            freq_list.append(header['FREQUENCY'])
    else:
        freq_list = [25,50,75]
    return freq_list

# check for concentration (0 fg/ml)
    # get electrode with only that data (electrode + specific concentration)
    # find peaks for it, append
    # my goal: take after-before @ same elctrode and same frequency
def normFreq(electrodesFolders, dataType, concentrations, df, electrodeNames,switch):
    normalized = []
    allPeaks = []
    i = 0
    while i < len(df.index) - 1:
        for z, electrode in enumerate(electrodesFolders):
            print(electrode)
            currentPeaks = []
            # r'\b' + words + r'\b'
            for concentration in concentrations:
                # concentration = electrodeNames[z] + concentration
                fileList = extractConcentration(electrode, dataType, concentration)
                print(fileList)
                print(i,electrode,concentration)
                freq_list = frequencies(fileList, electrode,dataType)
                method1 = df.iloc[i, 0]
                fraction1 = df.iloc[i, 1]
                method2 = df.iloc[i, 2]
                fraction2 = df.iloc[i, 3]
                print(method1, method2)
                currentPeaks.append(peaks(fileList, electrode, method1, fraction1, method2, fraction2,switch))
                i = i + 1
            allPeaks.append(currentPeaks)
            currentDf = pd.DataFrame(currentPeaks, columns = freq_list)
            currentNorm = ((currentDf.loc[1] - currentDf.loc[0])/currentDf.loc[0])*100
            normalized.append(currentNorm)
        allPeaks = pd.DataFrame(allPeaks)
    print("Raw Peaks [nA] \n", allPeaks)
    norm = pd.DataFrame(normalized, columns = freq_list)
    return currentDf,norm, freq_list

def plotFreqN(norm, freq_list,electrodeNames):
    r, c = norm.shape
    zero = []
    for x in list(range(c)):
        zero.append(0)
    plt.rcParams['ytick.labelsize'] = 25
    plt.rcParams['xtick.labelsize'] = 25
    plt.figure()
    plt.title('Change in Signal After S1', fontsize='28')
    plt.xlabel('Frequency (Hz)', fontsize='24')
    plt.ylabel('Normalized Change (%)', fontsize='24')
    for i in list(range(r)):
        print(i)
        plt.plot(freq_list, norm.loc[i], marker = 'o', linestyle = '-', linewidth=2, markersize =8)
    plt.plot(freq_list, zero, linestyle='--', marker='', color='lightgray')
    plt.legend(electrodeNames, fontsize = '18')

    averages = stats(norm)
    avgs = averages.iloc[0, :]
    std = averages.iloc[1, :]
    plt.figure()
    plt.plot(freq_list, zero, linestyle='--', marker='', color='lightgray')
    plt.plot(freq_list, avgs, marker='o',linewidth=3)
    plt.errorbar(freq_list, avgs, yerr=std, ecolor='g', linestyle='', capsize=4, linewidth=3)
    plt.title('Average Change in Signal After CSF (n=3)', fontsize='28')
    plt.xlabel('Frequency (Hz)', fontsize='24')
    plt.ylabel('Normalized Change (%)', fontsize='24')


def plotFreqNOverlay(normA, freq_listA, normB, freq_listB):
    legend_names = ['CSF','S1']
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24

    rA, cA = normA.shape
    averagesA = stats(normA)
    averagesA = pd.DataFrame(averagesA)
    print('\nAveraged CSF\n',averagesA)
    avgsA = averagesA.iloc[0, :]
    stdA = averagesA.iloc[1, :]
    plt.figure()
    plt.plot(freq_listA, avgsA, marker='o', linewidth = 2, color = 'fuchsia', markersize = 6)
    plt.errorbar(freq_listA, avgsA, yerr=stdA, ecolor='fuchsia', linestyle='', capsize=5, linewidth = 3)

    rB, cB = normB.shape
    averagesB = stats(normB)
    averagesB = pd.DataFrame(averagesB)
    print('\nAveraged S1\n',averagesB)
    avgsB = averagesB.iloc[0, :]
    stdB = averagesB.iloc[1, :]
    plt.plot(freq_listB, avgsB, marker='o', linewidth = '2', color = 'k', markersize = 6)
    plt.errorbar(freq_listB, avgsB, yerr=stdB, ecolor='k', linestyle='', capsize=5, linewidth=3)

    # rC, cC = normC.shape
    # averagesC = stats(normC)
    # avgsC = averagesC.iloc[0, :]
    # stdC = averagesC.iloc[1, :]
    # plt.plot(freq_listC, avgsC, marker='o', linewidth='3', color='g', markersize=6)
    # plt.errorbar(freq_listC, avgsC, yerr=stdC, ecolor='g', linestyle='', capsize=5, linewidth=3)

    # 0 dotted line
    zero = []
    for x in list(range(cA)):
        zero.append(0)

    plt.plot(freq_listA, zero, linestyle = '--', marker = '', color = 'lightgray')

    plt.title('DPV Mini Cells After CSF and S1 (n=3)', fontsize='28')
    plt.xlabel('Frequency (Hz)', fontsize='24')
    plt.ylabel('Normalized Signal Change (%)', fontsize='24')
    plt.legend(legend_names, fontsize = '18')

def plotFreq(electrodesFolders, dataType,concentrations, df):
    peaksMatrix = []
    i = 0
    while i < len(df.index)-1:
        for concentration in concentrations:
            for electrode in electrodesFolders:
                fileList = extractConcentration(electrode, dataType,concentration)
                freq_list = frequencies(fileList, electrode)
                print(i,electrode)
                #print(concentration)
                method1 = df.iloc[i, 0]
                fraction1 = df.iloc[i, 1]
                method2 = df.iloc[i, 2]
                fraction2 = df.iloc[i, 3]
                peaksMatrix.append(peaks(fileList, electrode, method1, fraction1,method2,fraction2))
                i=i+1
    plt.figure()
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.plot(freq_list, peaksMatrix[0], '-ro', label = 'Wrinkled 1_15 w/ aptamer' )
    plt.plot(freq_list, peaksMatrix[4], '--ro', label = 'Wrinkled 1_15 w/ MCH')
    plt.plot(freq_list, peaksMatrix[1], '-go', label = 'Wrinkled 1_19 w/ aptamer')
    plt.plot(freq_list, peaksMatrix[5], '--go', label='Wrinkled 1_19 w/ MCH')
    plt.plot(freq_list, peaksMatrix[2], '-bo', label = 'Wrinkled 1_20 w/ aptamer')
    plt.plot(freq_list, peaksMatrix[6], '--bo', label = 'Wrinkled 1_20 w/ MCH')
    plt.plot(freq_list, peaksMatrix[3], '-mo', label='Wrinkled 1_21 w/ aptamer')
    plt.plot(freq_list, peaksMatrix[7], '--mo', label='Wrinkled 1_21 w/ MCH')
    plt.title('SWV Frequency Map Wrinkled Electrodes', fontsize = '22')
    plt.xlabel('Frequencies (Hz)', fontsize = '18')
    plt.ylabel('Peak Current (nA)', fontsize = '18')
    plt.legend( fontsize = '14')
    return peaksMatrix

"""
peaks: calculates the peaks of all the DTA files in the electrode folder
inputs: filesList --> list of DTA files of interest
        electrodeFolder --> filepath of folder containing DTA files
        methodNum --> method for baseline calculation 
outputs: filesList --> returns a list of files of certain dataType 
"""
def peaks(filesList, electrodeFolder, method1, fraction1, method2, fraction2,switch):
   peakHeights = []
   n = 0
   while n < len(filesList):
       filename = filesList[n]
       gp = parser.GamryParser()
       entryPath = os.path.join(electrodeFolder, filename)
       gp.load(filename=entryPath)
       data = gp.get_curve_data()

       yVals = data['Idif']  # units A
       xVals = data['Vfwd']  # units = V
       # plt.figure()
       # plt.title('Raw %s Data' %(filename))
       # plt.xlabel('Vfwd [V]')
       # plt.ylabel('Idif [A]')
       # plt.scatter(xVals,yVals)

       # Remove non significant data points
       newY = yVals[10:None]
       newX = xVals[10:None]
       # plt.figure()
       # plt.title('% s Data Without First 5 Points' %filename)
       # plt.xlabel('Vfwd [V]')
       # plt.ylabel('Idif [A]')
       # plt.plot(newX,newY)
       # plt.xlim([xVals[0],xVals[len(xVals)-1]])

       # Add indices to numpy series to allow for indexing later
       ind = list(range(0, len(newY)))
       newY.index = [ind]
       newX.index = [ind]

       # Median Filter
       newY = scipy.ndimage.filters.median_filter(newY, size=7)
       # plt.figure()
       # plt.title('Smoothed %s Data' % filename)
       # plt.xlabel('Vfwd [V]')
       # plt.ylabel('Idif [A]')
       # plt.scatter(newX, newY)

       # Smooth out raw data
       smoothY = scipy.signal.savgol_filter(newY, 11, 3)
       # plt.figure()
       # plt.title('Smoothed %s Data' % filename)
       # plt.xlabel('Vfwd [V]')
       # plt.ylabel('Idif [A]')
       # plt.plot(newX, smoothY)

       # # Find local "minimas" for baseline
       firstDeriv = np.gradient(smoothY)
       firstDeriv = scipy.signal.savgol_filter(firstDeriv, 25, 3)
       # plt.figure()
       # plt.title('Derivative of DPV %s Data' % filename)
       # plt.xlabel('Vfwd [V]')
       # plt.ylabel('Idif [A]')
       # plt.plot(newX, firstDeriv)

       secondDeriv = np.gradient(firstDeriv)
       secondDeriv = scipy.signal.savgol_filter(secondDeriv,25,3)
       # plt.figure()
       # plt.title('Second Derivative of DPV %s Data' %filename)
       # plt.xlabel('Vfwd [V]')
       # plt.ylabel('Idif [A]')
       # plt.plot(newX,secondDeriv)

       index = np.where(firstDeriv == 0)[0]
       midpoint = round((len(firstDeriv) - 1) / 2)

       a = 0
       b = 15 #30
       if  switch:
           a = 0
           b = 10  #20


        # Method Calibration
       # Using two methods
       if method2:
           if n <= a: # Low concentration = Method 1
               methodL = method1[0]
               methodR = method1[1]
               leftFrac = fraction1[0]
               rightFrac = fraction1[1]
           else: # High concentration = Method 2
               methodL = method2[0]
               methodR = method2[1]
               leftFrac = fraction2[0]
               rightFrac = fraction2[1]
       # Using one method
       else:
           methodL = method1[0]
           methodR = method1[1]
           leftFrac = fraction1[0]
           rightFrac = fraction1[1]

       # Left Half of Derivative Graph
       offset = b
       leftHalf = firstDeriv[0:midpoint - offset]  # left half of slopes
       if methodL == 1: # Use endpoint
           min1 = 0
       elif methodL == 2: # Use point with slope closest to zero
           # posleftHalf = abs(leftHalf) # Magnitude of slopes
           # min1 = np.where(posleftHalf == np.amin(posleftHalf))[0][0]  # index of slope closest to 0
           min1 = (np.abs(leftHalf - 0)).argmin()
       else: # Use point where slope a fraction of the most negative slope
           minSlope = np.amin(leftHalf)  # Find the most negative slope
           min1 = (np.abs(leftHalf - (minSlope * leftFrac))).argmin()

       # Right Half of Derivative Graph
       rightHalf = firstDeriv[midpoint+offset:len(smoothY) - 1]
       if methodR == 1: # Use endpoint
           min2 = len(firstDeriv)-1
       elif methodR == 2: # Use point with slope closest to zero
           posrightHalf = abs(rightHalf) # Magnitude of slopes
           # rightMin = np.where(posrightHalf == np.amin(posrightHalf))[0][0]  # index of slope closest to 0
           rightMin = (np.abs(rightHalf-0)).argmin()
           min2 = rightMin + len(leftHalf) + offset + offset +1 # Adjust index in respect to firstDeriv
       else: # Use point where slope a fraction of the most negative slope
           minSlope = np.amin(rightHalf)  # Find the most negative slope
           rightMin = (np.abs(rightHalf - (minSlope * rightFrac))).argmin()
           min2 = rightMin + len(leftHalf) + offset + offset +1  # Adjust index in respect to firstDeriv

       val1 = smoothY[min1]
       val2 = smoothY[min2]
       val1x = newX[min1].values[0]
       val2x = newX[min2].values[0]

       # Create baseline from local minimas
       m = (val2 - val1) / (val2x - val1x)
       b = val1 - m * val1x
       baseline = m * newX + b

       # Find number of intersections
       # intersections = np.argwhere(np.diff(np.sign(smoothY - baseline))).flatten()
       # print('Minima 1 @', val1x, 'V,', val1, 'A')
       # print("Minima 2 @", val2x, 'V,', val2, "A")

       # Peak height calculation relative to baseline
       difference = [ogVals - baseVals for ogVals, baseVals in zip(smoothY, baseline)]
       newPeakHeight = np.amax(difference[min1:min2])  # max difference
       newPeakHeightInd = np.where(difference == newPeakHeight)[0][0]  # index of max difference

      # # Traditional peak height
      #  newPeakHeight = np.amax(smoothY[min1:min2])
      #  newPeakHeightInd = np.where(smoothY==newPeakHeight)[0][0]
      #  baseVal = baseline[newPeakHeightInd].values[0]
      #  newPeakHeight = newPeakHeight - baseVal
      #
       plt.figure()
       plt.title('%s Data' % filename)
       plt.xlabel('Vfwd [V]')
       plt.ylabel('Idif [A]')
       plt.plot(newX, smoothY)
       plt.plot(val1x, val1, marker='.')
       plt.plot(val2x, val2, marker='.')
       plt.plot(newX, baseline, linestyle='--')
       plt.plot(newX[newPeakHeightInd], smoothY[newPeakHeightInd], '*')
       plt.annotate(newPeakHeight, (newX[newPeakHeightInd], smoothY[newPeakHeightInd]))

       nA = round(newPeakHeight * (10 ** 9), 3)
       peakHeights.append(nA)
       n+=1
   return peakHeights

"""
normalize: Calculates the normalized signal change of peaks sequentially between conditions   
inputs: df_peaks --> dataframe of raw peaks for all electrodes
        referenceMeasurement --> index of column to be used in normalization calculation
outputs: stats --> dataframe of normalized signal change for all electrodes
"""
def normalize(df_peaks,referenceMeasurement,experimentConditions):
    normal_pks = []  # Store normalized peaks for each electrode
    colNames = df_peaks.columns[2:len(df_peaks.columns)]  # Keeps track of column names
    counter = referenceMeasurement
    while counter < len(df_peaks.columns):  # Iterate through columns of dataframe
        reference = np.array(df_peaks.iloc[:, referenceMeasurement])
        currentRow = np.array(df_peaks.iloc[:, counter])
        normalVal = ((currentRow - reference) / reference)*100  # Normalization Calculation
        normal_pks.append(np.abs(np.around(normalVal, 3)))  # Add normalized value to array
        counter += 1
    normal_pks = np.array(normal_pks).transpose()  # Switch rows and columns for proper dataframe size
    df_normal = pd.DataFrame(normal_pks, columns=experimentConditions)  # Convert matrix to dataframe
    return df_normal

"""
stats: Calculates the mean and standard deviation of peaks across electrodes
inputs: df_peaks --> dataframe of raw peaks for all electrodes
outputs: stats --> dataframe of means and standard deviations
"""
def stats(df_peaks):
    avg = []
    std = []
    objects = list(df_peaks.columns)  # gets column names
    for i, _ in enumerate(objects):
        column = df_peaks.iloc[:, i]
        avg.append(np.average(column))
        std.append(np.std(column))
        # std_error=np.divide(np.std(column), np.sqrt(3))
        # std.append(std_error)

    data = [avg, std]
    averages = pd.DataFrame(data, columns = objects)
    return averages

"""
plotSignalStability: Plots normalized signal change per measurement for multiple electrodes
inputs: normSignal --> dataframe of normalized signal for all electrodes
        electrodeNames --> list of electrode names
outputs: Prints a figure
"""
def plotSignalStability(normSignal, electrodeNames):
    row = len(normSignal.index)
    col = len(normSignal.columns)
    x = np.arange(1, col+1, 1) # Set x axis equal to total number of measurements
    plt.figure()
    for i in range(row):
        plt.plot(x, normSignal.iloc[i,:], marker = 'o', linestyle = '')
    plt.title('Stability of Signal', fontsize=15)
    plt.ylabel('Normalized Signal Change [%]', fontsize='14')
    plt.xlabel('# of Measurements', fontsize='14')
    plt.xticks(np.arange(1, col + 1, 1))
    plt.legend(electrodeNames, fontsize='14')


"""
Concentration: a work in progress
take in .DPV files, each with a different s1 concentration
user can input at what concentrations
but the x-axis is a log scale
x-axis: 0 , 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 10^0 concentration
what if just had the log space equal, but then renamed them?
"""
def plotConcentration(averages, concentrations, electrodeNames,frequency):
    # print(averages)
    objects = list(averages.columns)  # gets column names
    dfy = list(averages.iloc[0, :])  # access row 1 of the data frame (the peak heights)
    error = list(averages.iloc[1, :])
    #x_pos = np.arange(len(objects))  # finds number of bars
    plt.figure()  # creates figure
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24
    # print(dfy)
    plt.plot(concentrations, dfy, marker = 'o', markersize = '12', linestyle = '', color='m', markeredgecolor='k', markeredgewidth=1.0)  # plots graph
    plt.errorbar(concentrations, dfy, yerr = error, ecolor = 'k', linestyle = '', capsize = 3)
    plt.xlabel('Concentration S1 (fg/mL)', fontsize = '18')
    plt.ylabel('Normalized change in signal (%)', fontsize = '18')  # gives y-label for bar graph
    #plt.ylabel('Change normalized to 1 fg/mL (%)', fontsize='30')  # gives y-label for bar graph
    plt.title('Change in Signal at %s (n=3)' %frequency, fontsize='24', fontweight = 'bold')  # display title
    #plt.legend(conditions, fontsize = '16')
    plt.xscale('symlog', linthresh=1e-1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlim(-0.01, 150)
    plt.yticks(np.arange(0,90,10))
    #plt.xlim(0.75, 15000)

def plotPeaks(rawPeaks, concentrations, electrodeNames, frequency):
    rawPeaks= rawPeaks.iloc[0,:]
    plt.figure()  # creates figure
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.scatter(concentrations, rawPeaks)  # plots graph
    plt.plot(concentrations, rawPeaks)
    #plt.errorbar(concentrations, dfy, yerr = error, ecolor = 'k', linestyle = '', capsize = 5)
    plt.xlabel('Concentration S1 (fg/mL)', fontsize = '10')
    plt.ylabel('Peak Height (nA)', fontsize = '10')  # gives y-label for bar graph
    #plt.ylabel('Change normalized to 1 fg/mL (%)', fontsize='30')  # gives y-label for bar graph
    plt.title('SWV %s Peak Height (%s)'% (electrodeNames, frequency), fontsize='15', fontweight = 'bold')  # display title
    #plt.legend(conditions, fontsize = '16')
    plt.xscale('symlog', linthresh=1e-1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlim(-0.01, 150)

def rawOverlay(electrodeFolders,frequency,dataType,electrodeNames,methods):
    rawVals = []
    baselineVals = []
    peakVals =  []
    for i,electrode in enumerate(electrodeFolders):
        filesList = extractDataType(electrode,dataType,frequency)
        print(filesList)
        method1 = methods.iloc[i, 0]
        fraction1 = methods.iloc[i, 1]
        method2 = methods.iloc[i, 2]
        fraction2 = methods.iloc[i, 3]
        plt.figure()
        n = 0
        while n < len(filesList):
            filename = filesList[n]
            gp = parser.GamryParser()
            entryPath = os.path.join(electrode, filename)
            gp.load(filename=entryPath)
            data = gp.get_curve_data()

            yVals = data['Idif']  # units A
            xVals = data['Vfwd']  # units = V

            # Remove non significant data points
            newY = yVals[10:None]
            newX = xVals[10:None]

            # Add indices to numpy series to allow for indexing later
            ind = list(range(0, len(newY)))
            newY.index = [ind]
            newX.index = [ind]

            # Median Filter
            newY = scipy.ndimage.filters.median_filter(newY, size=10)

            # Smooth out raw data
            smoothY = scipy.signal.savgol_filter(newY, 11, 3)
            rawVals.append(newX)
            rawVals.append(smoothY)

            # Find local "minimas" for baseline
            firstDeriv = np.gradient(smoothY)
            firstDeriv = scipy.signal.savgol_filter(firstDeriv, 25, 3)

            index = np.where(firstDeriv == 0)[0]
            midpoint = round((len(firstDeriv) - 1) / 2)

            a = 2
            b = 30
            # if switch:
            #     a = 1
            #     b = 20

            # Method Calibration
            # Using two methods
            if method2:
                if n <= a:  # Low concentration = Method 1
                    methodL = method1[0]
                    methodR = method1[1]
                    leftFrac = fraction1[0]
                    rightFrac = fraction1[1]
                else:  # High concentration = Method 2
                    methodL = method2[0]
                    methodR = method2[1]
                    leftFrac = fraction2[0]
                    rightFrac = fraction2[1]
            # Using one method
            else:
                methodL = method1[0]
                methodR = method1[1]
                leftFrac = fraction1[0]
                rightFrac = fraction1[1]

            # Left Half of Derivative Graph
            offset = b
            leftHalf = firstDeriv[0:midpoint - offset]  # left half of slopes
            if methodL == 1:  # Use endpoint
                min1 = 0
            elif methodL == 2:  # Use point with slope closest to zero
                # posleftHalf = abs(leftHalf) # Magnitude of slopes
                # min1 = np.where(posleftHalf == np.amin(posleftHalf))[0][0]  # index of slope closest to 0
                min1 = (np.abs(leftHalf - 0)).argmin()
            else:  # Use point where slope a fraction of the most negative slope
                minSlope = np.amin(leftHalf)  # Find the most negative slope
                min1 = (np.abs(leftHalf - (minSlope * leftFrac))).argmin()

            # Right Half of Derivative Graph
            rightHalf = firstDeriv[midpoint + offset:len(smoothY) - 1]
            if methodR == 1:  # Use endpoint
                min2 = len(firstDeriv) - 1
            elif methodR == 2:  # Use point with slope closest to zero
                posrightHalf = abs(rightHalf)  # Magnitude of slopes
                # rightMin = np.where(posrightHalf == np.amin(posrightHalf))[0][0]  # index of slope closest to 0
                rightMin = (np.abs(rightHalf - 0)).argmin()
                min2 = rightMin + len(leftHalf) + offset + offset + 1  # Adjust index in respect to firstDeriv
            else:  # Use point where slope a fraction of the most negative slope
                minSlope = np.amin(rightHalf)  # Find the most negative slope
                rightMin = (np.abs(rightHalf - (minSlope * rightFrac))).argmin()
                min2 = rightMin + len(leftHalf) + offset + offset + 1  # Adjust index in respect to firstDeriv

            val1 = smoothY[min1]
            val2 = smoothY[min2]
            val1x = newX[min1].values[0]
            val2x = newX[min2].values[0]

            # Create baseline from local minimas
            m = (val2 - val1) / (val2x - val1x)
            b = val1 - m * val1x
            baseline = m * newX + b
            baselineVals.append(baseline)

            # Peak height calculation relative to baseline
            difference = [ogVals - baseVals for ogVals, baseVals in zip(smoothY, baseline)]
            newPeakHeight = np.amax(difference[min1:min2])  # max difference
            newPeakHeightInd = np.where(difference == newPeakHeight)[0][0]  # index of max difference
            peakVals.append(newPeakHeightInd)

            if n ==0:
                plt.plot(newX, smoothY,color='k',linewidth= 2)
            else:
                plt.plot(newX, smoothY, color='b', linewidth = 2)
            plt.plot(newX[newPeakHeightInd], smoothY[newPeakHeightInd], '*')
            plt.annotate(newPeakHeight, (newX[newPeakHeightInd], smoothY[newPeakHeightInd]),size= 14)
            n+=1
    plt.title('SWV %s 50 Hz' % electrodeNames, size = 24)
    plt.xlabel('Vfwd [V]', size = 16)
    plt.ylabel('Idif [A]', size = 16)

    # Align baselines
    print(peakVals)
    baseDifference =np.subtract(baselineVals[0],baselineVals[1])
    newBase = np.subtract(baselineVals[0],baseDifference)
    new1 =np.subtract(rawVals[1],baseDifference)
    new1 = new1.to_numpy()
    print(new1,type(new1))
    new2 = rawVals[3]
    print(new2,type(new2))
    x1 = rawVals[0]
    x2 = rawVals[2]
    print(x1)
    print(x2)
    plt.figure()
    plt.plot(rawVals[0],new1,color='k',linewidth= 2)
    plt.plot(x1, baseline, linestyle='--',color= 'r')
    plt.plot(rawVals[2],new2,color='b',linewidth= 2)
    plt.plot(x2, newBase, linestyle='--',color = 'm')

    peakInd1 = peakVals[0]
    peakInd2 = peakVals[1]
    peak1 = new1[peakInd1]
    peak2 = new2[peakInd2]
    plt.plot(x1[peakInd1], peak1, '*')
    plt.plot(x2[peakInd2], peak2, '*')
    plt.annotate(peak1, (x1[peakInd1], peak1), size=14)
    plt.annotate(peak2, (x2[peakInd2], peak2), size=14)
    plt.title('SWV %s 50 Hz' % electrodeNames, size = 24)
    plt.xlabel('Vfwd [V]', size = 16)
    plt.ylabel('Idif [A]', size = 16)

"""
plotSelectivity: Plots bar graph of either raw or normalized signal change after being averaged
inputs: A -->
        B --> 
outputs: Prints a figure 
"""
def plotSelectivity(stats, plotTitle):
    objects = list(stats.columns)  # gets column names
    x_pos = np.arange(len(objects))  # finds number of bars
    avgs = stats.iloc[0, :]
    std = stats.iloc[1, :]

    # plt.figure()  # creates figure
    # plt.bar(x_pos, avgs, yerr=std)  # plots bar graph
    # plt.title(plotTitle)  # display title
    # plt.ylabel('Average Peak Height')  # gives y-label for bar graph
    # plt.xticks(x_pos, objects)  # plots bar names