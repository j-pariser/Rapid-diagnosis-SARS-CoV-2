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
                if ('#') not in file and 'fgmL' in file:
                    if '0.01' not in file and '0_01' not in file:
                        # if '0.01' not in file and '0.05' not in file:
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
                filesList.append(file) # adds file name to list if it is
    for i, entry in enumerate(filesList):
        if 'aptMCH' in entry:
            new = entry.replace('aptMCH', '0fgml')
            filesList[i] = new
            os.rename(os.path.join(electrodeFolder, entry), os.path.join(electrodeFolder, new))
    filesList = natsort.natsorted(filesList)
    print(filesList)
    return filesList

def frequencies(filesList, electrodesFolder):
    gp = parser.GamryParser()
    freq_list = []
    for i, entry in enumerate(filesList):
        entryPath = os.path.join(electrodesFolder, entry)
        gp.load(filename=entryPath)
        header = gp.get_header()
        freq_list.append(header['FREQUENCY'])
    return freq_list

def plotFreq(electrodesFolders, dataType,concentrations, df):
    peaksMatrix = []
    i =0
    while i < len(df.index)-1:
        for electrode in electrodesFolders:
            for concentration in concentrations:
                fileList = extractConcentration(electrode, dataType,concentration)
                freq_list = frequencies(fileList, electrode)
                print(i,electrode,concentration)
                method1 = df.iloc[i, 0]
                fraction1 = df.iloc[i, 1]
                method2 = df.iloc[i, 2]
                fraction2 = df.iloc[i, 3]
                print(method1,method2)
                peaksMatrix.append(peaks(fileList, electrode, method1, fraction1,method2,fraction2))
                print(peaksMatrix)
                i=i+1
    plt.figure()
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.plot(freq_list, peaksMatrix[0], '-ro', label = '1_15 w/ MCH' )
    plt.plot(freq_list, peaksMatrix[1], '--ro', label = '1_15 w/ 1 ng S1')
    plt.plot(freq_list, peaksMatrix[2], '-go', label='1_19 w/ MCH')
    plt.plot(freq_list, peaksMatrix[3], '--go', label = '1_19 w/ 1 ng S1')
    plt.plot(freq_list, peaksMatrix[4], '-bo', label='1_20 w/ MCH')
    plt.plot(freq_list, peaksMatrix[5], '--bo', label='1_20 w/ 1 ng S1')
    plt.plot(freq_list, peaksMatrix[6], '-mo', label='1_21 w/ MCH')
    plt.plot(freq_list, peaksMatrix[7], '--mo', label='1_21 w/ 1 ng S1')
    plt.title('SWV Frequency Map Wrinkled Electrodes', fontsize = '22')
    plt.xlabel('Frequencies (Hz)', fontsize = '18')
    plt.ylabel('Peak Current (nA)', fontsize = '18')
    plt.legend( fontsize = '14')
    return peaksMatrix

def plotFreqN(electrodeFolders, dataType, electrodeNames, methodL, methodR, referenceMeasurement, concentration):
    plt.figure()
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['xtick.labelsize'] = 30
    for entry in electrodeFolders:
        fileList = extractConcentration(entry, dataType, concentration)
        #fileList = extractDataType(entry, dataType)
        freq_list = frequencies(fileList, entry)
        peakHeights = peaks(fileList, entry, methodL, methodR)
        df_peaks = pd.DataFrame(columns=np.arange(np.size(peakHeights)))
        df_peaks.loc[0] = peakHeights
        norm = normalize(df_peaks, referenceMeasurement)
        plt.plot(freq_list, norm.loc[0], marker='o', markersize = 13, )
    plt.title('SWV Freq Map (10 fg/mL)', fontsize='30', fontweight = 'bold')
    plt.xlabel('Frequencies (Hz)', fontsize='30')
    plt.ylabel('Normalized Change in Peak Current (%)', fontsize='30')
    plt.legend(electrodeNames, fontsize='30')

def plotFrequencyNavg(averages, electrodesFolders, dataType):
    for entry in electrodesFolders:
        fileList = extractDataType(entry, dataType)
        freq_list = frequencies(fileList, entry)
    avgs = averages.iloc[0, :]
    std = averages.iloc[1, :]
    plt.figure()
    plt.plot(freq_list, avgs, marker='o')
    plt.title('Average Change in Signal After S1 addition (n=4)', fontsize='16')
    plt.xlabel('Freqeuncy[Hz]', fontsize='16')
    plt.ylabel('Normalized Change in Peak Current[%]', fontsize='16')

"""
peaks: calculates the peaks of all the DTA files in the electrode folder
inputs: filesList --> list of DTA files of interest
        electrodeFolder --> filepath of folder containing DTA files
        methodNum --> method for baseline calculation 
outputs: filesList --> returns a list of files of certain dataType 
"""
def peaks(filesList, electrodeFolder, method1, fraction1, method2, fraction2):
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
       smoothY = scipy.signal.savgol_filter(newY, 23, 3)
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

        # Method Calibration
       # Using two methods
       if method2:
           if n <= 0: # Low concentration = Method 1
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
       offset = 30
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

       # ---------------uncomment to show baselines--------------------------
       # plt.figure()
       # plt.title('%s Data' % filename)
       # plt.xlabel('Vfwd [V]')
       # plt.ylabel('Idif [A]')
       # plt.plot(newX, smoothY)
       # plt.plot(val1x, val1, marker='.')
       # plt.plot(val2x, val2, marker='.')
       # plt.plot(newX, baseline, linestyle='--')
       # plt.plot(newX[newPeakHeightInd], smoothY[newPeakHeightInd], '*')
       # plt.annotate(newPeakHeight, (newX[newPeakHeightInd], smoothY[newPeakHeightInd]))

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
        normalVal = abs(normalVal)  # changes all values to non-negative
        normal_pks.append((np.around(normalVal, 3)))  # Add normalized value to array
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
    return averages  # returns a dataframe

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
    print('\naverages:\n',averages, '\n')
    print('concentrations: ',concentrations, '\n')
    objects = list(averages.columns)  # gets column names
    dfy = list(averages.iloc[0, :])  # access row 1 of the data frame (the peak heights)
    error = list(averages.iloc[1, :])  # access row 2 of the data frame (the std deviation)
    #x_pos = np.arange(len(objects))  # finds number of bars
    plt.figure()  # creates figure
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24

    colormap = plt.cm.plasma(.75) # change color map here! Colormap set to plasma.
    plt.plot(concentrations, dfy, marker = 'o', markersize = '10', linestyle = '', color = colormap, markeredgecolor='k', markeredgewidth=1.0)  # plots graph
    # plt.errorbar(concentrations, dfy, yerr = error, ecolor = 'k', linestyle = '', capsize = 3) # creates error bars
    plt.fill_between(concentrations, np.array(dfy)-np.array(error), np.array(dfy)+np.array(error), color=colormap, alpha=.2)
    plt.xlabel('Concentration S1 (fg/mL)', fontsize = '18')
    plt.ylabel('Normalized change in signal (%)', fontsize = '18')  # gives y-label for bar graph
    #plt.ylabel('Change normalized to 1 fg/mL (%)', fontsize='30')  # gives y-label for bar graph
    plt.title('Change in Signal at %s (n=3) {for electrodes 13, 17, 18}' %frequency, fontsize='28', fontweight = 'bold')  # display title
    #plt.legend(conditions, fontsize = '16')
    # plt.xscale('symlog', linthresh=1e-1, subs=[2, 3, 4, 5, 6, 7, 8, 9])  # converts x scale to log
    # plt.yticks(np.arange(0,90,10))
    # plt.xlim(0.75, 15000)
    # plt.xlim(-0.01, 150)

    x = concentrations
    x[0] = 10**(-9)  # replaces the zero point
    y = dfy
    p = np.polyfit(np.log(x), y, 1)  # creates equation for line of degree 1 best fit
    polynomial = np.poly1d(p)
    log10_y_fit = polynomial(np.log(x))
    plt.plot(x, log10_y_fit, 'b-', label="linear fit")
    plt.xscale('log', subs=[2, 3, 4, 5, 6, 7, 8, 9])  # converts x scale to log
    plt.xlim(1e-5, 1e3)

def plotPeaks(rawPeaks, concentrations, electrodeNames, frequency):
    plt.figure()  # creates figure
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24
    i=0
    while i <len(rawPeaks.index):
        row = rawPeaks.iloc[i,:]
        plt.plot(concentrations, row, marker='o', linestyle='-')  # plots graph
        i+=1
    plt.xlabel('Concentration S1 (fg/mL)', fontsize = '24')
    plt.ylabel('Peak Height (nA)', fontsize = '24')  # gives y-label for bar graph
    plt.title('SWV %s Peak Height (%s)'% (electrodeNames, frequency), fontsize='28', fontweight = 'bold')  # display title
    plt.legend(electrodeNames, fontsize = '18')
    plt.xscale('symlog', linthresh=1e-1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlim(-0.01, 150)
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