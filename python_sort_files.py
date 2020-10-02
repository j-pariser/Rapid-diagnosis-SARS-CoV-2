# Code for sorting files into folders
# Instructions: 1. Load all files you want sorted into the same folder as file_sorter.py
#               2. Change the list folder_names to the names of the electrodes
#               3. Hit run!
#
# WARNING: FILES WILL NOT SORT IF FOLDER NAMES AREN'T IN CORRESPONDING FORMAT. Example: '1_56'

import os
import shutil

# list of electrodes --> will be used to create folder names
folder_names = ['1_55', '1_56', '1_57', '1_58', '1_62', '1_63']  # <----CHANGE FOLDER NAMES HERE--------------

# get current working directory
path = os.getcwd()

# get all the names of every file in directory
names = os.listdir(path)

# get parent directory of folder with all files
parent = os.path.dirname(path)

# makes folders for every name in folder_names
for i, files in enumerate(folder_names):
    # checks if folder name already exists
    if not os.path.exists(os.path.join(parent, folder_names[i])):
        # makes a new folder using names listed in folder_names
        os.makedirs(os.path.join(parent, folder_names[i]))

# moves all the files into their corresponding folder using keywords
for j in range(len(folder_names)):
    for file in names:
        # checks if an individual folder has a keyword
        if folder_names[j] in file:
            temp = os.path.join(parent, folder_names[j])
            if not os.path.exists(os.path.join(temp, file)):
                # moves files into corresponding folders
                shutil.move(os.path.join(path, file), os.path.join(parent, folder_names[j]))
