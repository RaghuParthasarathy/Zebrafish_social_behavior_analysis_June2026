# !/usr/bin/env python3  
# IO_toolkit.py
# -*- coding: utf-8 -*- 
"""
Author:   Raghuveer Parthasarathy
Created on Mon Aug 25 20:59:37 2025
Last modified February 24, 2026 -- Raghuveer Parthasarathy

Description
-----------

Author:   Raghuveer Parthasarathy, May-July 2023
Last modified by Raghuveer Parthasarathy, March 12, 2026

Module containing functions for handling data files, configuration
files, and output files -- reading and writing.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
import os
from pathlib import Path
import pickle
import pandas as pd
import yaml
import tkinter as tk
import tkinter.filedialog as filedialog
from time import perf_counter
import re
import tifffile
import imageio.v3 as iio
from scipy.stats import binned_statistic_2d

from toolkit import get_Nfish, \
    combine_all_values_constrained, get_effective_dims, \
    dilate_frames, get_values_subset, make_frames_dictionary
# Note: get_fish_angular_speeds is imported locally inside the function that
# uses it (to avoid a circular import with behavior_identification_single).

# Function to get a valid path from the user (base Path or config path)
def get_basePath():
    """
    Ask the user for the "base" Path that either contains all the 
    CSV trajectory files or that contains "subgroup" folders 
    with the subgroup CSV files.  Either text input or, if blank,
    provide a dialog box.
    Verify that the basePath contains:
        expt_config.yaml, analysis_parameters.yaml, CSVcolumns.yaml

    """
    print('\n\nSelect the "CSV files" folder for this experiment')
    while True:
        user_input = input("Type the full path name; leave empty for dialog box: ")
        if user_input.lower() == '':
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            selected_path = tk.filedialog.askdirectory(title="Select CSV files folder")
        else:
            selected_path = user_input
        if os.path.isdir(selected_path):
            # It's a valid path
            # Check that it contains the required config files
            if not (os.path.isfile(os.path.join(selected_path, "expt_config.yaml")) and
                    os.path.isfile(os.path.join(selected_path, "analysis_parameters.yaml")) and
                    os.path.isfile(os.path.join(selected_path, "CSVcolumns.yaml"))):
                print(f'\n\nFolder {selected_path} does not contain all three config files:')
                print('      expt_config.yaml, analysis_parameters.yaml, CSVcolumns.yaml')
                if not os.path.isfile(os.path.join(selected_path, "expt_config.yaml")):
                    print('expt_config.yaml is missing; maybe the filename has an "s" at the end?')
                print("Invalid path. Please try again.")
            else: 
                return selected_path
        else:
            print("Invalid path. Please try again.")


def get_loading_option():
    """
    Prompt the user to select a loading option (from CSVs or Pickle)
    
    Returns
    -------
    loading_option : str
        The selected loading option.
    """
    options = {
        1: "load_from_CSV",
        2: "load_from_pickle"
        # Add more options here as needed
    }
    
    print("\nLoading options:")
    print("   load_from_CSV -- Reads CSV files. Do this for the first analysis of an experiment")
    print("   load_from_pickle -- Loads trajectories from pre-existing pickle file; re-analyzes")
    print("\nSelect loading option:")
    for key, value in options.items():
        print(f"   {key}: {value} (default)" if key == 1 else f"   {key}: {value}")
    
    try:
        user_input = input("Enter the number corresponding to your choice: ")
        selected_option = int(user_input) if user_input else 1
    except ValueError:
        selected_option = 1
    
    loading_option = options.get(selected_option, "load_from_CSV")
    return loading_option


def get_valid_file(fileTypeString = 'Config File'):
    """
    Check if the file+path exists; if not, dialog box.

    Parameters
    ----------
    fileTypeString : String, to use in prompt text

    Returns
    -------
    selected_file : selected path+file

    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    while True:
        titleString = f'Select {fileTypeString}'
        selected_file = tk.filedialog.askopenfilename(title=titleString)
        if os.path.isfile(selected_file):
            return selected_file
        else:
            print(f"Invalid file. Please select a valid {fileTypeString}.")
             

def load_expt_config(config_path, config_file):
    """ 
    Loads the experimental configuration file
    Image scale and arena centers paths will be appended to config_path
        (same as basePath in main code).
    Inputs:
        config_path, config_file: path and file name of the yaml config file
    Outputs:
        expt_config : dictionary of configuration information
    """
    config_file_full = os.path.join(config_path, config_file)
    # Note that we have already checked that basePath contains all three 
    # config files, including expt_config.yaml
    # Check if the config file exists; dialog box if not
    
    with open(config_file_full, 'r') as f:
        all_config = yaml.safe_load(f)
    
    # There need only be one experimental configuration in the config file,
    # but allow for the possibility of more
    all_expt_names = list(all_config.keys())

    if len(all_expt_names)> 1:
        print('More...')
        print('\n\nALl experiments in this config file: ')
        for j, key in enumerate(all_expt_names):
            print(f'  {j}: {key}')
        expt_choice = input('Select experiment (name string or number): ')
        # Note that we're not checking if the choice is valid, i.e. if in 
        # all_expt_names (if a string) or if in 0...len(all_expt_names) (if 
        # a string that can be converted to an integer.)
        try:
            # Is the input string just an integer? Try integer...
            expt_config = all_config[all_expt_names[int(expt_choice)]]
        except:
            # Must be a string
            expt_config = all_config[all_expt_names[expt_choice]]
    else:
        expt_config = all_config[all_expt_names[0]]

    # Experiment Name; None if this isn't specified 
    if ("expt_name" not in expt_config.keys()):
        expt_config['expt_name'] = None
    
    # Image scale file name and path, if specified
    if ("imageScaleFilename" in expt_config.keys()) and \
        (expt_config['imageScaleFilename'] is not None):
        if ("imageScalePathName" in expt_config.keys()):
            if expt_config['imageScalePathName'] is not None:
                imageScalePathNameFull = os.path.join(config_path, 
                                                      expt_config['imageScalePathName'])
            else:
                imageScalePathNameFull = config_path
        expt_config['imageScaleLocation'] = os.path.join(imageScalePathNameFull, 
                                             expt_config['imageScaleFilename'])            
    else:
        expt_config['imageScaleLocation'] = None    

    if ("arenaCentersFilename" in expt_config.keys()):
        if expt_config['arenaCentersFilename'] != None:
            print('config_path: ', config_path)
            print('for arenaCentersPathName: ', 
                  expt_config['arenaCentersPathName'])
            print('for arenaCentersFilename: ', 
                  expt_config['arenaCentersFilename'])
            if expt_config['arenaCentersPathName'] is not None:
                expt_config['arenaCentersLocation'] = os.path.join(config_path,
                                                               expt_config['arenaCentersPathName'], 
                                                               expt_config['arenaCentersFilename'])
            else:
                expt_config['arenaCentersLocation'] = os.path.join(config_path,
                                                               expt_config['arenaCentersFilename'])
        else:
            expt_config['arenaCentersLocation'] = None
    else:
        expt_config['arenaCentersLocation'] = None
    
    return expt_config


def load_analysis_parameters(basePath, params_file):
    """ 
    Loads the analysis parameters file
    Inputs:
        basePath, params_file: path and file name of the yaml parameter file
    Outputs:
        params : dictionary of analysis parameters
    """
    params_file_full = os.path.join(basePath, params_file)
    # Note that we already checked if this exists
    with open(params_file_full, 'r') as f:
        all_param = yaml.safe_load(f)
    params = all_param['params']

    return params
        

def check_analysis_parameters(params):
    """
    Checks that all the keys in the analysis parameters file exist.
    If a key doesn't exist, prompts the user for its value with 
    defaults from the file.
    Performs various checks
    Inputs:
        params : dictionary of analysis parameters
    Outputs:
        params : dictionary of analysis parameters
    """
    
    # Define the required keys and their default values
    required_keys = {
        "edge_rejection_threshold_mm": None,  
        "edge_proximity_threshold_mm": 5,
        "cos_theta_AP_threshold": -0.7,
        "cos_theta_tangent_threshold": 0.34,
        "cos_theta_90_thresh": 0.26,
        "cosSeeingAngle": 0.5,
        "perp_windowsize": 2,
        "perp_maxDistance_mm": 12.0,
        "contact_distance_threshold_mm": 2.5,
        "contact_inferred_distance_threshold_mm": 3.5,
        "contact_inferred_window": 3,
        "tail_rub_ws": 2,
        "tailrub_maxTailDist_mm": 2.0,
        "tailrub_maxHeadDist_mm": 12.5,
        "cos_theta_antipar": -0.8,
        "bend_min_deg": 10,
        "bend_Jmax_deg": 50, 
        "bend_Cmin_deg": 100, 
        "angle_xcorr_windowsize": 25,
        "approach_speed_threshold_mm_second": 20,
        "approach_cos_angle_thresh": 0.5,
        "approach_min_frame_duration": [2,2],
        "motion_speed_threshold_mm_second": 9,
        "proximity_1_threshold_mm": [5.0, 15.0],
        "proximity_1_distance_measure": 'closest',
        "proximity_2_threshold_mm": [5.0, 15.0],
        "proximity_2_distance_measure": 'closest',
        "output_subFolder": 'Analysis',
        "allDatasets_ExcelFile": 'behavior_counts.xlsx', 
        "allDatasets_markFrames_ExcelFile":  'behaviors_in_each_frame.xlsx'
        # "another_key": default_value,
    }
    
    # if a proximity threshold is just a single number, make the range
    #    0.0 to that number (see Dec. 2025 notes)
    for prox_key in ("proximity_1_threshold_mm", "proximity_2_threshold_mm"):
        if prox_key in params:
            if type(params[prox_key]) == int or type(params[prox_key]) == float:
                params[prox_key] = [0.0, params[prox_key]]

    for prox_key in ("proximity_1_distance_measure", "proximity_2_distance_measure"):
        if prox_key not in params:
            print(f'Using closest distance for "{prox_key}"')
            params[prox_key] = 'closest'
        if (params[prox_key] != 'closest') and (params[prox_key] != 'head_to_head'):
            print(f'INVALID OPTION for {prox_key}. Using closest distance.')
            params[prox_key] = 'closest'
        
    # If 'allDatasets_markFrames_ExcelFile' is empty or None, set to None
    if "allDatasets_markFrames_ExcelFile" not in params:
        print('\n\nSetting allDatasets_markFrames_ExcelFile to None.\n')
        params['allDatasets_markFrames_ExcelFile'] = None
    if params['allDatasets_markFrames_ExcelFile'].lower() == 'none':
        params['allDatasets_markFrames_ExcelFile'] = None
    
    # Check for missing keys and prompt the user for their values
    for key, default_value in required_keys.items():
        if key not in params:
            user_input = input(f"Enter value for {key} (default: {default_value}): ")
            params[key] = user_input if user_input else default_value
    
    # Set edge rejection criterion to None if 'None', and if negative
    if isinstance(params["edge_rejection_threshold_mm"], str):
        if params["edge_rejection_threshold_mm"].lower() == 'none':
            params["edge_rejection_threshold_mm"] = None
        else:
            raise ValueError("edge_rejection_threshold_mm : only allowed string is None") 
    if isinstance(params["edge_rejection_threshold_mm"], float):
        if params["edge_rejection_threshold_mm"] < 0.0:
            params["edge_rejection_threshold_mm"] = None
    
    return params


def load_global_expt_config(config_path, config_file):
    """ 
    Loads the global experimental configuration file, which points to the
    experiment-specific configuration files
    
    PLACEHOLDER
    To be written.
    Modify all_expt_configs to just contain basePath for the various experiments
    Note that this uses get_valid_file(); it's the only function that does;
       will need to load this in behaviors_main.py

    Inputs:
        config_path, config_file: path and file name of the yaml config file
    Outputs:
        expt_config : dictionary of configuration information
    """
    config_file_full = os.path.join(config_path, config_file)
    # Check if the config file exists; dialog box if not
    if not os.path.isfile(config_file_full):
        print(f"The config file '{config_file_full}' does not exist.")
        config_file_full = get_valid_file(fileTypeString = 'Config File')
    
    with open(config_file_full, 'r') as f:
        all_config = yaml.safe_load(f)
    all_expt_names = list(all_config.keys())
    print('\n\nAll experiments: ')
    for j, key in enumerate(all_expt_names):
        print(f'  {j}: {key}')

    #----
    # MODIFY HERE: As below, select an experiment, but just use this path info, 
    # and load the expt config file with load_expt_config
        
    expt_choice = input('Select experiment (name string or number): ')
    # Note that we're not checking if the choice is valid, i.e. if in 
    # all_expt_names (if a string) or if in 0...len(all_expt_names) (if 
    # a string that can be converted to an integer.)
    try:
        # Is the input string just an integer? Try integer...
        expt_config = all_config[all_expt_names[int(expt_choice)]]
    except:
        # Must be a string
        expt_config = all_config[all_expt_names[expt_choice]]
    
    return expt_config           

def get_CSV_filenames(basePath, expt_config, startString="results"):
    """
    Select subgroup (if applicable) and get a list of all CSV files 
    whose names start with startString, probably "results," 
    in the basePath previously specified

    Inputs:
        basePath : folder containing folders with CSV files for analysis;
                    dataPathMain will be appended to this.
                    Required, even if dataPathFull overwrites it
        expt_config : dictionary containing subGroup info (if it exists)
        startString : the string that all CSV files to be considered should
                        start with. Default "results"
    Returns:
        A tuple containing
        - dataPath : the folder path containing CSV files
        - allCSVfileNames : a list of all CSV files with names 
                            starting with startString (probably "results")
        - subGroupName : Path name of the subGroup; None if no subgroups
    
    """

    if ('subGroups' in expt_config.keys()) and (expt_config['subGroups'] != None):
        print('\nSub-Experiments:')
        for j, subGroup in enumerate(expt_config['subGroups']):
            print(f'  {j}: {subGroup}')
        subGroup_choice = input('Select sub-experiment (string or number): ')
        try:
            subGroupName = expt_config['subGroups'][int(subGroup_choice)]
        except:
            subGroupName = expt_config['subGroups'][subGroup_choice]
        dataPath = os.path.join(basePath, subGroupName)
    else:
        subGroupName = None
        dataPath = basePath
        
    # Validate the folder path
    while not os.path.isdir(dataPath):
        print("Invalid data path. Please try again (manual entry).")
        dataPath = input("Enter the folder path: ")

    print("Selected folder path: ", dataPath)
    
    # Make a list of all relevant CSV files in the folder
    allCSVfileNames = []
    for filename in os.listdir(dataPath):
        if (filename.endswith('.csv') and filename.startswith(startString)):
            allCSVfileNames.append(filename)

    return dataPath, allCSVfileNames, subGroupName


def get_dataset_name(CSVfileName):
    """ 
    Extract the "dataset name" from the CSV filename. Delete
    "results_SocPref_", "_ALL.csv"; keep everything else
    E.g. file name "results_SocPref_3c_2wpf_k2_ALL.csv" gives
      dataset_name = "3c_2wpf_k2""
    
    Returns:
        dataset_name : string
    """
    dataset_name = CSVfileName.replace("results_SocPref_", '')
    dataset_name = dataset_name.replace("_ALL", '')
    dataset_name = dataset_name.replace(".csv", '')
    return dataset_name


def set_outputFile_params(params, expt_config, subGroupName):
    """
    Fill in keys in params corresponding to output folders, Excel file names

    Parameters
    ----------
    params : dictionary of analysis parameters.
    expt_config : dictionary of experiment parameters.
    subGroupName : subgroup name (String) or None if no subgroup

    Returns
    -------
    params : dictionary of analysis parameters.

    """
    # Append experiment and subGroup names to Analysis output folder names
    if subGroupName is None:
        params['output_subFolder'] =  expt_config['expt_name'] + '_' + \
            params['output_subFolder']
    else:
        params['output_subFolder'] =  expt_config['expt_name'] + '_' + \
            subGroupName + '_' + params['output_subFolder']
    
    # Add subgroup name (if it exists) and the experiment name (i.e.
    # folder name) to the output Excel file names, appending these to
    # (1) "behavior_counts.xlsx" (or whatever params["allDatasets_ExcelFile"]
    #     is) for summary statistics of each behavior for each dataset
    # (2) "behaviors_in_each_frame.xlsx" (or whatever params["allDatasets_markFrames_ExcelFile"])
    #     is for marking behaviors in each frame
    base_name1, extension1 = os.path.splitext(params["allDatasets_ExcelFile"])
    if subGroupName is None:
        params["allDatasets_ExcelFile"] = f"{expt_config['expt_name']}_{base_name1}{extension1}"
    else:
        params["allDatasets_ExcelFile"] = f"{expt_config['expt_name']}_{subGroupName}_{base_name1}{extension1}" 
    print(f"Modifying output allDatasets_ExcelFile file name to be: {params['allDatasets_ExcelFile']}")
    if params["allDatasets_markFrames_ExcelFile"] is not None:
        base_name, extension = os.path.splitext(params["allDatasets_markFrames_ExcelFile"])
        if subGroupName is None:
            params["allDatasets_markFrames_ExcelFile"] = f"{expt_config['expt_name']}_{base_name}{extension}"
        else:
            params["allDatasets_markFrames_ExcelFile"] = f"{expt_config['expt_name']}_{subGroupName}_{base_name}{extension}"
        print(f"Modifying output allDatasets_markFrames_ExcelFile file name to be: {params['allDatasets_markFrames_ExcelFile']}")
    
    
    # If there are subgroups, modify the output Excel file name for
    # summary statistics of each behavior for each dataset -- instead
    # of "behavior_counts.xlsx" (or whatever params["allDatasets_ExcelFile"]
    # currently is), append subGroupName
    if not subGroupName==None:
        # base_name, extension = os.path.splitext(params["allDatasets_ExcelFile"])
        params["allDatasets_ExcelFile"] = f"{base_name1}_{subGroupName}{extension1}"
        print(f"Modifying output allDatasets_ExcelFile file name to be: {params['allDatasets_ExcelFile']}")

    return params


def load_all_position_data(allCSVfileNames, expt_config, CSVcolumns,
                           dataPath, params):
    """
    For all CSV files in the list, call load_data() to load all position
    data, and determine general parameters such as fps and scale
    Inputs:
        allCSVfileNames : a list of all CSV file names to consider
        expt_config : experiment configuration dictionary
        CSVcolumns : CSV column dictionary (what's in what column)
        dataPath : Path containing data files
        params : dictionary of analysis parameters 
                (used only for plotAllPositions())
        
    Returns:
        all_position_data : list of numpy arrays with position information for
                    each dataset. Nframes x Ncolumns x Nfish
        datasets : list of dictionaries, one for each dataset. 
                    datasets[j] contains information for dataset j.
    
    """
    # Number of datasets
    N_datasets = len(allCSVfileNames)

    # initialize a list of numpy arrays and a list of dictionaries
    all_position_data = [{} for j in range(N_datasets)]
    datasets = [{} for j in range(N_datasets)]
    os.chdir(dataPath)

    # For each dataset, get general properties and load all position data
    for j, CSVfileName in enumerate(allCSVfileNames):
        datasets[j]["CSVfilename"] = CSVfileName
        datasets[j]["dataset_name"] = get_dataset_name(CSVfileName)
        datasets[j]["image_scale"] = float(get_imageScale(datasets[j]["dataset_name"], 
                                                    expt_config))
        datasets[j]["fps"] = expt_config["fps"]
        
        # Get arena center, subtracting image position offset
        datasets[j]["arena_center"] = get_ArenaCenter(datasets[j]["dataset_name"], 
                                                      expt_config)
        # Estimate center location of Arena
        # datasets[j]["arena_center"] = estimate_arena_center(all_position_data[j],
        #                                                    CSVcolumns["head_column_x"],
        #                                                    CSVcolumns["head_column_y"])

        # Load all the position information as a numpy array
        print('Loading dataset: ', datasets[j]["dataset_name"])
        all_position_data[j], datasets[j]["frameArray"] = \
            load_data(CSVfileName, CSVcolumns["N_columns"]) 
        datasets[j]["Nframes"] = len(datasets[j]["frameArray"])
        datasets[j]["Nfish"] = all_position_data[j].shape[2]
        print('   ', 'Number of frames: ', datasets[j]["Nframes"] )
        datasets[j]["total_time_seconds"] = (np.max(datasets[j]["frameArray"]) - \
            np.min(datasets[j]["frameArray"]) + 1.0) / datasets[j]["fps"]
        print('   ', 'Total duration: ', datasets[j]["total_time_seconds"], 'seconds')
    

    return all_position_data, datasets
    
def load_data(CSVfileName, N_columns):
    """
    Loads position data from a CSV file and returns a single array
    containing position information for all fish
    (position, angle, body markers etc.)
    Works for any number of fish -- infers this from the first column
    Also returns frame numbers (first column of CSV), checking that 
    the frame number array is the same for each fish id of the 
    dataset.
    Checks that frame numbers are consecutive integers from 1 to Nframes 
    for each ID; raises an Error otherwise.

    Args:
        CSVfileName (str): CSV file name with tracking data
        N_columns: number of columns (probably 26).

    Returns:
        all_data : a single numpy array with all the data 
                   (all columns of CSV)
                   Rows = frames
                   Col = CSV columns
                   Layers = fish (Nfish)
        frameArray : array of all frame numbers
    """
    data = np.genfromtxt(CSVfileName, delimiter=',')
    id_numbers = data[:, 0]
    frame_numbers = data[:, 1]
    unique_ids = np.unique(id_numbers)
    Nfish = len(unique_ids)
    # print('Number of fish: ', Nfish)
    
    # (1) Check that the set of frame numbers is the same for all ID numbers
    frame_sets = [set(frame_numbers[id_numbers == id]) for id in unique_ids]
    if not all(frame_set == frame_sets[0] for frame_set in frame_sets):
        raise ValueError("Frame numbers are not consistent across all ID numbers")

    # (2) Get the number of unique frame numbers, and an array of all frame numbers
    frameArray = np.sort(np.array(list(frame_sets[0]), dtype=int))
    Nframes = len(frame_sets[0])

    # (3) Check that frame numbers are consecutive integers from 1 to Nframes for each ID
    for id in unique_ids:
        id_frames = np.sort(frame_numbers[id_numbers == id])
        if not np.array_equal(id_frames, np.arange(1, Nframes + 1)):
            raise ValueError(f"Frame numbers for ID {id} are not consecutive from 1 to {Nframes}")

    # (4) Create the all_data array
    all_data = np.zeros((Nframes, data.shape[1], Nfish))
    for j, id in enumerate(unique_ids):
        id_data = data[id_numbers == id]
        # sort by frame number, though these should be in order.
        sorted_indices = np.argsort(id_data[:, 1])
        all_data[:, :, j] = id_data[sorted_indices]

    return all_data, frameArray


def get_ArenaCenter(dataset_name, expt_config):
    """ 
    Extract the x,y positions of the Arena centers from the 
    arenaCentersFilename CSV -- previously tabulated.
    Image offsets also previously tabulated, "arenaCentersColumns" 
         columns of offsetPositionsFilename
    Uses first column in the arenaCenters file to match to dataset name

    Inputs:
        dataset_name :
        expt_config : configuration dictionary. Contains:
            arenaCentersFilename: csv file name with arena centers 
                (and one header row). If None, estimate centers from well
                offsets
            arenaCentersColumns : columns containing image offsets
            offsetPositionsFilename : csv file name with well offset positions
            datasetRemoveStrings: strings to remove from file and dataset names
    
    Returns:
        arenaCenterCorrected: tuple of x, y positions of arena Center
    Returns none if no rows match the input dataset_name, 
        and error if >1 match
        
    """
    datasetColumns = [0] # First column only for matching dataset names.
    
    # Find the row of this dataset in the well offset data file
    matching_offset_rows = []
    with open(expt_config['offsetPositionsFilename'], 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if match_dataset_name(dataset_name, row, 
                                  removeStrings = expt_config['datasetRemoveStrings'],
                                  datasetColumns = datasetColumns)==True:
                matching_offset_rows.append(row)
                
    if len(matching_offset_rows) == 0:
        # No matching rows in the offset file were found.
        arenaOffset = None
        print(f"\nFilename: {expt_config['offsetPositionsFilename']}")
        raise ValueError(f"get_ArenaCenter: No rows contain the input dataset_name: {dataset_name}")
    elif len(matching_offset_rows) > 1:
        raise ValueError("get_ArenaCenter: Multiple rows contain the input dataset_name string")
    else:
        arenaOffset = np.array((matching_offset_rows[0][1], 
                                matching_offset_rows[0][2])).astype(float)
        # There's offset data, now load or estimate arena center
        if expt_config['arenaCentersLocation'] != None:
            # Find the uncorrected arena positions
            matching_rows = []
            with open(expt_config['arenaCentersLocation'], 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row
                for row in reader:
                    if match_dataset_name(dataset_name, row, 
                                          removeStrings = expt_config['datasetRemoveStrings'],
                                          datasetColumns = datasetColumns)==True:
                        matching_rows.append(row)
        
                if len(matching_rows) == 0:
                    arenaCenterUncorrected = None
                elif len(matching_rows) > 1:
                    arenaCenterUncorrected = None
                    raise ValueError("get_ArenaCenter: Multiple rows contain the input dataset_name string")
                else:
                    arenaCenterUncorrected = np.array((matching_rows[0][expt_config['arenaCentersColumns'][0]], 
                                                       matching_rows[0][expt_config['arenaCentersColumns'][1]])).astype(float)
                    arenaCenterCorrected = arenaCenterUncorrected - arenaOffset
        else:
             # Estimate arena positions based on well offset positions
             arenaCenterCorrected = np.array((matching_offset_rows[0][3], 
                                    matching_offset_rows[0][4]), 
                                             dtype=float)/2.0

    if arenaOffset is not None:
        return arenaCenterCorrected
    else:
        return None
        

def get_imageScale(dataset_name, expt_config):
    """ 
    Extract the image scale (um/px) from the config file (same for all datasets
    in this experiment) or from the imageScaleFilename CSV 

    Inputs:
        dataset_name : name of dataset
        expt_config : dictionary that contains as keys
            imageScale : image scale, um/px
            OR
            imageScaleLocation : Path and filename of CSV file containing 
                             image scale information
            imageScaleColumn : column (0-index) with image scale
            datasetColumns: columns to concatenate to match dataset name .
            datasetRemoveStrings: strings to remove from file and dataset names

    Returns:
        image scale (um/px)
    Returns none if no rows match the input dataset_name, 
        and error if >1 match
        
    Code partially from GPT3-5 (openAI)
    """
    
    if ("imageScale" in expt_config.keys()) and \
        expt_config['imageScale'] != None:
            return expt_config['imageScale']
    else:
        # Read from CSV file
        matching_rows = []
        with open(expt_config['imageScaleLocation'], 'r', 
                  encoding='Latin1') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                if match_dataset_name(dataset_name, row, 
                                      removeStrings = expt_config['datasetRemoveStrings'],
                                      datasetColumns = expt_config['datasetColumns']) \
                        == True:
                    matching_rows.append(row)
    
        if len(matching_rows) == 0:
            print('Dataset name: ', dataset_name)
            raise ValueError(f"get_imageScale: No row found with the input string {dataset_name}")
        elif len(matching_rows) > 1:
            # print(dataset_name, ' in rows: ', matching_rows[:][0])
            raise ValueError("get_imageScale: Multiple rows contain the input dataset_name string")
        else:
            return matching_rows[0][expt_config['imageScaleColumn']]
        

def match_dataset_name(dataset_name, row_array, 
                       removeStrings = [], datasetColumns = [0]):
    """ 
    Test whether the string formed from the datasetColumns columns of 
         row_array[] (appending '_') between columns) matches the string 
         dataset_name, modifying both strings by deleting
         the strings in the list removeStrings).
    
    Return True if dataset_name ad row_array match (given modifications);
    False otherwise
    
    To be used by get_imageScale() and get_ArenaCenter()
    
    """
    
    # Modifications to dataset_name
    mod_dataset_name = dataset_name
    for remove_string in removeStrings:
        mod_dataset_name = mod_dataset_name.replace(remove_string, '')
        
    # Append another column to row[0], if desired, with underscore
    mod_row_array = row_array[datasetColumns[0]]
    for j in range(1, len(datasetColumns)):
        mod_row_array = mod_row_array + '_' + row_array[datasetColumns[j]]

    #print('mod_row_array, initial: ', mod_row_array)
    # Modifications to row_array
    for remove_string in removeStrings:
        mod_row_array = mod_row_array.replace(remove_string, '')
        #print('  - mod_row_array: ', mod_row_array)
    
    #print('mod_dataset_name: ', mod_dataset_name)
    #print('mod_row_array: ', mod_row_array)
    if mod_dataset_name == mod_row_array:
        return True
    else:
        # print('dataset name: ', mod_dataset_name, ' row array: ', mod_row_array)
        return False
    
        
def get_output_pickleFileNames(expt_name, subGroupName = None):
    """
    Get / construct pickle file name. 
    First construct the "base" filename, then filenames for
    both Pickle file outputs, appending "positionData" and "datasets"
    
    Parameters
    ----------
    expt_name : experiment name, probably expt_config['expt_name']
    subGroupName : subGroupName, None if no subgroups

    Returns
    -------
    pickleFileNames : tuple of three strings, name of pickle files for
                     "positionData" and "datasets" pickle files, and "base" name

    """                              
    print('\nEnter the base filename for the output pickle files;')
    print('   Do not include ".pickle"; appended later.')
    print('   Enter "none" to skip pickle output.')
    # Append experiment and subGroup names to Analysis output folder
    if subGroupName is None:
        defaultPickleFileNameBase =  expt_name 
    else:
        defaultPickleFileNameBase =  expt_name + '_' + subGroupName
    pickleFileNameBase = input(f'   Press Enter for default {defaultPickleFileNameBase}: ')
    if pickleFileNameBase == '':
        pickleFileNameBase = defaultPickleFileNameBase
    
    # Make sure doesn't end with .pickle
    if pickleFileNameBase.endswith('.pickle'):
            pickleFileNameBase = pickleFileNameBase[:-7]
    
    pickleFileNames = (pickleFileNameBase + "_positionData.pickle", 
                       pickleFileNameBase + "_datasets.pickle",
                       pickleFileNameBase)
    
    return pickleFileNames
    

def write_pickle_file(dict_for_pickle, dataPath, outputFolderName, pickleFileName):
    """
    Write Pickle file containing a dictionary of variables in the analysis folder
    
    Parameters
    ----------
    dict_for_pickle : dictionary of variables to save in the Pickle file
    dataPath : CSV data path
    outputFolderName : output path, should be params['output_subFolder'],
                       appended to dataPath
    pickleFileName : string, filename, including .pickle
    Returns
    -------
    None.
    """
    # Use pathlib for better path handling
    pickle_folder = Path(dataPath) / outputFolderName
    pickle_file_path = pickle_folder / pickleFileName
    
    # Check path length and shorten if necessary
    if len(str(pickle_file_path)) > 250:  # Conservative limit for Windows
        print(f"Warning: Path too long ({len(str(pickle_file_path))} chars). Shortening...")
        
        # Try shortening the filename first
        name_part, ext_part = pickleFileName.rsplit('.', 1) if '.' in pickleFileName else (pickleFileName, '')
        max_filename_len = 250 - len(str(pickle_folder)) - 1  # -1 for path separator
        
        if max_filename_len > 20:  # Need reasonable minimum
            shortened_name = name_part[:max_filename_len-len(ext_part)-1] + '.' + ext_part if ext_part else name_part[:max_filename_len]
            pickle_file_path = pickle_folder / shortened_name
            print(f"Shortened filename to: {shortened_name}")
        else:
            # If even shortening filename doesn't work, use a generic name
            shortened_name = f"datasets_{hash(pickleFileName) % 10000}.pickle"
            pickle_file_path = pickle_folder / shortened_name
            print(f"Using generic filename: {shortened_name}")
    
    try:
        # Create output directory with exist_ok=True to handle race conditions
        pickle_folder.mkdir(parents=True, exist_ok=True)
        
        # Double-check the directory was created
        if not pickle_folder.exists():
            raise FileNotFoundError(f"Failed to create directory: {pickle_folder}")
        
        # Save the pickle file
        print(f'\nWriting pickle file: {pickle_file_path.name} in {pickle_folder}\n')
        with open(pickle_file_path, 'wb') as handle:
            pickle.dump(dict_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle file saved successfully.")
        
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        print(f"Failed path: {pickle_file_path}")
        print(f"Path length: {len(str(pickle_file_path))} characters")
        print("Possible causes:")
        print("- Path too long for Windows (>260 chars)")
        print("- Parent directory doesn't exist or is inaccessible")
        print("- Invalid characters in path")
        
        # Try one more fallback with a very short name
        fallback_name = f"data_{hash(str(pickle_file_path)) % 1000}.pkl"
        fallback_path = pickle_folder / fallback_name
        print(f"Attempting fallback with short filename: {fallback_name}")
        
        try:
            with open(fallback_path, 'wb') as handle:
                pickle.dump(dict_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Fallback successful! File saved as: {fallback_path}")
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise e  # Raise original error
            
    except PermissionError as e:
        print(f"PermissionError: {e}")
        print(f"Check write permissions for: {pickle_folder}")
        raise
    except OSError as e:
        print(f"OSError: {e}")
        print(f"Path length: {len(str(pickle_file_path))} characters")
        if len(str(pickle_file_path)) > 250:
            print("This is likely due to Windows path length limitations.")
            print("Consider using shorter folder/file names or enabling long path support.")
        raise


def load_and_assign_from_pickle(pickleFileName1 = None, pickleFileName2 = None):
    """
    Calls load_dict_from_pickle() and assign_variables_from_dict()
    to load *two* pickle files and assign variables.
    Asks users for filenames.
    
    Inputs:
        pickleFileName1, pickleFileName2 : Full pickle file paths + names,
            see code. Leave as None for user input or dialog box
        
    Outputs
        all_position_data : all position data, from first pickle file
        variable_tuple : tuple of variables, from the second pickle file
    
    """
    print('\n\nLoading from Pickle.')
    print('\n   Note that this requires *two* pickle files:')
    print('     (1) position data, probably in the CSV folder')
    print('     (2) "datasets" and other information, probably in Analysis folder')
    
    if (pickleFileName1 is None) or (pickleFileName2 is None):
        print('For each, enter the full path or just the filename; leave empty for a dialog box.')
        print('\n')

    if pickleFileName1 is None:
        pickleFileName1 = input('(1) Pickle file name for position data; blank for dialog box: ')
        if pickleFileName1 == '': pickleFileName1 = None
    pos_dict = load_dict_from_pickle(pickleFileName=pickleFileName1)
    print("Loaded pickle file 1")
    all_position_data = assign_variables_from_dict(pos_dict, inputSet = 'positions')

    if pickleFileName2 is None:
        pickleFileName2 = input('(2) Pickle file name for datasets etc.; blank for dialog box: ')
        if pickleFileName2 == '': pickleFileName2 = None
    data_dict = load_dict_from_pickle(pickleFileName=pickleFileName2)
    print("Loaded pickle file 2")
    variable_tuple = assign_variables_from_dict(data_dict, inputSet = 'datasets')
    
    return all_position_data, variable_tuple
    
def load_dict_from_pickle(pickleFileName=None):
    """
    Load contents from pickle file
    Assumes pickle file contains a dictionary of variables;
    returns these variables as a dictionary.
    
    Parameters
    ----------
    pickleFileName : pickle file name; can include path to append to basePath

    Returns
    -------
    dict_of_variables : dictionary containing all variables in the Pickle file
    """

    badFile = True  # for verifying
    while badFile:
        if pickleFileName is None or pickleFileName == '':
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            titleString = 'Select pickle file'
            pickleFileName = filedialog.askopenfilename(title=titleString,
                                                        filetypes=[("pickle files", "*.pickle")])
        else:
            if not os.path.isabs(pickleFileName):
                # Get parent folder information
                # Get path from dialog
                root = tk.Tk()
                root.withdraw()  # Hide the root window
                basePath = filedialog.askdirectory(title=f"Select folder containing specified pickle file, {pickleFileName}")
                pickleFileName = os.path.join(basePath, pickleFileName)
        if os.path.isfile(pickleFileName):
            badFile = False
        else:
            print(f"\n\nInvalid pickle file path or name: {pickleFileName}")
            print("Please try again; will force dialog box.")
            pickleFileName = None

    print(f'\nOpening pickle file: {pickleFileName}')
    print('\n')
    with open(pickleFileName, 'rb') as handle:
        dict_of_variables = pickle.load(handle)

    return dict_of_variables

def assign_variables_from_dict(dict_of_variables, inputSet):
    """
    Assign dictionary elements loaded from pickle file, from load_dict_from_pickle(),
    to variables. Hard-coded variables for each "inputSet" (positions or datasets etc.)
    Input
        dict_of_variables : dictionary containing all variables in the Pickle file
        inputSet : 'positions' or 'datasets' for the different pickle files
    Outputs
        variable_tuple : tuple of variables
    """
    
    if inputSet == 'positions':
        variable_tuple = dict_of_variables['all_position_data']
    elif inputSet == 'datasets': 
        datasets = dict_of_variables['datasets']
        CSVcolumns = dict_of_variables['CSVcolumns']
        expt_config = dict_of_variables['expt_config']
        params = dict_of_variables['params']
        basePath = dict_of_variables['basePath']
        dataPath = dict_of_variables['dataPath']
        subGroupName = dict_of_variables['subGroupName']
        
        # Number of datasets
        N_datasets = len(datasets)
        Nfish = get_Nfish(datasets)
   
        variable_tuple = (datasets, CSVcolumns, expt_config, params, N_datasets,
                          Nfish, basePath, dataPath, subGroupName)
    else :
        raise ValueError('Invalid inputSet for assigning variables')

    return variable_tuple


def combine_expts_from_pickle():
    """
    Load data from multiple experiments, each with two pickle files,
    combine the variables, and save the output to two new pickle files.
    
    Asks user for number of experiments to combine and loads data from
    each experiment's pickle files. Combines position data and datasets,
    verifies consistency of parameters, and creates new combined output.
    
    Also can write new Excel, CSV, YAML files based on the composite 
    dataset – optional, asks user.
    
    Returns
    -------
    None
    """
    # Get number of experiments to combine
    while True:
        try:
            N_expts = int(input('\nHow many experiments would you like to combine? '))
            if N_expts > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Initialize lists to store data from all experiments
    all_position_data_combined = []
    datasets_combined = []
    all_Nfish = []
    all_expt_configs = []
    all_params = []
    all_subGroupNames = []
    
    # Load data from each experiment
    for i in range(N_expts):
        print(f'\nLoading data for experiment {i+1} of {N_expts}')
        pos_data, var_tuple = load_and_assign_from_pickle()
        
        # Unpack variables from tuple
        (datasets, CSVcolumns, expt_config, params, N_datasets, 
         Nfish, basePath, dataPath, subGroupName) = var_tuple
        
        # Verify datasets length matches N_datasets
        if len(datasets) != N_datasets:
            raise ValueError(f'Experiment {i+1}: Length of datasets ({len(datasets)}) ' 
                           f'does not match N_datasets ({N_datasets})')
        
        # Store data for later combining/comparison
        all_position_data_combined.extend(pos_data)
        datasets_combined.extend(datasets)
        all_Nfish.append(Nfish)
        all_expt_configs.append(expt_config)
        all_params.append(params)
        all_subGroupNames.append(subGroupName)
    
    # Verify Nfish is the same for all experiments
    if len(set(all_Nfish)) != 1:
        raise ValueError(f'Nfish varies across experiments: {all_Nfish}')
    Nfish = all_Nfish[0]
    
    # Combine expt_config dictionaries
    combined_expt_config = {}
    for key in all_expt_configs[0].keys():
        if key in ['imageScaleLocation', 'arenaCentersLocation']:
            combined_expt_config[key] = None
        else:
            values = [config[key] for config in all_expt_configs]
            if len(set(str(v) for v in values)) > 1:  # Convert to string for comparison
                print(f'\nDifferent values found for expt_config["{key}"]:')
                for i, v in enumerate(values):
                    print(f'  Experiment {i+1}: {v}')
                value = input('Enter value to use for combined dataset: ')
                # Try to convert input to original type
                orig_type = type(values[0])
                try:
                    combined_expt_config[key] = orig_type(value)
                except ValueError:
                    combined_expt_config[key] = value
            else:
                combined_expt_config[key] = values[0]
    
    # Combine params dictionaries
    combined_params = {}
    for key in all_params[0].keys():
        if key == 'output_subFolder':
            continue  # Skip this key as it will be set later
        values = [p[key] for p in all_params]
        if len(set(str(v) for v in values)) > 1:  # Convert to string for comparison
            print(f'\nDifferent values found for params["{key}"]:')
            for i, v in enumerate(values):
                print(f'  Experiment {i+1}: {v}')
            value = input('Enter value to use for combined dataset: ')
            # Try to convert input to original type
            orig_type = type(values[0])
            try:
                combined_params[key] = orig_type(value)
            except ValueError:
                combined_params[key] = value
        else:
            combined_params[key] = values[0]
    
    # Handle subGroupName
    if len(set(all_subGroupNames)) > 1:
        combined_subGroupName = all_subGroupNames
    else:
        combined_subGroupName = all_subGroupNames[0]
    
    # Get base filename for output
    basePickleFileName = input('\nEnter base name for output pickle files; will add .pickle etc.: ')
    outputPickleFileNames = [
        f'{basePickleFileName}_positionData.pickle',
        f'{basePickleFileName}_datasets.pickle'
    ]
    
    # Get output path with option for text input or dialog
    print('\nEnter output path or leave blank for dialog box (create folder if necessary):')
    outputPath = input('Path: ').strip()
    if not outputPath:
        root = tk.Tk()
        root.withdraw()
        outputPath = filedialog.askdirectory(title="Select output directory; create if necessary")
        if not outputPath:  # If user cancels dialog
            raise ValueError("No output path selected")
    
    # Get subfolder name with default "Analysis"
    subfolderName = input('\nEnter name for analysis subfolder; default "Analysis": ').strip()
    if not subfolderName:
        subfolderName = "Analysis"
    
    # Create full output paths
    outputSubPath = os.path.join(outputPath, subfolderName)
    
    # Create subfolder if it doesn't exist
    if not os.path.exists(outputSubPath):
        os.makedirs(outputSubPath)
    
    # Create and write output pickle files
    variables_dict = {'all_position_data': all_position_data_combined}
    write_pickle_file(variables_dict, dataPath=outputPath,
                     outputFolderName='',
                     pickleFileName=outputPickleFileNames[0])
    
    # Update combined_params with output_subFolder
    combined_params['output_subFolder'] = outputSubPath
    
    variables_dict = {
        'datasets': datasets_combined,
        'CSVcolumns': CSVcolumns,
        'expt_config': combined_expt_config,
        'params': combined_params,
        'basePath': outputPath,  # Use outputPath as the new basePath
        'dataPath': None,
        'subGroupName': combined_subGroupName
    }
    write_pickle_file(variables_dict, dataPath=outputSubPath,
                     outputFolderName='',
                     pickleFileName=outputPickleFileNames[1])
    
    print('\nPickle files successfully combined and written to:')
    print(f'  {os.path.join(outputPath, outputPickleFileNames[0])}')
    print(f'  {os.path.join(outputSubPath, outputPickleFileNames[1])}')
    
    outputExcel = input('\nOutput Excel, CSV, and YAML files? y=yes, n=no: ')
    if outputExcel.lower()=='y':
        write_CSV_Excel_YAML(expt_config = combined_expt_config,
                             params = combined_params, 
                             dataPath = outputPath, 
                             datasets = datasets_combined)


def write_output_files(params, output_path, datasets):
    """
    Write the output files (several) for all datasets
    Inputs:
        params : analysis parameters; we use the output file pathinfo
        output_path : output path, probably os.path.join(dataPath, params['output_subFolder']
        datasets : list of dictionaries: all dataset and analysis output
        
    Outputs:
        summary_excel_path
        (and multiple file outputs)
    """
    
    t_writing_all_start = perf_counter()
    
    print('\n\nWriting output files...')
    N_datasets = len(datasets)
    
    # Convert to Path object for better handling
    output_path = Path(output_path)
    
    # Create output directory, if it doesn't exist
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        print(f"Path: {output_path}")
        print(f"Path length: {len(str(output_path))} characters")
        raise

    # Store original directory to restore later (instead of os.chdir)
    original_cwd = Path.cwd()

    try:
        # behaviors (events) to write. (Superset)
        Nfish = datasets[0]["Nfish"] # number of fish, take from the first set;
                                 # don't bother checking if same for all
        key_list = ["perp_noneSee",
                    "perp_oneSees", "perp_bothSee",
                    "perp_larger_fish_sees", "perp_smaller_fish_sees",
                    "contact_any", "contact_head_head", "contact_head_body",
                    "contact_tail_tail",
                    "contact_larger_fish_head", "contact_smaller_fish_head",
                    "contact_inferred", "tail_rubbing_AP", "tail_rubbing_P",
                    "maintain_proximity_1", "maintain_proximity_2", "anyPairBehavior"]
        for j in range(Nfish):
            key_list.extend([f"Cbend_Fish{j}"])
        key_list.extend(["Cbend_any"])  # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"Rbend_Fish{j}"])
        key_list.extend(["Rbend_any"])  
        for j in range(Nfish):
            key_list.extend([f"Jbend_Fish{j}"])
        key_list.extend(["Jbend_any"]) # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"approaching_Fish{j}"])
        if Nfish > 1:
            key_list.extend(["approaching_any"])
            key_list.extend(["approaching_all"])
        for j in range(Nfish):
            key_list.extend([f"fleeing_Fish{j}"])
        if Nfish > 1:
            key_list.extend(["fleeing_any"])
            key_list.extend(["fleeing_all"])
        for j in range(Nfish):
            key_list.extend([f"isMoving_Fish{j}"])
        key_list.extend(["isMoving_any", "isMoving_all"]) # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"isBending_Fish{j}"])
        key_list.extend(["isBending_any", "isBending_all"]) # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"isActive_Fish{j}"])
        key_list.extend(["isActive_any", "isActive_all"]) # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"close_to_edge_Fish{j}"])
        key_list.extend(["close_to_edge_any", "close_to_edge_all"]) # formerly had a condition "if Nfish > 1:"
        key_list.extend(["edge_frames", "bad_bodyTrack_frames"])
        # Remove any keys that are not in the first dataset, for example
        # two-fish behaviors if that dataset was for single fish data
        key_list_revised = [key for key in key_list if key in datasets[0]]
        
        
        if params['allDatasets_markFrames_ExcelFile'] is not None:
            # Mark frames for each dataset
            # Create the ExcelWriter object with path validation
            excel_filename = sanitize_filename(params['allDatasets_markFrames_ExcelFile'])
            excel_file_path = output_path / excel_filename
            
            # Check if full path length is reasonable
            if len(str(excel_file_path)) > 250:  # Conservative limit
                # Try shortening the filename
                name, ext = excel_filename.rsplit('.', 1)
                shortened_name = name[:100] + '_shortened.' + ext
                excel_file_path = output_path / shortened_name
                print(f"Warning: Excel filename was too long, shortened to: {shortened_name}")
            
            try:
                writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
            except (FileNotFoundError, OSError, PermissionError) as e:
                print(f"Error creating Excel file: {e}")
                print(f"Attempted path: {excel_file_path}")
                print(f"Path length: {len(str(excel_file_path))} characters")
                
                # Fallback: try with a very short filename
                fallback_name = "behav_frame.xlsx"
                excel_file_path = output_path / fallback_name
                print(f"Trying fallback filename: {fallback_name}")
                writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
    
            # Call the function to write frames for each dataset
            print('   Marking behaviors in each frame for each dataset.')
            t_mark_frames_start = perf_counter()
            for j in range(N_datasets):
                # Excel worksheet name handling - more robust sanitization
                dataset_name = str(datasets[j]["dataset_name"])
                sheet_name = sanitize_sheet_name(dataset_name)
                
                mark_behavior_frames_Excel(writer, datasets[j], key_list_revised, 
                                           sheet_name, mark=1)
            t_mark_frames_end = perf_counter()
            print(f'       ... {N_datasets} x mark_behavior_frames_Excel: ' + 
                  f'time {t_mark_frames_end - t_mark_frames_start:.1f} s')
            
            # Save and close the Excel file
            try:
                writer.close()
            except Exception as e:
                print(f"Warning: Error closing Excel writer: {e}")
                # Try to save manually if possible
                try:
                    writer.save()
                except:
                    pass
        else:
            print('   *Not* creating Excel file with marked behaviors in each frame.')
            
        print('   Writing summary text file and basic measurements for each dataset.')
        # For each dataset, summary text file and basic measurements    
        t_writing_summary_start = perf_counter()
        
        # Change to output directory for the text file functions
        # (assuming they expect to write in current directory)
        os.chdir(output_path)
        
        try:
            for j in range(N_datasets):
                # Write for this dataset: summary in text file
                write_behavior_txt_file(datasets[j], key_list_revised)
                # Write for this dataset: frame-by-frame "basic measurements"
                write_basicMeasurements_file(datasets[j])
        finally:
            # Always restore original directory
            os.chdir(original_cwd)
            
        t_writing_summary_end = perf_counter()
        print( f'   ... Elapsed time {t_writing_summary_end - t_writing_summary_start:.1f} s')
            
            
        # Excel workbook for summary of all behavior counts, durations,
        # relative durations, and relative durations normalized to activity
        print('   Writing summary of all behavior counts, durations to ' + 
              f"   {params['allDatasets_ExcelFile']}")
        initial_keys = ["dataset_name", "fps", "image_scale",
                        "total_time_seconds",
                        "speed_mm_s_mean", "speed_whenMoving_mm_s_mean",
                        "angular_speed_rad_s_mean", "angular_speed_whenMoving_rad_s_mean",
                        "bout_rate_bpm", "bout_duration_s", "bout_ibi_s",
                        "fish_length_Delta_mm_mean",
                        "head_head_distance_mm_mean", "closest_distance_mm_mean",
                        "AngleXCorr_mean"]
        initial_strings = ["Dataset", "Frames per sec",  "Image scale (um/px)",
                           "Total Time (s)",
                           "Mean speed (mm/s)", "Mean moving speed (mm/s)",
                           "Mean angular speed (rad/s)",
                           "Mean moving angular speed (rad/s)",
                           "Mean bout rate (/min)", "Mean bout duration (s)", "Mean inter-bout interval (s)",
                           "Mean difference in fish lengths (mm)",
                           "Mean head-head dist (mm)", "Mean closest distance (mm)",
                           "AngleXCorr_mean"]

        # Remove any keys that are not in the first dataset, for example
        # two-fish behaviors if that dataset was for single fish data
        # Also remove the corresponding strings
        initial_keys_revised = []
        initial_strings_revised = []
        
        for key, name in zip(initial_keys, initial_strings):
            if key in datasets[0]:
                initial_keys_revised.append(key)
                initial_strings_revised.append(name)

        # Sanitize the Excel filename for the summary file
        summary_excel_filename = sanitize_filename(params["allDatasets_ExcelFile"])
        summary_excel_path = output_path / summary_excel_filename
        
        # Check path length for summary file too
        if len(str(summary_excel_path)) > 250:
            name, ext = summary_excel_filename.rsplit('.', 1)
            shortened_name = name[:100] + '_summary.' + ext
            summary_excel_path = output_path / shortened_name
            print(f"Warning: Summary Excel filename was too long, shortened to: {shortened_name}")

        write_behaviorCounts_Excel(str(summary_excel_path), 
                                  datasets, key_list_revised, 
                                  initial_keys_revised, initial_strings_revised)
        
        t_writing_all_end = perf_counter()
        print('   Done writing output files. ' + 
              f'Elapsed time {t_writing_all_end - t_writing_all_start:.1f} s')

        # Return the full path for use by add_statistics_to_excel
        return str(summary_excel_path)
        
    except Exception as e:
        print(f"Error in write_output_files: {e}")
        raise
    finally:
        # Ensure we're back in the original directory
        try:
            os.chdir(original_cwd)
        except:
            pass
    

def sanitize_filename(filename):
    """
    Sanitize filename by removing or replacing invalid characters
    """
    if not filename:
        return "output_file"
    
    # Remove or replace invalid characters for Windows/Unix
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', str(filename))
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename isn't empty after sanitization
    if not sanitized:
        sanitized = "output_file"
    
    return sanitized


def sanitize_sheet_name(dataset_name):
    """
    Sanitize and truncate Excel sheet name to meet Excel's requirements
    """
    if not dataset_name:
        return "Sheet1"
    
    # Excel sheet names cannot contain: [ ] : * ? / \
    invalid_chars = r'[\[\]:*?/\\]'
    sanitized = re.sub(invalid_chars, '_', str(dataset_name))
    
    # Remove leading/trailing spaces
    sanitized = sanitized.strip()
    
    # Truncate to 31 characters (Excel limit)
    if len(sanitized) > 31:
        sanitized = sanitized[-31:]  # Take last 31 chars as in original
    
    # Ensure sheet name isn't empty
    if not sanitized:
        sanitized = "Sheet1"
    
    return sanitized

        
def write_behavior_txt_file(dataset, key_list):
    """
    Creates a txt file of the relevant window frames and event durations
    for a set of social behaviors in a given *single dataset*
    Output text file name: dataset_name + .txt, one per dataset,
    in Analysis output folder

    Inputs:
        dataset  : dictionary with all dataset info
        key_list : list of dictionary keys corresponding to each 
                   behavior to write frame information for

    Returns:
        N/A
    """
    with open(f"{dataset['dataset_name']}.txt", "w") as results_file:
        results_file.write(f"{dataset['dataset_name']}\n")
        results_file.write("Experimental parameters\n")
        results_file.write(f"   Image scale (um/px): {dataset['image_scale']:.1f}\n")
        results_file.write(f"   frames per second: {dataset['fps']:.1f}\n")
        results_file.write(f"   Number of frames: {dataset['Nframes']}\n")
        results_file.write(f"   Duration: {dataset['total_time_seconds']:.1f} s\n")
        results_file.write("Tracking Properties\n")
        results_file.write(f"   Number of spans of continuous good tracking: {dataset['good_tracking_spans']['number']}\n")
        results_file.write(f"   Mean length of spans of continuous good tracking: {dataset['good_tracking_spans']['mean_frames']:.1f} frames\n")
        results_file.write(f"   Minimum length of spans of continuous good tracking: {dataset['good_tracking_spans']['min_frames']} frames\n")
        results_file.write(f"   Maximum length of spans of continuous good tracking: {dataset['good_tracking_spans']['max_frames']} frames\n")
        results_file.write("Basic Properties\n")
        results_file.write(f"   Mean fish length: {dataset['fish_length_mm_mean']:.3f} mm\n")
        results_file.write(f"   Mean fish speed: {dataset['speed_mm_s_mean']:.3f} mm/s\n")
        results_file.write(f"   Mean fish speed when moving > threshold: {dataset['speed_whenMoving_mm_s_mean']:.3f} mm/s\n")
        results_file.write(f"   Mean fish angular speed: {dataset['angular_speed_rad_s_mean']:.5f} rad/s\n")
        results_file.write(f"   Mean fish angular speed when moving > threshold: {dataset['angular_speed_whenMoving_rad_s_mean']:.5f} rad/s\n")
        results_file.write(f"   Bout rate: {dataset['bout_rate_bpm']:.1f} bouts/minute\n")
        results_file.write(f"   Mean bout duration: {dataset['bout_duration_s']:.2f} seconds\n")
        results_file.write(f"   Mean inter-bout interval: {dataset['bout_ibi_s']:.2f} seconds\n")
        if dataset["Nfish"]==2:
            results_file.write(f"   Mean difference in fish length: {dataset['fish_length_Delta_mm_mean']:.3f} mm\n")
            results_file.write(f"   Mean head-to-head distance: {dataset['head_head_distance_mm_mean']:.3f} mm\n")
            results_file.write(f"   Mean closest distance: {dataset['closest_distance_mm_mean']:.3f} mm\n")
        for k in key_list:
            outString = f'{k} N_events: {dataset[k]["combine_frames"].shape[1]}\n' + \
                    f'{k} Total N_frames: {dataset[k]["total_duration"]}\n' + \
                    f'{k} frames: {dataset[k]["combine_frames"][0,:]}\n' + \
                    f'{k} durations: {dataset[k]["combine_frames"][1,:]}\n'
            results_file.write(outString)


def write_basicMeasurements_file(dataset):
    """
    Creates a CSV file of "basic" speed and distance measurements for
    a given *single dataset* at each frame.
    Assesses what to write given number of fish. (For example,
            don't attempt inter-fish distance if Nfish==1)
    Rows = Frames
    Columns = 
        Head-to-head distance (mm) ["head_head_distance_mm"]
        Closest inter-fish distance (mm) ["closest_distance_mm"]
        Speed of each fish (mm/s); frame-to-frame speed, recorded as 0 for the first frame. ["speed_array_mm_s"]
        Angular speed of each fish (rad/s) ["angular_speed_array_rad_s"]. 
        Relative orientation, i.e. angle between heading and 
            head-to-head vector, for each fish (radians) ["relative_orientation"]
        Relative orientation, i.e. angle between heading and 
            head-to-head vector, for each fish (radians) ["relative_orientation"]
        Relative_heading_angle: The difference in heading angle between the 
            two fish (in range [0, pi]), radians. ["relative_heading_angle"]
        Edge flag (1 or 0) ["edge_frames"]
        Bad track (1 or 0) ["bad_bodyTrack_frames"]

    Caution: 
        Number of fish hard-coded as 2
        Key names hard coded!
        
    Output text file name: dataset_name + _basicMeasurements.txt, 
    one per dataset, in Analysis output folder

    Inputs:
        dataset : dictionary with all dataset info

    Returns:
        N/A
    """
    Nframes = dataset["Nframes"] # number of frames
    Nfish = dataset["Nfish"] # number of fish
    frames = np.arange(1, Nframes+1)
    
    # Prepare edge and tracking flags more efficiently
    EdgeFlag = np.zeros(Nframes, dtype=int)
    if len(dataset["edge_frames"]['raw_frames']) > 0:
        EdgeFlagIdx = dataset["edge_frames"]['raw_frames'] - 1
        EdgeFlag[EdgeFlagIdx] = 1
        
    BadTrackFlag = np.zeros(Nframes, dtype=int)
    BadTrackIdx = dataset["bad_bodyTrack_frames"]['raw_frames'] - 1
    BadTrackFlag[BadTrackIdx] = 1

    # Create headers
    headers = ["frame"]
    if Nfish == 2:
        headers.extend(["head_head_distance_mm", "closest_distance_mm", "relative_heading_angle"])
        headers.extend([f"rel_orientation_rad_Fish{j}" for j in range(Nfish)])
    
    headers.extend([f"speed_mm_s_Fish{j}" for j in range(Nfish)])
    headers.extend([f"angular_speed_rad_s_Fish{j}" for j in range(Nfish)])
    headers.extend([f"r_fromCtr_mm_Fish{j}" for j in range(Nfish)])
    headers.extend(["edge flag", "bad tracking"])

    # Create CSV data more efficiently with string building
    # Prepare buffer for rows
    rows = []
    for j in range(Nframes):
        row_parts = [f"{frames[j]}"]
        
        if Nfish == 2:
            row_parts.extend([
                f"{dataset['head_head_distance_mm'][j].item():.3f}",
                f"{dataset['closest_distance_mm'][j].item():.3f}",
                f"{dataset['relative_heading_angle'][j].item():.3f}"
            ])
            row_parts.extend([f"{dataset['relative_orientation'][j, k].item():.3f}" for k in range(Nfish)])
        
        row_parts.extend([f"{dataset['speed_array_mm_s'][j, k].item():.3f}" for k in range(Nfish)])
        row_parts.extend([f"{dataset['angular_speed_array_rad_s'][j, k].item():.3f}" for k in range(Nfish)])
        row_parts.extend([f"{dataset['radial_position_mm'][j, k].item():.3f}" for k in range(Nfish)])
        row_parts.extend([f"{EdgeFlag[j]}", f"{BadTrackFlag[j]}"])
        
        rows.append(",".join(row_parts))
    
    # Write the CSV file in one operation
    with open(f"{dataset['dataset_name']}_basicMeasurements.csv", "w", newline="") as f:
        f.write(",".join(headers) + "\n")
        f.write("\n".join(rows))
    
    
def write_CSV_Excel_YAML(expt_config, params, dataPath, datasets):
    """
    Write the output files (CSV, Excel, and YAML (parameters))
    Calls write_output_files(), add_statistics_to_excel()
    
    expt_config : experimental configuration dictionary
    params : analysis parameters
    dataPath : primary output folder, e.g. with CSV files
    datasets : list of dictionaries of datasets
    
    """
    print(f'\nWriting to dataPath: {dataPath}')
    # Write the output files (CSV, Excel) 
    output_path = os.path.join(dataPath, params['output_subFolder'])
    excel_file_path = write_output_files(params, output_path, datasets)
    
    # Modify the Excel file with statistics
    print(f'\nAdding statistics to Excel file: {excel_file_path}')
    add_statistics_to_excel(excel_file_path)
    
    # Write a YAML file with parameters  
    more_param_output = {'dataPath': dataPath}
    all_outputs = expt_config | params | more_param_output
    print('\nWriting output YAML file.')
    yaml_path = os.path.join(output_path, 'all_params.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(all_outputs, file)
        

def mark_behavior_frames_Excel(writer, dataset, key_list, sheet_name,
                               mark = 1):
    """
    Create and fill in a sheet in an Excel workbook, marking with 
    “1” a behavior (columns) that occurs in a given frame (rows); 
    one sheet per dataset. .
    
    Optimized Aug. 2025. Uses numpy arrays and vectorized operation.
    
    Inputs:
        writer (pandas.ExcelWriter): The ExcelWriter object representing the Excel file.
        dataset (dict): Dictionary with all dataset info.
        key_list (list): List of dictionary keys corresponding to each behavior to write.
        sheet_name (str): Name of the sheet to be created.
        mark : str or int or bool, default 1
            The value to place in marked cells. Using True/1 is faster than 
            the string 'X'.
            
    Returns:
        N/A
        
    """
    
    maxFrame = int(np.max(dataset["frameArray"]))
    
    n_behaviors = len(key_list)
    # data_array = np.full((maxFrame, n_behaviors), '', dtype='U17')
    # Work in boolean first (fastest). We'll map to 'mark' at the end if needed.
    data_bool = np.zeros((maxFrame, n_behaviors), dtype=bool)
    
    for col_idx, k in enumerate(key_list):
        behavior_data = dataset[k]["combine_frames"]
        # Expect shape (2, N): row 0 -> starts (1-based), row 1 -> durations
        # Skip if no events
        if behavior_data.shape[1] == 0:
            continue

        # Extract frame ranges
        starts = behavior_data[0, :].astype(np.int64) - 1  # 0-indexed
        durations = behavior_data[1, :].astype(np.int64)

        # Clamp invalid starts/durations
        valid = (starts >= 0) & (durations > 0) & (starts < maxFrame)
        if not np.any(valid):
            continue
        starts = starts[valid]
        durations = durations[valid]
        # Compute ends and clamp to maxFrame
        ends = starts + durations
        np.minimum(ends, maxFrame, out=ends)
            
        # Prefix-sum difference array to mark ranges efficiently
        diff = np.zeros(maxFrame + 1, dtype=np.int32)
        # Add +1 at starts
        np.add.at(diff, starts, 1)
        # Subtract -1 at ends
        np.add.at(diff, ends, -1)        
        
        # Create boolean mask for all frames that should be marked
        mask = np.cumsum(diff[:-1]) > 0
        data_bool[:, col_idx] = mask
        
    
    # Prepare DataFrame
    frames = np.arange(1, maxFrame + 1, dtype=np.int32)

    # If mark is boolean or numeric, write directly (fastest).
    # If it's a string, convert only marked cells to that string, else empty.
    if isinstance(mark, bool) or isinstance(mark, (int, np.integer)):
        df = pd.DataFrame(data_bool.astype(type(mark)), columns=key_list)
        # If mark is 1 and unmarked should be 0, that's already correct.
        # If mark is True and unmarked should be False, also correct.
        pass
    else:
        # mark is likely 'X' (string). Create a small object array and fill selectively.
        df = pd.DataFrame('', index=np.arange(maxFrame), columns=key_list)
        # Efficient column-wise assignment: where True -> mark
        for j, col in enumerate(key_list):
            # np.where returns an array; assign to the column
            df[col] = np.where(data_bool[:, j], mark, '')

    df.insert(0, 'Frame', frames)

    # Write once
    # Tip: Construct your ExcelWriter with engine='xlsxwriter' for speed, e.g.:
    # with pd.ExcelWriter(path, engine='xlsxwriter', options={'strings_to_numbers': True}) as writer:
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    

def write_behaviorCounts_Excel(ExcelFileName, datasets, key_list, 
                              initial_keys, initial_strings):
    """
    Creates an Excel with summary statistics of each behavior for each dataset, 
    indicating the number of events, duration (number of frames), 
    and relative duration of each of the behaviors. 
    
    The function also calculates durations relative to the number of 
    "active" frames, both for "isActive_any" and "isActive_all". 
    These are not saved outside this function.
    
    Each row is one dataset. Each column is one behavior; 
    the first few columns are general dataset properties. 
    In addition, separated by a blank row, each sheet contains 
    statistics over the datasets (mean, std. dev., and s.e.m.).
        
    Output text file name: dataset_name + _basicMeasurements.txt, 
    one per dataset, in Analysis output folder

    Inputs:
        ExcelFileName : File name to write
        datasets : list of dictionaries with behavior information,
        key_list : list of behavior dictionary keys to write; note that each 
                    contains Nframes, durations, and relative durations
        initial_keys : keys to write that are single values, not behavior
                    dictionaries 
        initial_strings : column header strings corresponding to initial_keys

    Returns:
        N/A
    """
    t_write_behavCounts_start = perf_counter()

    Ndatasets = len(datasets)

    # Calculate relative durations in one pass
    for j in range(Ndatasets):
        # Calculate these values once per dataset
        active_any_total = datasets[j]["isActive_any"]["total_duration"]
        active_all_total = datasets[j]["isActive_all"]["total_duration"]
        
        for key in key_list:
            total_duration = datasets[j][key]["total_duration"]
            datasets[j][key]['rel_duration_active_any'] = total_duration / active_any_total 
            datasets[j][key]['rel_duration_active_all'] = total_duration / active_all_total
    
    # Create a new Excel writer object
    writer = pd.ExcelWriter(ExcelFileName, engine='xlsxwriter')

    # Define the sheet names and corresponding data keys
    sheets = {
        "N_events": "N_events",
        "Durations (frames)": "total_duration",
        "Relative Durations": "relative_duration",
        "Durations rel Any Active": "rel_duration_active_any",
        "Durations rel All Active": "rel_duration_active_all"
    }

    # Process all sheets in one pass
    for sheet_name, data_key in sheets.items():
        # Create data for the dataframe
        data = [
            [datasets[j].get(key, {}) for key in initial_keys] + 
            [datasets[j][key][data_key] for key in key_list]
            for j in range(Ndatasets)
        ]
        
        # Create DataFrame and write to Excel
        df = pd.DataFrame(data, columns=initial_strings + key_list)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the Excel file
    writer.close()
    t_write_behavCounts_end = perf_counter()
    print('      write_behaviorCounts_Excel: ' + 
          f'time {t_write_behavCounts_end - t_write_behavCounts_start:.1f} s')


def add_statistics_to_excel(file_path='behaviorCounts.xlsx'):
    """
    Modify the Excel sheet containing behavior counts to include
    summary statistics for all datasets (e.g. average for 
    each behavior)
    
    Parameters:
    -----------
    file_path : str or Path
        Full path to the Excel file (can be relative or absolute)
    """
    # Convert to Path object for better handling
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        # Try to find it in current directory if only filename was provided
        if not file_path.is_absolute():
            cwd_file = Path.cwd() / file_path.name
            if cwd_file.exists():
                file_path = cwd_file
            else:
                print(f"Error: Excel file not found at: {file_path}")
                print(f"Current working directory: {Path.cwd()}")
                print(f"Also tried: {cwd_file}")
                
                # List files in current directory for debugging
                try:
                    files = list(Path.cwd().glob("*.xlsx"))
                    if files:
                        print(f"Excel files in current directory: {[f.name for f in files]}")
                    else:
                        print("No .xlsx files found in current directory")
                except:
                    pass
                
                raise FileNotFoundError(f"Could not find Excel file: {file_path}")
    
    print(f"Processing Excel file: {file_path}")
    
    try:
        # Load the Excel file
        xls = pd.ExcelFile(file_path)
        
        # Create a temporary file for writing (to avoid conflicts)
        temp_file = file_path.parent / f"temp_{file_path.name}"
        writer = pd.ExcelWriter(temp_file, engine='openpyxl')
        
        for sheet_name in xls.sheet_names:
            try:
                # Read the sheet
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # Separate non-numeric and numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) == 0:
                    print(f"Warning: No numeric columns found in sheet '{sheet_name}', skipping statistics")
                    # Just write the original data
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue
                
                # Calculate statistics for numeric columns
                mean = df[numeric_cols].mean()
                std = df[numeric_cols].std()
                n = df[numeric_cols].count()
                sem = std / np.sqrt(n)
                
                # Write the original data first
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Get the worksheet to add statistics
                worksheet = writer.sheets[sheet_name]
                
                # Add a blank row, then statistics
                start_row = len(df) + 3  # +2 for header and +1 for blank row
                
                # Create statistics data
                stats_data = [
                    ['Mean'] + [mean.get(col, '') for col in numeric_cols],
                    ['Std. Dev.'] + [std.get(col, '') for col in numeric_cols],
                    ['Std. Error of Mean'] + [sem.get(col, '') for col in numeric_cols],
                    ['N datasets'] + [n.get(col, '') for col in numeric_cols]
                ]
                
                # Write statistics with proper column alignment
                non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                
                for i, row_data in enumerate(stats_data):
                    # Start with the statistic name
                    worksheet.cell(row=start_row + i, column=1, value=row_data[0])
                    
                    # Fill in empty cells for non-numeric columns
                    for j, col in enumerate(non_numeric_cols[1:], start=2):  # Skip first col (statistic name)
                        worksheet.cell(row=start_row + i, column=j, value='')
                    
                    # Add the numeric values in the correct columns
                    numeric_start_col = len(non_numeric_cols) + 1
                    for j, value in enumerate(row_data[1:]):
                        if value != '':  # Only write non-empty values
                            worksheet.cell(row=start_row + i, column=numeric_start_col + j, value=value)
                
                # Adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value is not None:
                                length = len(str(cell.value))
                                if length > max_length:
                                    max_length = length
                        except:
                            pass
                    
                    # Set width with some padding
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 for very long headers
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                    
            except Exception as e:
                print(f"Error processing sheet '{sheet_name}': {e}")
                # Still write the original data even if statistics fail
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                except:
                    print(f"Could not even write original data for sheet '{sheet_name}'")
        
        # Save and close the writer
        writer.close()
        xls.close()
        
        # Replace the original file with the updated one
        if file_path.exists():
            file_path.unlink()  # Delete original
        temp_file.rename(file_path)  # Rename temp to original
        
        print(f"Successfully added statistics to: {file_path}")
        
    except Exception as e:
        print(f"Error in add_statistics_to_excel: {e}")
        # Clean up temp file if it exists
        temp_file = file_path.parent / f"temp_{file_path.name}"
        if temp_file.exists():
            temp_file.unlink()
        raise


def simple_write_CSV(x, y_list, filename, header_strings=None):
    """
    Simple function to save x and y arrays to a CSV file, intended to
    output the points plotted on graphs, especially from plot_probability_distr()

    Parameters:
    x (numpy.ndarray): 1D array for the x column.
    y_list (list of numpy.ndarray): List of 1D arrays for y columns.
    filename (str): Output CSV file name.
    header_strings (list of str, optional): Custom header names. 
        If None, defaults to ['x', 'y1', 'y2', ...].
    """
    # Validate input lengths
    length = len(x)
    for i, y in enumerate(y_list):
        if (length > 1) and (len(y) != length):
            raise ValueError(f"Length mismatch: y[{i}] has length {len(y)}, expected {length}.")

    # Create DataFrame
    data = {'x': x}
    for i, y in enumerate(y_list):
        data[f'y{i+1}'] = y

    df = pd.DataFrame(data)

    # Apply custom header if provided
    if header_strings:
        if len(header_strings) != len(df.columns):
            raise ValueError("header_string length must match number of columns.")
        df.columns = header_strings

    # Save to CSV
    print(f'\nWriting CSV file: {filename}\n')
    df.to_csv(filename, index=False)

    
def combine_images_to_tiff(filenamestring, path, ext="png", 
                           exclude_string=None):
    """
    Load all images containing a given string in their filename and save as 
    multipage TIFF.
    Orders pages by the datestamp of the image files, oldest-newest
    
    Parameters:
    -----------
    filenamestring : str
        String to search for in filenames
    path : str or Path
        Directory path to search for images
    ext : str, optional
        File extension to filter by (default: "png")
    exclude_string : str, optional
        String to exclude from filenames (default: None). If '', make None
    
    Returns:
    --------
    str
        Path to the created multipage TIFF file, or None if no matching files found
    """
    # Convert path to Path object
    search_path = Path(path)
    
    # Ensure extension starts with a dot
    if not ext.startswith('.'):
        ext = f'.{ext}'
    
    if exclude_string=='':
        exclude_string = None
        
    # Find all matching files
    matching_files = [
        f for f in search_path.glob(f'*{ext}')
        if filenamestring in f.stem
    ]
    
    # Filter out files containing exclude_string
    if exclude_string is not None:
        matching_files = [
            f for f in matching_files
            if exclude_string not in f.stem
        ]
    
    # Sort files by modification time (oldest to newest)
    matching_files = sorted(matching_files, key=lambda f: f.stat().st_mtime)
        
    if not matching_files:
        exclude_msg = f" (excluding '{exclude_string}')" if exclude_string else ""
        print(f"No files found with '{filenamestring}' in filename{exclude_msg} and extension '{ext}'")
        return None
    
    print(f"Found {len(matching_files)} matching files:")
    for f in matching_files:
        print(f"  - {f.name}")
    
    # Load all images as numpy arrays
    image_arrays = []
    for file_path in matching_files:
        try:
            img_array = iio.imread(file_path)
            image_arrays.append(img_array)
        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
    
    if not image_arrays:
        print("No images could be loaded successfully")
        return None
    
    # Check if all images have the same shape
    shapes = [img.shape for img in image_arrays]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print(f"\nWarning: Images have different shapes: {unique_shapes}")
        print("Padding smaller images with black pixels to match largest dimensions...")
        
        # Find maximum dimensions
        max_height = max(img.shape[0] for img in image_arrays)
        max_width = max(img.shape[1] for img in image_arrays)
        
        # Check if images are grayscale or color
        has_channels = len(image_arrays[0].shape) == 3
        if has_channels:
            n_channels = image_arrays[0].shape[2]
        
        # Pad each image to match max dimensions
        padded_arrays = []
        for img in image_arrays:
            height, width = img.shape[0], img.shape[1]
            
            if height == max_height and width == max_width:
                padded_arrays.append(img)
            else:
                # Calculate padding (top=0, bottom=pad_h, left=0, right=pad_w)
                pad_height = max_height - height
                pad_width = max_width - width
                
                if has_channels:
                    # For color images: pad each channel
                    padded = np.pad(
                        img,
                        ((0, pad_height), (0, pad_width), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                else:
                    # For grayscale images
                    padded = np.pad(
                        img,
                        ((0, pad_height), (0, pad_width)),
                        mode='constant',
                        constant_values=0
                    )
                
                padded_arrays.append(padded)
                print(f"  Padded {height}x{width} -> {max_height}x{max_width}")
        
        image_arrays = padded_arrays
        print(f"All images now have shape: {image_arrays[0].shape}")
    
    # Stack images along first axis (Z-axis for ImageJ)
    image_stack = np.stack(image_arrays, axis=0)
    
    n_images = len(image_arrays)
    
    # Create output filename
    output_path = search_path / f'{filenamestring}_combined.tiff'
    
    # Prepare metadata for ImageJ
    metadata = {
        'slices': n_images
    }
    
    # Determine axes based on image dimensions
    if image_stack.ndim == 3:
        # Grayscale images: (slices, height, width)
        axes = 'ZYX'
    elif image_stack.ndim == 4:
        # Color images: (slices, height, width, channels)
        axes = 'ZYXC'
    else:
        axes = None
    
    # Save as ImageJ-compatible TIFF
    tifffile.imwrite(
        output_path,
        image_stack,
        imagej=True,
        metadata=metadata,
        compression='deflate'
    )
    
    print(f"\nSuccessfully created ImageJ-compatible multipage TIFF: {output_path}")
    print(f"Total slices: {n_images}")
    print(f"Metadata: {metadata}")
    
    return str(output_path)

def modify_params(pickleFilePath, pickleFileName, dict_to_modify = 'params',
                  keys_to_update = ['contact_distance_threshold_mm', 
                                    'contact_inferred_distance_threshold_mm', 
                                    'contact_inferred_window'],
                  vals_to_update = [2.5, 3.5, 3],
                  keys_to_add = ['max_motion_gap_s', 
                                 'min_proximity_duration_s'],
                  vals_to_add = [0.5, 0.0]
                  ):
    """
    Load pickle file and modify the analysis parameters, adding keys
        or changing key values.
    The dict to update is likely 'params' (analysis parameters) but
        could be expt_config (e.g. image scale)
    Useful if we want to modify analysis without re-reading all CSVs
    
    Parameters:
    -----------
    pickleFilePath : str or Path
        Path of pickle file
    pickleFileName : str or Path
        File name of pickle file
        Note that the params and expt_config dictionaries are stored 
        in the “datasets” pickle file, not “positionData”.
        e.g. pickleFileName = r'old_Behavior_atah_1_pair_Group_1_datasets.pickle'
    keys_to_update : list 
        keys to update; can be None
    vals_to_update : list
        corresponding values to update (list)
    keys_to_add : list
        keys to add; can be None
    vals_to_add : list 
        corresponding values to add (list )
    
    Returns:
    --------
    None (overwrites original pickle file)
    
    """    

    # Step 1: Load all objects from the pickle file
    pickleCompletePath = os.path.join(pickleFilePath, pickleFileName)
    objects = []
    with open(pickleCompletePath, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
                objects.append(obj)
            except EOFError:
                break
    
    print('\nOriginal values:')
    print(objects[0][dict_to_modify])
    # Step 2: Find and modify the 'params' dictionary
    for obj in objects:
        if isinstance(obj, dict) and dict_to_modify in obj:
            # Update keys
            if keys_to_update is not None:
                for k, v in zip(keys_to_update, vals_to_update):
                    obj[dict_to_modify][k] = v  # Update key1
            # Add keys
            if keys_to_add is not None:
                for k, v in zip(keys_to_add, vals_to_add):
                    obj[dict_to_modify][k] = v  # Add new key
            break  # Assuming only one 'params' dict exists
    
    print('\nNew values:')
    print(objects[0][dict_to_modify])

    # Step 3: Write all objects back to the pickle file
    with open(pickleCompletePath, "wb") as f:
        for obj in objects:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def plot_probability_distr(x_list, bin_width=1.0, bin_range=[None, None], 
                           xlabelStr='x', titleStr='Probability density',
                           yScaleType = 'log', 
                           normalize_by_inv_bincenter = False, 
                           flatten_dataset = False,
                           plot_each_dataset = True,
                           plot_sem_band = False,
                           xlim = None, ylim = None, polarPlot = False,
                           unit_scaling_for_plot = 1.0,
                           color = 'black', 
                           outputFileName = None,
                           closeFigure = False,
                           outputCSVFileName = None):
    """
    Calculate and plot the probability distribution (normalized histogram) 
    for the concatenated array of all items in input x_list. 
    Can also plot the probability distribution for each array in x_list 
    (semi-transparent) and plot the uncertainty in the concatenated 
    probability distribution (semi-transparent). 
    Can also output the concatenated probability distribution.
    Can plot in polar coordinates – useful for angle distributions.
    Typically use this with the output of combine_all_values_constrained().
    Optional writing of values to a CSV file.
    
    Inputs:
       x_list : list of numpy arrays
       bin_width : bin width
       bin_range : list of smallest and largest bin edges; if None
                   use min and max of all arrays combined. 
                   Or if None and polarPlot==True use nearest
                   integer multiple of pi below/above min and max 
       xlabelStr : string for x axis label
       titleStr : string for title
       yScaleType : either "log" or "linear"
       normalize_by_inv_bincenter : if True, normalize counts by 
                   1 / bin centers -- i.e. if we're plotting p(r)
                   and need to normalize by 1/r for disk density.
       flatten_dataset : if true, flatten each dataset's array for 
                           individual dataset plots. If false, plot each
                           array column (fish, probably) separately
       plot_each_dataset : (bool) if True, plot the prob. distr. for each array               
       plot_sem_band : (bool) if True, plot the +/- sem from each dataset
           considered individually as a shaded band around the pooled value
       xlim : (optional) tuple of min, max x-axis limits
       ylim : (optional) tuple of min, max y-axis limits
       polarPlot : if True, use polar coordinates for the histogram.
                   Will not plot x and y labels.
                   Strongly reommended to set y limit (which will set r limit)
                   NOTE: for polar plot, labels will be in degrees automatically
                   Can't use unit_scaling_for_plot to rescale
       unit_scaling_for_plot : float for x; 
                   for plots only, multiply values by unit_scaling_for_plot,  
                   for example to convert radians to degrees.
                   Note that xlim should *not* incorporate this; the code takes
                   care of changing the limits
                   Not applicable for for polar plots.
       color: plot color (uses alpha for indiv. dataset colors)
       outputFileName : if not None, save the figure with this filename 
                       (include extension)
       closeFigure : (bool) if True, close a figure after creating it.
       outputCSVFileName : if not None, save to a CSV file the following (columns):
                   - plotted "X" positions (bin_centers*unit_scaling_for_plot)
                   - plotted mean "Y" positions (prob_dist_all)
                   - Standard deviation (prob_dist_each_std)
                   - Standard error of the mean (prob_dist_each_sem)
                   - Each individual probability distribution array
                       
    Returns:
        prob_dist_all, bin_centers : concatenated probability distribution
            and bin centers
        prob_dist_each_std, prob_dist_each_sem : standard deviation and
            s.e.m. of the probability distribution of each dataset separately.
        Nsets_total : total number of individual datasets (= number of
                            datasets x number of fish)
    """ 
    if polarPlot:
        if np.abs(unit_scaling_for_plot - 1.0) > 1e-6:
            print('For polar plot, cannot rescale axes to degrees; forcing')
            print(f' unit_scaling_for_plot to be 1.0, not {unit_scaling_for_plot:.3f}')
            unit_scaling_for_plot = 1.0
            
    # Concatenate all arrays for the overall, combined plot
    x_all = np.concatenate([arr.flatten() for arr in x_list])
    
    # Determine bin range if not provided
    slight_theta_shift = 0.01 # for axis range of polar plots
    if bin_range[0] is None:
        if polarPlot==True:
            bin_range[0] = (np.nanmin(x_all) // np.pi)*np.pi # floor * pi
        else:
            bin_range[0] = np.nanmin(x_all)
    if bin_range[1] is None:
        if polarPlot==True:
            # next mult of pi
            bin_range[1] = (((np.nanmax(x_all)-slight_theta_shift) // np.pi)+1)*np.pi
        else:
            bin_range[1] = np.nanmax(x_all)
    Nbins = np.round((bin_range[1] - bin_range[0])/bin_width + 1).astype(int)
    bin_edges = np.linspace(bin_range[0], bin_range[1], num=Nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate concatenated distribution
    counts_all, _ = np.histogram(x_all, bins=bin_edges)
    if normalize_by_inv_bincenter:
        counts_all = counts_all / bin_centers
    prob_dist_all = counts_all / np.sum(counts_all) / bin_width
    
    
    # Calculate individual distributions and their variance
    prob_dist_each = []
    datasetLabel_each = []
    for i, x in enumerate(x_list):
        if flatten_dataset:
            x = x.flatten()
        if x.ndim==1:
            Nsets = 1
            counts, _ = np.histogram(x.flatten(), bins=bin_edges)
            if normalize_by_inv_bincenter:
                counts = counts / bin_centers
            datasetLabel_each.append(f'Dataset {i+1}')
            prob_dist_each.append(counts / np.sum(counts) / bin_width)
        else:
            Nsets = x.shape[1] # number of measures per dataset, e.g. number of fish
            for j in range(Nsets):
                counts, _ = np.histogram(x[:,j].flatten(), bins=bin_edges)
                if normalize_by_inv_bincenter:
                    counts = counts / bin_centers
                datasetLabel_each.append(f'Dataset {i+1}: {j+1}')
                prob_dist_each.append(counts / np.sum(counts) / bin_width)
    Nsets_total = len(prob_dist_each)
    print(f'Calculating variance from {Nsets_total} datasets x fish')
    prob_dist_each_std = np.std(prob_dist_each, axis=0)
    prob_dist_each_sem = prob_dist_each_std / np.sqrt(Nsets_total)

    # Plot the concatenated distribution, etc.
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(polar=polarPlot)
    plt.plot(bin_centers*unit_scaling_for_plot, prob_dist_all, 
             color=color, linewidth=2, 
             label='All Datasets')
    
    # Plot sem as shaded bands
    alpha_sem = 0.4
    if plot_sem_band:
        plt.fill_between(bin_centers*unit_scaling_for_plot, 
                         prob_dist_all - prob_dist_each_sem, 
                         prob_dist_all + prob_dist_each_sem, color=color, 
                         alpha=alpha_sem, label='s.e.m.')
        
    # Plot individual distributions
    if plot_each_dataset:
        alpha_each = np.max((0.7/len(x_list), 0.05))
        for i in range(len(x_list)):
            ax.plot(bin_centers*unit_scaling_for_plot, prob_dist_each[i], color=color, 
                        alpha=alpha_each, label=datasetLabel_each[i])

    # Labels, etc.
    if not polarPlot:
        plt.xlabel(xlabelStr, fontsize=16)
        plt.ylabel('Probability density', fontsize=16)
    plt.title(titleStr, fontsize=18)
    if polarPlot:
        ax.set_thetalim(bin_range[0]*unit_scaling_for_plot, 
                        bin_range[1]*unit_scaling_for_plot)
    if xlim is not None:
        plt.xlim(tuple([item * unit_scaling_for_plot for item in xlim]))

    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale(yScaleType)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend()
    plt.show()
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')

    if closeFigure:
        print(f'Closing figure {titleStr}')
        plt.close(fig)
        
    # Output points to CSV (optional)
    if outputCSVFileName is not None:
        header_strings = [xlabelStr.replace(',', '_'), 
                          'prob', 'prob_std', 'prob_sem']
        for j in range(len(x_list)):
            header_strings.append(f'prob_Dataset_{j+1}')
        p_list_to_output = [prob_dist_all, prob_dist_each_std, prob_dist_each_sem]
        
        for j in range(len(x_list)):
            p_list_to_output.append(prob_dist_each[j],)
        simple_write_CSV(bin_centers*unit_scaling_for_plot, 
                         p_list_to_output, 
                         filename =  outputCSVFileName, 
                         header_strings=header_strings)
            
    return prob_dist_all, bin_centers, prob_dist_each_std, \
           prob_dist_each_sem, Nsets_total


def plot_behavior_probability_bin_by_property(bin_centers, counts, 
                                    behavior_key_for_title, property_key_for_label,
                                    xlabelStr=None, ylabelStr='Number of events',
                                    titleStr=None, normalize=True,
                                    xlim = None, ylim = None, 
                                    outputFileName=None):
    """
    Plot the probability of a behavior binned by some quantitative property,
    such as distance. Note that the behavior probability is conditional
    on the quantitative property bin, regardless of the probability of 
    that property bin. In other words, for example, we want the probability
    of a turn at d = 5 mm and the probability of a turn at d = 10 mm not
    to depend on how many times d = 5 mm and d = 10 mm occur. The former 
    is (# turning frames when d = 5mm) / (# frames when d = 5 mm)
    
    Parameters
    ----------
    bin_centers : array of bin centers
    counts : array of counts in each bin
    behavior_key_for_title : str, behavior name (for title)
    property_key_for_label : str, property name (for labels)
    xlabelStr : str or None, x-axis label (default based on property_key)
    ylabelStr : str, y-axis label
    titleStr : str or None, title (default based on behavior and property)
    normalize : bool, if True normalize to probability density
    xlim : (optional) tuple of min, max x-axis limits
    ylim : (optional) tuple of min, max y-axis limits
    outputFileName : str or None, filename to save figure
    """
    
    print('\n\n*Unfinished* -- does not normalize by the number of occurrences of that property overall!')
    _ = input('Press enter to continue; recommend abort.')
    
    if bin_centers is None or counts is None:
        print("No data to plot")
        return
    
    # Default labels
    if xlabelStr is None:
        xlabelStr = property_key_for_label.replace('_', ' ')
    
    if titleStr is None:
        titleStr = f'{behavior_key_for_title}: Distribution by {property_key_for_label}'
    
    # Normalize if requested
    if normalize:
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
        total_area = np.sum(counts) * bin_width
        plot_counts = counts / total_area if total_area > 0 else counts
        ylabelStr = 'Probability density'
    else:
        plot_counts = counts
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot as bar chart
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    plt.plot(bin_centers, plot_counts, linestyle='--', linewidth=2.0, 
            color='steelblue', marker='o', markersize=12)
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(xlabelStr, fontsize=16)
    plt.ylabel(ylabelStr, fontsize=16)
    plt.title(titleStr, fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')
    
    plt.show()
    
    
def plot_function_allSets(y_list, x_array = None, 
                           xlabelStr='x', ylabelStr='y', titleStr='Value',
                           average_in_dataset = False,
                           plot_each_dataset = True,
                           xlim = None, ylim = None, 
                           color = 'black', 
                           outputFileName = None,
                           closeFigure = False,
                           outputCSVFileName = None):
    """
    Plot some function that has been calculated for all datasets, 
    such as the autocorrelation, for the average array from all 
    items in y_list and all individual items (semi-transparent)
    x_array (x values) to plot will be the same for all datasets
    
    Inputs:
       y_list : list of numpy arrays
       x_array : x-values, same for all datasets. If None, just use indexes
       xlabelStr : string for x axis label
       ylabelStr : string for y axis label
       titleStr : string for title
       average_in_dataset : if true, average each dataset's arrays for 
                            the individual dataset plots
       plot_each_dataset : (bool) if True, plot each array               
       xlim : (optional) tuple of min, max x-axis limits
       ylim : (optional) tuple of min, max y-axis limits
       color: plot color (uses alpha for indiv. dataset colors)
       outputFileName : if not None, save the figure with this filename 
                       (include extension)
       closeFigure : (bool) if True, close figure after creating it.
       outputCSVFileName : if not None, save to a CSV file the following (columns):
                   - plotted "X" positions (x_array)
                   - plotted mean "Y" positions (prob_dist_all)
                   - Standard deviation (prob_dist_each_std)
                   - Standard error of the mean (prob_dist_each_sem)
                       
    Returns:
        x_array, y_mean : x values and average y values
        
    """ 
    # Ensure y_list is a list of numpy arrays
    if not isinstance(y_list, list):
        raise ValueError("y_list must be a list of numpy arrays")

    # Determine Nfish based on the shape of the first array in y_list
    # Note that Nfish is actually # dof
    if len(y_list[0].shape) == 1:
        Nfish = 1
    else:
        Nfish = y_list[0].shape[1]

    # Determine N (number of x points)
    N = y_list[0].shape[0]

    # Create x_array if not provided
    if x_array is None:
        x_array = np.arange(0, N)
    elif len(x_array) != N:
        raise ValueError("Length of x_array must match the number of rows in y arrays")

    if Nfish == 1:
        y_mean = np.mean([y.flatten() for y in y_list], axis=0)
    else:
        y_mean = np.mean([np.mean(y, axis=1) for y in y_list], axis=0)
    
    if x_array is None:
        x_array = np.arange(len(y_list[1]))

    fig = plt.figure(figsize=(12, 6))

    if plot_each_dataset:
        # Plot individual y data
        alpha_each = np.max((0.7/len(y_list), 0.05))
        for y in y_list:
            if Nfish==1:
                plt.plot(x_array, y, color=color, alpha=alpha_each)            
            else:
                for fish in range(Nfish):
                    plt.plot(x_array, y[:, fish], color=color, alpha=alpha_each)

    # Plot y_mean
    plt.plot(x_array, y_mean, color=color, linewidth=2, label='Mean')

    plt.xlabel(xlabelStr, fontsize=16)
    plt.ylabel(ylabelStr, fontsize=16)
    plt.title(titleStr, fontsize=20)    # plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')
    if closeFigure:
        plt.close(fig)
    
    # Output points to CSV (optional)
    if outputCSVFileName is not None:
        if Nfish==1:
            header_strings = [xlabelStr.replace(',', '_'), f'{ylabelStr}_mean'] + \
                              [f'{ylabelStr}_{i}' for i in range(len(y_list))]
            y_to_write = [y_mean] + y_list
        else:
            y_list_labels = [f'{ylabelStr}_mean']
            y_to_write = [y_mean]
            for j in range(len(y_list)):
                for k in range(Nfish):
                    y_list_labels.append(f'{ylabelStr}_set{j}_fish{k}')
                    y_to_write.append(y_list[j][:,k])
            header_strings = [xlabelStr.replace(',', '_')] + y_list_labels
        simple_write_CSV(x_array, y_to_write, 
                         filename =  outputCSVFileName, 
                         header_strings=header_strings)
        
    return x_array, y_mean


def plot_waterfall_binned_crosscorr(binned_crosscorr_all, bin_centers, t_lag,
                                     bin_counts_all=None, xlabelStr='Time lag (s)', 
                                     titleStr='Distance-Binned Cross-correlation',
                                     offset_scale=0.5,  
                                     colormap='RdBu_r', unit_string = 'mm', 
                                     outputFileName=None,
                                     vmin=None, vmax=None, 
                                     plot_heatmap = False, 
                                     heatmap_ylabelStr='Distance (mm)',
                                     closeFigure = False):
    """
    Create a waterfall-style plot of cross-correlations binned by a parameter
    (e.g., inter-fish distance), with each bin's correlation plotted as a separate 
    curve offset vertically.
    
    Parameters
    ----------
    binned_crosscorr_all : list of 2D numpy arrays from calculate_value_corr_all_binned
    bin_centers : array of bin centers (e.g. distance values)
    t_lag : time lag array
    bin_counts_all : list of bin count arrays (optional, for weighting)
    xlabelStr : x-axis label
    titleStr : plot title
    outputFileName : if not None, save figure with this filename
    colormap : colormap for the waterfall plot
    unit_string : string to append to labels on waterfall plot, indicating units
        for the binning values. Use '' for none
    offset_scale : vertical offset between traces as fraction of 
        max cross-correlation range
    vmin, vmax : color scale limits (if None, use data range)
    plot_heatmap : if True, also make a heatmap plot.
    heatmap_ylabelStr : y-axis label for heatmap
    closeFigure : (bool) if True, close figure after creating it.
    """
    
    # Calculate average across all datasets
    n_datasets = len(binned_crosscorr_all)
    n_bins, n_time_lags = binned_crosscorr_all[0].shape
    
    # Initialize arrays for weighted average
    total_crosscorr = np.zeros((n_bins, n_time_lags))
    total_weights = np.zeros(n_bins)
    
    # Calculate weighted average if bin counts provided, otherwise simple average
    for i, binned_crosscorr in enumerate(binned_crosscorr_all):
        if bin_counts_all is not None and bin_counts_all[i] is not None:
            weights = bin_counts_all[i]
        else:
            weights = np.ones(n_bins)
            
        for bin_idx in range(n_bins):
            # Check if this bin has any valid (non-NaN) data
            if not np.all(np.isnan(binned_crosscorr[bin_idx])) and weights[bin_idx] > 0:
                # Use nansum for proper NaN handling in averaging
                valid_mask = ~np.isnan(binned_crosscorr[bin_idx])
                total_crosscorr[bin_idx] += np.where(valid_mask, 
                                                   binned_crosscorr[bin_idx] * weights[bin_idx], 
                                                   0)
                # Only count weights where we have valid data
                total_weights[bin_idx] += weights[bin_idx]
    
    # Calculate final averages with proper NaN handling
    avg_crosscorr = np.full((n_bins, n_time_lags), np.nan)
    for bin_idx in range(n_bins):
        if total_weights[bin_idx] > 0:
            avg_crosscorr[bin_idx] = total_crosscorr[bin_idx] / total_weights[bin_idx]
            # Set values back to NaN where all original data was NaN
            all_nan_mask = np.all([np.isnan(binned_crosscorr_all[i][bin_idx]) 
                                 for i in range(n_datasets)], axis=0)
            avg_crosscorr[bin_idx][all_nan_mask] = np.nan
    
    # Debug: Print information about the data
    print(f"Bin centers: {bin_centers[:5]}...{bin_centers[-5:]}")
    print(f"Array shape: {avg_crosscorr.shape}")
    print(f"First few rows have data: {[not np.all(np.isnan(avg_crosscorr[i])) for i in range(min(5, avg_crosscorr.shape[0]))]}")
    print(f"Last few rows have data: {[not np.all(np.isnan(avg_crosscorr[i])) for i in range(max(0, avg_crosscorr.shape[0]-5), avg_crosscorr.shape[0])]}")

    # Waterfall plot with constant offset and distance labels
    fig, ax = plt.subplots(figsize=(7, 12))
    
    valid_bins = ~np.all(np.isnan(avg_crosscorr), axis=1)
    valid_bin_indices = np.where(valid_bins)[0]
    n_valid_bins = len(valid_bin_indices)
    
    if n_valid_bins > 0:
        # Calculate constant offset between traces based on correlation data range (ignoring NaNs)
        finite_data = avg_crosscorr[np.isfinite(avg_crosscorr)]
        if len(finite_data) > 0:
            data_range = np.max(finite_data) - np.min(finite_data)
        else:
            data_range = 1.0  # fallback
        offset = offset_scale * data_range
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_valid_bins))
        
        traces_plotted = 0
        for i, bin_idx in enumerate(valid_bin_indices):
            y_values = avg_crosscorr[bin_idx] + i * offset
            
            # Only plot non-NaN portions of the data
            valid_mask = ~np.isnan(y_values)
            if np.any(valid_mask):
                ax.plot(t_lag[valid_mask], y_values[valid_mask], 
                       color=colors[i], linewidth=1.5,
                       label=f'{bin_centers[bin_idx]:.1f} ' + unit_string)
                
                # Add text label showing distance bin center
                # Use the mean of valid y-values for text positioning
                valid_y = y_values[valid_mask]
                text_x = t_lag[-1] * 0.95  # 95% along the x-axis
                text_y = np.mean(valid_y)  # Use mean y-value of valid data
                ax.text(text_x, text_y, f'{bin_centers[bin_idx]:.1f}' + unit_string, 
                       fontsize=10, verticalalignment='center', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                traces_plotted += 1
        
        ax.set_xlabel(xlabelStr, fontsize=14)
        ax.set_ylabel('Cross-correlation + offset', fontsize=14)
        ax.set_title(f'{titleStr}', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # Add legend for a subset of traces to avoid clutter
        if traces_plotted <= 10:
            ax.legend(fontsize=10, loc='upper left')  
        else:
            # Show only every nth trace in legend
            step = max(1, traces_plotted // 8)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::step], labels[::step], fontsize=10, loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No valid data to plot', transform=ax.transAxes, 
               ha='center', va='center', fontsize=16)
    
    plt.tight_layout()
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')
    
    plt.show()

    if closeFigure:
        plt.close(fig)

    # Print summary statistics
    valid_bins = ~np.all(np.isnan(avg_crosscorr), axis=1)  
    print('\nSummary:')
    print(f'Number of datasets: {n_datasets}')
    print(f'Bins with data: {np.sum(valid_bins)}/{len(bin_centers)}')
    print(f'Traces plotted in waterfall: {traces_plotted}')

    if plot_heatmap:
        # Heatmap plot with proper NaN handling
        fig, ax2 = plt.subplots(figsize=(8, 8))
        
        # Calculate vmin/vmax from finite (non-NaN) data only
        finite_data = avg_crosscorr[np.isfinite(avg_crosscorr)]
        if len(finite_data) > 0:
            if vmin is None or vmax is None:
                data_range = np.percentile(finite_data, [5, 95])
                vmin = vmin or data_range[0]
                vmax = vmax or data_range[1]
            print(f'Cross-correlation range: {np.min(finite_data):.3f} to {np.max(finite_data):.3f}')
        else:
            vmin, vmax = -1, 1
            print('No finite data found for heatmap')
        
        # Create masked array for proper NaN handling in imshow
        masked_data = np.ma.masked_where(np.isnan(avg_crosscorr), avg_crosscorr)
        
        # Calculate bin edges for proper extent
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 5
        bin_edges_min = bin_centers[0] - bin_width/2
        bin_edges_max = bin_centers[-1] + bin_width/2
        
        print(f"Extent: [{t_lag[0]}, {t_lag[-1]}, {bin_edges_min}, {bin_edges_max}]")
        
        im = ax2.imshow(masked_data, aspect='auto', origin='lower', 
                        extent=[t_lag[0], t_lag[-1], bin_edges_min, bin_edges_max],
                        cmap=colormap, vmin=vmin, vmax=vmax)
        
        ax2.set_xlabel(xlabelStr, fontsize=14)
        ax2.set_ylabel(heatmap_ylabelStr, fontsize=14)
        ax2.set_title(f'{titleStr}', fontsize=16)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Cross-correlation', fontsize=12)
    
        plt.tight_layout()
        
        if outputFileName is not None:
            plt.savefig('heatmap_' + outputFileName, bbox_inches='tight')
        
        plt.show()

        if closeFigure:
            plt.close(fig)
    

def calculate_property_1Dbinned(datasets, keyName, keyIdx=None,
                                key_is_a_behavior=False,
                                binKeyName='closest_distance_mm',
                                bin_range=(0.0, 50.0), Nbins=20,
                                constraintKey=None, constraintRange=None,
                                constraintIdx=None,
                                use_abs_value=False,
                                use_abs_value_constraint=False,
                                dilate_minus1=True):
    """
    Calculate mean of a quantitative property binned by another property.

    Analogous to calculate_bout_property_binned_by_distance() for IBI, but
    works for any quantitative property or behavior probability.

    If key_is_a_behavior=True, keyName should be a behavior key (e.g. "Jbend_any")
    and the function computes the conditional probability of that behavior in
    each bin of binKeyName.

    Parameters
    ----------
    datasets : list of dataset dictionaries
    keyName : str
        Property to average, or behavior key if key_is_a_behavior=True
    keyIdx : int, str, or None
        Which fish/operation to use for keyName
    key_is_a_behavior : bool
        If True, treat keyName as a behavior and compute its probability
    binKeyName : str
        Property to bin by (e.g., 'closest_distance_mm')
    bin_range : tuple
        (min, max) for binning
    Nbins : int
        Number of bins
    constraintKey, constraintRange, constraintIdx : constraint parameters
    use_abs_value : bool
        Use absolute value of keyName
    use_abs_value_constraint : bool
        Use absolute value of constraint
    dilate_minus1 : bool
        Dilate bad frames by -1

    Returns
    -------
    binned_mean : numpy array (Nbins, 3) — [mean, std, sem] per bin
    bin_centers : numpy array — bin center values
    binned_mean_each_dataset : numpy array (Ndatasets, Nbins)
    binned_mean_each_fish : numpy array (nfish_total, Nbins)
    """
    
    # Set up bins 
    bin_edges = np.linspace(bin_range[0], bin_range[1], Nbins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    Ndatasets = len(datasets)
    
    # Calculate total number of fish across all datasets
    nfish_total = sum(dataset["Nfish"] for dataset in datasets)
    
    # Initialize arrays
    binned_mean_each_dataset = np.full((Ndatasets, Nbins), np.nan)
    binned_mean_each_fish = np.full((nfish_total, Nbins), np.nan)
    
    print(f'\nBinning {keyName} by {binKeyName}.... ', end='')
    
    fish_counter = 0  # Track global fish index across datasets
    
    for j in range(Ndatasets):
        print(f' {j}... ', end='')
        dataset = datasets[j]
        Nfish = dataset["Nfish"]
        frames = dataset["frameArray"]
        idx_offset = min(frames)
        if idx_offset != 1:
            raise ValueError('Minimum frame number is not 1!')

        # Handle bad frames
        badTrackFrames = dataset["bad_bodyTrack_frames"]["raw_frames"]
        if dilate_minus1:
            dilate_badTrackFrames = dilate_frames(badTrackFrames,
                                                  dilate_frames=np.array([-1]))
            bad_frames_set = set(dilate_badTrackFrames)
        else:
            bad_frames_set = set(badTrackFrames)
        
        # Get data
        if key_is_a_behavior:
            behavior_frames = dataset[keyName]["raw_frames"]
            is_in_bf = np.isin(frames, behavior_frames)
            property_data = is_in_bf.astype(float)
        else:
            # Quantitative property
            property_data = get_values_subset(dataset[keyName], keyIdx=keyIdx,
                                              use_abs_value=use_abs_value)

        bin_data = get_values_subset(dataset[binKeyName], keyIdx=None,
                                     use_abs_value=False)
        # Determine effective dimensions
        Ndim_property = get_effective_dims(property_data)
        Ndim_bin = get_effective_dims(bin_data)
        
        # Apply constraints if provided
        # Get constraint data but don't create mask yet - will do per fish
        if constraintKey is not None and constraintRange is not None:
            constraint_data = get_values_subset(dataset[constraintKey],
                                               keyIdx=constraintIdx,
                                               use_abs_value=use_abs_value_constraint)
            Ndim_constraint = get_effective_dims(constraint_data)
        else:
            constraint_data = None
            Ndim_constraint = None
        
        # Remove bad frames
        good_frames_mask = np.isin(frames - idx_offset,
                                   np.array(list(bad_frames_set)) - idx_offset,
                                   invert=True)

        # Loop over each fish
        for k in range(Nfish):
            # Extract data for this specific fish
            if property_data.ndim == 1:
                property_data_fish = property_data
            else:
                property_data_fish = property_data[:, k]
            
            if bin_data.ndim == 1:
                bin_data_fish = bin_data
            else:
                bin_data_fish = bin_data[:, k] if bin_data.shape[1] > 1 else bin_data[:, 0]
            
            good_frames_mask_fish = good_frames_mask if good_frames_mask.ndim == 1 else good_frames_mask[:, k]
            
            # Handle constraint for this fish
            if constraint_data is not None:
                # Extract constraint for this specific fish
                if constraint_data.ndim == 1:
                    # Constraint is 1D (e.g., inter-fish distance), same for all fish
                    constraint_data_fish = constraint_data
                else:
                    # Constraint is multi-dimensional, extract column k
                    constraint_data_fish = constraint_data[:, k]
                
                # Create mask for this fish's constraint
                constraint_mask_fish = ((constraint_data_fish >= constraintRange[0]) &
                                       (constraint_data_fish <= constraintRange[1]))
                valid_mask = good_frames_mask_fish & constraint_mask_fish
            else:
                valid_mask = good_frames_mask_fish
            
            # Bin data for this fish
            for bin_idx in range(Nbins):
                bin_mask = ((bin_data_fish >= bin_edges[bin_idx]) &
                           (bin_data_fish < bin_edges[bin_idx + 1]))
                combined_mask = valid_mask & bin_mask
                
                if np.sum(combined_mask) > 0:
                    binned_mean_each_fish[fish_counter, bin_idx] = np.nanmean(
                        property_data_fish[combined_mask]
                    )
            
            fish_counter += 1
        
        # Calculate mean over all fish in this dataset
        fish_start = fish_counter - Nfish
        fish_end = fish_counter
        binned_mean_each_dataset[j, :] = np.nanmean(
            binned_mean_each_fish[fish_start:fish_end, :], axis=0
        )
    
    print('... done.\n')
    
    # Calculate overall statistics across all fish
    binned_mean = np.zeros((Nbins, 3))
    binned_mean[:, 0] = np.nanmean(binned_mean_each_fish, axis=0)
    binned_mean[:, 1] = np.nanstd(binned_mean_each_fish, axis=0)
    binned_mean[:, 2] = (np.nanstd(binned_mean_each_fish, axis=0) /
                        np.sqrt(np.sum(~np.isnan(binned_mean_each_fish), axis=0)))

    return binned_mean, bin_centers, binned_mean_each_dataset, binned_mean_each_fish


def plot_2D_heatmap(Z, X, Y, Z_unc=None,
                   titleStr=None, xlabelStr='X', ylabelStr='Y', clabelStr='Z',
                   colorRange=None, cmap='RdYlBu',
                   unit_scaling_for_plot=[1.0, 1.0, 1.0],
                   mask_by_sem_limit=None,
                   outputFileName=None, outputCSVFileName=None,
                   closeFigure=False):
    """
    Create a 2D heatmap plot.
    Based on approach formerly in make_2D_histogram()
    
    Parameters
    ----------
    Z : 2D numpy array
        Values to plot as heatmap
    X, Y : 2D numpy arrays
        Meshgrid coordinates
    Z_unc : 2D numpy array or None
        Uncertainty values (for masking)
    titleStr, xlabelStr, ylabelStr, clabelStr : str
        Labels for plot
    colorRange : tuple or None
        (vmin, vmax) for color scale
    cmap : str
        Colormap name
    unit_scaling_for_plot : list of 3 floats
        Scaling factors for [X, Y, Z]
    mask_by_sem_limit : float or None
        If not None, mask points where Z_unc > this value
    outputFileName : str or None
        If not None, save figure to this file
    outputCSVFileName : str or None
        If not None, write Z (raw, unscaled) as a wide-format CSV with X[:,0]
        as the row index and Y[0,:] as column headers. If Z_unc is also
        provided, write it to a parallel file with '_unc' inserted before
        the extension.
    closeFigure : bool
        If True, close figure after creating

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """

    # Apply masking if requested
    if (mask_by_sem_limit is not None) and (Z_unc is not None):
        mask = Z_unc > mask_by_sem_limit
        Z_plot = np.ma.masked_array(Z * unit_scaling_for_plot[2], mask=mask)
    else:
        Z_plot = Z * unit_scaling_for_plot[2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    if colorRange is None:
        pcm = ax.pcolormesh(X * unit_scaling_for_plot[0],
                           Y * unit_scaling_for_plot[1],
                           Z_plot, shading='nearest', cmap=cmap)
        cbar = fig.colorbar(pcm, ax=ax)
    else:
        pcm = ax.pcolormesh(X * unit_scaling_for_plot[0],
                           Y * unit_scaling_for_plot[1],
                           Z_plot, shading='nearest',
                           vmin=colorRange[0] * unit_scaling_for_plot[2],
                           vmax=colorRange[1] * unit_scaling_for_plot[2],
                           cmap=cmap)
        cbar = fig.colorbar(pcm, ax=ax,
                           boundaries=np.linspace(
                               colorRange[0] * unit_scaling_for_plot[2],
                               colorRange[1] * unit_scaling_for_plot[2], 256),
                           ticks=np.linspace(
                               colorRange[0] * unit_scaling_for_plot[2],
                               colorRange[1] * unit_scaling_for_plot[2], 7))
    
    # Labels
    ax.set_xlabel(xlabelStr, fontsize=16)
    ax.set_ylabel(ylabelStr, fontsize=16)
    ax.set_title(titleStr, fontsize=18)
    cbar.set_label(clabelStr, fontsize=14)
    
    plt.tight_layout()
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight', dpi=300)
    
    if closeFigure:
        plt.close(fig)
    else:
        plt.show()

    """
    Should the order be this?
    plt.show()
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')

    if closeFigure:
        print(f'Closing figure {titleStr}')
        plt.close(fig)
    """

    if outputCSVFileName is not None:
        x_vals = X[:, 0]   # unique x centers (raw, unscaled)
        y_vals = Y[0, :]   # unique y centers (raw, unscaled)
        pd.DataFrame(Z, index=x_vals, columns=y_vals).to_csv(outputCSVFileName)
        if Z_unc is not None:
            base, ext = os.path.splitext(outputCSVFileName)
            pd.DataFrame(Z_unc, index=x_vals, columns=y_vals).to_csv(
                base + '_unc' + ext)

    return fig, ax
    

def plot_2Darray_linePlots(Z, X, Y, Z_unc=None,
                           titleStr=None, xlabelStr='X', ylabelStr='Y', clabelStr='Z',
                           colorRange=None, cmap='RdYlBu',
                           unit_scaling_for_plot=[1.0, 1.0, 1.0],
                           mask_by_sem_limit=None,
                           outputFileName=None, closeFigure=False):
    """
    Create a series of line plots of Z vs X for each unique Y value.
    
    Parameters
    ----------
    Z : 2D numpy array
        Values to plot (shape: len(X_unique) x len(Y_unique))
    X, Y : 2D numpy arrays
        Meshgrid coordinates
    Z_unc : 2D numpy array or None
        Uncertainty values for errorbars
    titleStr, xlabelStr, ylabelStr, clabelStr : str
        Labels for plot
    colorRange : tuple or None
        (vmin, vmax) for color scale (used to normalize colormap)
    cmap : str
        Colormap name
    unit_scaling_for_plot : list of 3 floats
        Scaling factors for [X, Y, Z]
    mask_by_sem_limit : float or None
        If not None, mask points where Z_unc > this value
    outputFileName : str or None
        If not None, save figure to this file
    closeFigure : bool
        If True, close figure after creating
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Apply unit scaling
    X_plot = X * unit_scaling_for_plot[0]
    Y_plot = Y * unit_scaling_for_plot[1]
    Z_plot = Z * unit_scaling_for_plot[2]
    if mask_by_sem_limit is not None:
        mask_by_sem_limit = mask_by_sem_limit * unit_scaling_for_plot[2]
    
    if Z_unc is not None:
        Z_unc_plot = Z_unc * unit_scaling_for_plot[2]
    else:
        Z_unc_plot = None
    
    # Get unique Y values (assuming meshgrid structure)
    # Y varies along axis=1 (columns)
    Y_unique = Y_plot[0, :]  # First row contains all unique Y values
    X_unique = X_plot[:, 0]  # First column contains all unique X values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Set up colormap
    cmap_obj = cm.get_cmap(cmap)
    
    # Determine color normalization
    if colorRange is not None:
        vmin = colorRange[0] * unit_scaling_for_plot[2]
        vmax = colorRange[1] * unit_scaling_for_plot[2]
    else:
        vmin = np.nanmin(Y_plot)
        vmax = np.nanmax(Y_plot)
    
    # Normalize Y values for coloring
    if vmax > vmin:
        Y_norm = (Y_unique - vmin) / (vmax - vmin)
    else:
        Y_norm = np.ones_like(Y_unique) * 0.5
    
    # Plot each Y slice
    for j, (y_val, y_norm) in enumerate(zip(Y_unique, Y_norm)):
        # Extract Z values for this Y
        Z_slice = Z_plot[:, j]
        
        # Apply masking if requested
        if mask_by_sem_limit is not None and Z_unc_plot is not None:
            mask = Z_unc_plot[:, j] > mask_by_sem_limit
            Z_slice = np.ma.masked_array(Z_slice, mask=mask)
            if Z_unc_plot is not None:
                Z_unc_slice = np.ma.masked_array(Z_unc_plot[:, j], mask=mask)
        else:
            if Z_unc_plot is not None:
                Z_unc_slice = Z_unc_plot[:, j]
        
        # Get color from colormap
        color = cmap_obj(y_norm)
        
        # Plot with errorbars if uncertainty is provided
        if Z_unc_plot is not None:
            ax.errorbar(X_unique, Z_slice, yerr=Z_unc_slice,
                       label=f'{y_val:.2f}',
                       color=color, marker='o', markersize=8,
                       linestyle='-', linewidth=2.5, capsize=3, alpha=0.8)
        else:
            ax.plot(X_unique, Z_slice,
                   label=f'{y_val:.2f}',
                   color=color, marker='o', markersize=8,
                   linestyle='-', linewidth=2.5, alpha=0.8)
    
    # Labels and formatting
    ax.set_xlabel(xlabelStr, fontsize=16)
    ax.set_ylabel(clabelStr, fontsize=16)
    ax.set_title(titleStr, fontsize=18)
    
    # Add legend (may want to adjust location or use fewer entries for many Y values)
    n_lines = len(Y_unique)
    if n_lines <= 15:
        ax.legend(loc='best', fontsize=10, framealpha=0.9, title = f'{ylabelStr}')
    else:
        # For many lines, show legend outside plot or with fewer entries
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                 fontsize=9, framealpha=0.9, title = f'{ylabelStr}')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight', dpi=300)
    
    if closeFigure:
        plt.close(fig)
    else:
        plt.show()
    
    return fig, ax
    
def revise_datasets(keys_to_modify=["relative_orientation"],
                    pickleFileName1=None, pickleFileName2=None,
                    writePickleOutput =  True, writeOtherOutput = False):
    """
    Load datasets from pickle files, recalculate some properties,
    update the “datasets” variable, save the revised datasets back to the 
    same pickle files (optional, recommended), and re-write CSV, text,
    and Excel files (optional).
    The recalculated properties can be:
      -  angle values using recalculate_angles(), 
          e.g. to revise after the method changed from unsigned to signed angles
      - "maintain_proximity_1" or "maintain_proximity_2", the maintaining-proximity
          behaviors, e.g. to use different ranges.
      - "angular_speed_rad_s_mean" : mean angular speed averaged over both fish,
          e.g. to revise after this property started to be calculated
      - "turning_angle_rad" : change in turning angle (-1.0 * change in
          heading angle (rad)) for each fish,
          e.g. to revise after this property started to be calculated
      - "angular_speed_array_rad_s" : angular speed,
          correcting an offset error (minor)
      - "IBI_properties" : per-fish inter-bout-interval properties via
          get_IBI_properties(); e.g. to add this key to existing pickle files
          without re-analyzing the CSVs. Requires the standard pipeline keys
          (isActive_Fish{k}, radial_position_mm, polar_angle_rad, heading_angle,
          and, for Nfish==2, head_head_distance_mm / closest_distance_mm /
          relative_orientation) to already be present in the loaded datasets.
    
    Parameters
    ----------
    keys_to_modify : list of str
        List of keys to recalculate. Must be subset of:
        ["relative_orientation", "bend_angle", "heading_angle",
         "maintain_proximity_1", "maintain_proximity_2", "angular_speed_rad_s_mean"]
        Default: ["relative_orientation"]
    pickleFileName1 : str or None
        Full path to position data pickle file. If None, will prompt user.
    pickleFileName2 : str or None
        Full path to datasets pickle file. If None, will prompt user.
    writePickleOutput : if True, write re-calculated output to Pickle files 
        (Recommended!)
    writeOtherOutput : if True, write re-calculated output to CSV, txt, 
        Excel files. Asks user for a new folder to write to.
    
    Returns
    -------
    Overwrites the original pickle files with updated data.
    
    Notes
    -----
    This function:
    1. Loads data from two pickle files (position data and datasets)
    2. Calls recalculate_angles() to update specified angle calculations
    3. Saves the updated datasets back to the original pickle files
    
    The position data pickle file is not modified (only read).
    The datasets pickle file is overwritten with updated values.
    
    Example usage:
    >>> from IO_toolkit import revise_datasets
    >>> # Recalculate relative orientation only (default, but I'll write 
          the input anyway)
    >>> revise_datasets(keys_to_modify=["relative_orientation", "bend_angle"])
    """
    
    # Import here to avoid circular import
    from behavior_identification import recalculate_angles, get_maintain_proximity_frames, \
        get_IBI_properties
    from behavior_identification_single import get_mean_speed, get_fish_delta_heading_angle

    print("\n" + "="*70)
    print("REVISING DATASETS: Recalculating angles in pickle files")
    print("="*70)
    print(f"\nKeys to modify: {keys_to_modify}\n")
    
    # Load from pickle files
    print("Loading data from pickle files...")
    all_position_data, variable_tuple = load_and_assign_from_pickle(
        pickleFileName1=pickleFileName1, 
        pickleFileName2=pickleFileName2
    )
    (datasets, CSVcolumns, expt_config, params, N_datasets, 
     Nfish, basePath, dataPath, subGroupName) = variable_tuple
    
    print(f"Loaded {N_datasets} datasets with {Nfish} fish each.")
    
    # Store the original pickle file path for default save location
    # We need to track what file was loaded
    # If pickleFileName2 was provided, use that
    # Otherwise, we need to prompt and remember the choice
    if pickleFileName2 is not None:
        original_pickle_path = pickleFileName2
        original_pickle_dir = os.path.dirname(pickleFileName2)
        original_pickle_filename = os.path.basename(pickleFileName2)
    else:
        # Files were selected via dialog, we don't have the path stored
        # We'll need to prompt for it below
        original_pickle_path = None
        original_pickle_dir = None
        original_pickle_filename = None
    
    # Recalculate angles if any of the "keys_to_modify" are in this set:
    angle_keys = ["relative_orientation", "bend_angle", "heading_angle"]
    angle_keys_to_modify = [key for key in keys_to_modify if key in angle_keys]
    if angle_keys_to_modify:
        datasets = recalculate_angles(all_position_data, datasets, CSVcolumns, 
                                  keys_to_modify=angle_keys_to_modify)

    # re-calculate  angular speed:
    if "angular_speed_array_rad_s" in keys_to_modify:
        from behavior_identification_single import get_fish_angular_speeds
        print('Revising angular speed array.')
        for j in range(N_datasets):
            datasets[j]["angular_speed_array_rad_s"] = \
                get_fish_angular_speeds(datasets[j])

    # re-calculate mean angular speed:
    if "angular_speed_rad_s_mean" in keys_to_modify:
        print('Revising mean angular speed.')
        for j in range(N_datasets):
            angular_speed_mean_all, _ = \
                get_mean_speed(datasets[j]["angular_speed_array_rad_s"], 
                               None, datasets[j]["bad_bodyTrack_frames"]["raw_frames"])
            # average over fish, since ID is unreliable
            datasets[j]["angular_speed_rad_s_mean"] = np.mean(angular_speed_mean_all)

    # re-calculate change in heading angle:
    if "turning_angle_rad" in keys_to_modify:
        print('Revising turning angle.')
        for j in range(N_datasets):
            datasets[j]["turning_angle_rad"] = \
                -1.0*get_fish_delta_heading_angle(datasets[j], frame_diff = (-1, 1))

    # re-calculate inter-bout-interval (IBI) properties (per-fish IBI means of
    # position, heading, turning angle; pair quantities if Nfish==2):
    if "IBI_properties" in keys_to_modify:
        datasets = get_IBI_properties(datasets)

    # Recalculate other behaviors; need to specify and write code for each
    for prox_num in (1, 2):
        prox_key = f"maintain_proximity_{prox_num}"
        if prox_key in keys_to_modify:
            print(f'\nRecalculating {prox_key}.')
            params = update_parameters(prox_key, params)
            for j in range(N_datasets):
                print(f'Revising {prox_key} for Dataset: ', datasets[j]["dataset_name"])
                pair_behavior_frames = get_maintain_proximity_frames(
                                              all_position_data[j],
                                              datasets[j],
                                              CSVcolumns, params,
                                              threshold_mm=params[f"proximity_{prox_num}_threshold_mm"],
                                              distance_measure=params[f"proximity_{prox_num}_distance_measure"])
                tuple_of_frames_to_reject = (datasets[j]["edge_frames"]["raw_frames"],
                      datasets[j]["bad_bodyTrack_frames"]["raw_frames"])
                datasets[j][prox_key] = make_frames_dictionary(pair_behavior_frames,
                                               tuple_of_frames_to_reject,
                                               behavior_name = prox_key,
                                               Nframes=datasets[j]['Nframes'])

    if writePickleOutput:
        # Save the updated datasets back to pickle
        print("\n" + "-"*70)
        print("Saving updated datasets to pickle file...")
        
        # Prepare the dictionary to save (same structure as original)
        variables_dict = {
            'datasets': datasets,
            'CSVcolumns': CSVcolumns,
            'expt_config': expt_config,
            'params': params,
            'basePath': basePath,
            'dataPath': dataPath,
            'subGroupName': subGroupName
        }
        
        # Determine save location
        if original_pickle_path is None:
            # User was prompted during load, need to get the file path
            print("\n\nWe need to use a dialog box to get the original file path.")
            print("Select the ORIGINAL datasets pickle file (the one you just loaded).")
            print("This will be used as the default save location.")
            print("Note that the dialog box may be hidden behind other windows.")
            
            root = tk.Tk()
            root.withdraw()
            original_pickle_path = filedialog.askopenfilename(
                title="Select original datasets pickle file",
                filetypes=[("pickle files", "*.pickle")]
            )
            
            if not original_pickle_path:
                print("\nNo file selected. Aborting without saving.")
                return
            
            original_pickle_dir = os.path.dirname(original_pickle_path)
            original_pickle_filename = os.path.basename(original_pickle_path)
        
        # Prompt for output pickle filename
        print(f"\nOriginal datasets file: {original_pickle_path}")
        print("\nEnter output filename (press Enter to overwrite original file):")
        print(f"  Default: {original_pickle_filename}")
        
        user_filename = input("  Output filename: ").strip()
        
        if user_filename == '':
            # Use original filename (overwrite)
            save_pickle_filename = original_pickle_filename
            save_pickle_path = original_pickle_path
            overwriting = True
        else:
            # User provided a new filename
            save_pickle_filename = user_filename
            # Make sure it ends with .pickle
            if not save_pickle_filename.endswith('.pickle'):
                save_pickle_filename += '.pickle'
            save_pickle_path = os.path.join(original_pickle_dir, save_pickle_filename)
            overwriting = (save_pickle_path == original_pickle_path)
        
        # Confirm action
        if overwriting:
            print(f"\nWill OVERWRITE: {save_pickle_path}")
            confirm = input("Type 'yes' to confirm overwrite: ")
        else:
            print(f"\nWill save to NEW file: {save_pickle_path}")
            confirm = input("Type 'yes' to confirm: ")
        
        if confirm.lower() != 'yes':
            print("\nAborted. No files were modified.")
            return
        
        # Use write_pickle_file to save
        write_pickle_file(variables_dict, dataPath=original_pickle_dir, 
                         outputFolderName='', 
                         pickleFileName=save_pickle_filename)
        
        print("\n" + "="*70)
        print("REVISION COMPLETE")
        print("="*70)
        print(f"\nSaved to: {save_pickle_path}")
        if overwriting:
            print("  (Original file was overwritten)")
        else:
            print("  (New file created, original unchanged)")
        print(f"Modified keys: {keys_to_modify}")
        print(f"Number of datasets updated: {N_datasets}")
        print("\nThe position data pickle file was not modified (read only).")
        print("="*70 + "\n")
    
    if writeOtherOutput:
        # Write the output files (CSV, Excel, and YAML (parameters))
        print(f"\n\nCurrent output path: {dataPath}.")
        print("STRONGLY RECOMMENDED: Input a new output path, to avoid overwriting!")
        newOutputPath = input('  Enter the new output path: ')
        if newOutputPath == '':
            newOutputPathSure = input('Same path; Are you sure?! (y/n): ')
            if newOutputPathSure.lower() != 'y':
                print('Canceling output writing; re-run if you want.')
                return
        else:
            dataPath = newOutputPath
        write_CSV_Excel_YAML(expt_config, params, dataPath, datasets)
        
    return datasets


def update_parameters(keyName, params):
    """
    Ask the user for updated parameters; presumably called if we're 
    re-calculating behaviors.
    Hard-code various options; add to this to enable more behavior revisions

    Parameters
    ----------
    keyName : string
        behavior key to be updated (e.g. "maintain_proximity_1" or "maintain_proximity_2").
    params : dict
        analysis parameters

    Returns
    -------
    params

    """
    
    if keyName in ('maintain_proximity_1', 'maintain_proximity_2'):
        prox_num = keyName[-1]  # '1' or '2'
        thresh_key = f'proximity_{prox_num}_threshold_mm'
        dist_key = f'proximity_{prox_num}_distance_measure'

        inputText = f"Enter new min and max proximity thresholds for proximity_{prox_num} *separated by a space*;\n" + \
            f"   Blank (Enter) to keep {params[thresh_key][0]} {params[thresh_key][1]}: "
        inputStr = input(inputText)
        if inputStr != '':
            params[thresh_key] = [float(x) for x in inputStr.split()]

        inputText = f"Enter the distance measure for proximity_{prox_num}: 'closest' or 'head_to_head'; blank for {params[dist_key]}: "
        inputStr = input(inputText)
        if inputStr != '':
            params[dist_key] = inputStr
        if (params[dist_key] != 'closest') and (params[dist_key] != 'head_to_head'):
            print(f'INVALID OPTION for {dist_key}. Using closest distance.')
            params[dist_key] = 'closest'

        inputText = "Enter the maximum gap in matching criterion to allow (s). \n" + \
            f"blank for {params['max_motion_gap_s']:.3f}: "
        inputStr = input(inputText)
        if inputStr != '':
            params['max_motion_gap_s'] = float(inputStr)

        inputText = "Enter the min duration that matching criteria must be met (s). \n" + \
            f"blank for {params['min_proximity_duration_s']:.3f}: "
        inputStr = input(inputText)
        if inputStr != '':
            params['min_proximity_duration_s'] = float(inputStr)

    else:
        raise ValueError('Invalid behavior key for update_parameters.')
        
    print('Parameters updated.')

    return params


def get_plot_and_CSV_filenames(s, outputFileNameBase, outputFileNameExt, 
                               writeCSVs):
    """
    Simple function to make filenames for output plot and CSV file
    
    Returns
        outputFileName = outputFileNameBase + s + '.' + outputFileNameExt
             None if either outputFileNameBase or outputFileNameExt is None
        outputCSVFileName = outputFileNameBase + s + '.csv'
             None if writeCSVs is False
    """
    if (outputFileNameBase is not None) and (outputFileNameExt is not None):
        outputFileName = outputFileNameBase + s + '.' + outputFileNameExt
    else:
        outputFileName = None
    # CSV file
    
    if (writeCSVs is True) and (outputFileNameBase is not None):
        outputCSVFileName = outputFileNameBase + s + '.csv'
    else:
        outputCSVFileName = None
    return outputFileName, outputCSVFileName