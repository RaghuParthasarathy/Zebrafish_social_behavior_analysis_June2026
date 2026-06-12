# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# behaviors_main.py
"""
#----------------------------------------------------------------------------
# Raghuveer Parthasarathy (2023-2025)
# Version ='2.0' Raghuveer Parthasarathy -- begun May 2023; see notes.
# Last modified: Raghuveer Parthasarathy, Dec. 31, 2025
# ---------------------------------------------------1------------------------
"""

import os
import numpy as np
import yaml
from time import perf_counter
from toolkit import get_Nfish, repair_heading_angles, \
        repair_head_positions, repair_double_length_fish, \
        relink_fish_ids_all_datasets, calc_good_tracking_spans, \
        make_frames_dictionary, get_edgeRejection_frames_dictionary, \
        get_badTracking_frames_dictionary, timeShift
from IO_toolkit import get_basePath, get_loading_option, load_and_assign_from_pickle, \
        load_expt_config, load_analysis_parameters, \
        get_output_pickleFileNames, \
        check_analysis_parameters, set_outputFile_params, \
        get_CSV_filenames, load_all_position_data, \
        write_CSV_Excel_YAML, write_pickle_file
from behavior_identification_single import get_single_fish_characterizations, \
        get_coord_characterizations, get_fish_lengths
from behavior_identification import extract_pair_behaviors, \
    get_basic_two_fish_characterizations, make_rel_orient_rank_key, \
    get_IBI_properties


# ---------------------------------------------------------------------------


def main():
    """
    Main function for calling data reading functions, 
    basic analysis functions, and behavior analysis functions for all 
    CSV files in a set.

    Will read all CSV files or a previously written pickle file. 
    Writes a pickle file containing all trajectory and analysis outputs (“datasets” variable), and other variables – optional but strongly recommended.
    
    Returns:
        datasets : list of dictionaries, one for each dataset. 
                    datasets[j] contains information for dataset j.
        all_position_data : list of numpy arrays with position information for
                    each dataset. Nframes x Ncolumns x Nfish
        CSVcolumns : dictionary of CSV column contents
        

    """
    # Timer
    t0 = perf_counter()
    
    cwd = os.getcwd() # Note the current working directory

    # for Raghu -- troubleshooting. Don't modify this.
    globalBasePath = r'C:\Users\Raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs'
    global_config_file = 'all_expt_configs.yaml'
    # expt_config = load_expt_config(basePath, config_file)
    
    loading_option = get_loading_option()
    print(f"\nSelected loading option: {loading_option}\n")
    
    if loading_option == 'load_from_CSV':
        # Read from CSV files
        
        # The main folder containing configuration and parameter files.
        basePath = get_basePath()
    
        # Load experiment configuration file
        config_file = 'expt_config.yaml'
        expt_config = load_expt_config(basePath, config_file)
        
        # Get experiment name
        if (expt_config['expt_name'] is None) or (expt_config['expt_name'] == ''):
            expt_config['expt_name'] = input('Enter the experiment name, to append to outputs: ')
        else:
            expt_name_prompt = input(f'Enter the experiment name, or Press Enter for {expt_config["expt_name"]}: ')
            if expt_name_prompt != '':
                expt_config["expt_name"] = expt_name_prompt
                
        # Get CSV column info from configuration file
        CSVinfo_file = 'CSVcolumns.yaml'
        CSVinfo_file_full = os.path.join(basePath, CSVinfo_file)
        # Note that we already checked if this exists
        with open(CSVinfo_file_full, 'r') as f:
            all_CSV = yaml.safe_load(f)
        CSVcolumns = all_CSV['CSVcolumns']
    
        # Get folder containing CSV files, and all "results" CSV filenames
        # Also get subgroup name
        # Note that dataPath is the path containing CSVs, which 
        # may be a subgroup path
        dataPath, allCSVfileNames, subGroupName = \
            get_CSV_filenames(basePath, expt_config, startString="results")
        if len(allCSVfileNames)==0:
            raise ValueError("Error: Zero CSV files found! Check folder structure.")
        print(f'\n\n All {len(allCSVfileNames)} CSV files starting with "results": ')
        print(allCSVfileNames)
        
        # Number of datasets
        N_datasets = len(allCSVfileNames)

        # Get behavior analysis parameter info from configuration file
        params_file = 'analysis_parameters.yaml'
        params = load_analysis_parameters(basePath, params_file)
        params = check_analysis_parameters(params)
        # Fill in keys in params corresponding to output folders, Excel file names
        params = set_outputFile_params(params, expt_config, subGroupName)
    
        # Output pickle file name
        pickleFileNames= get_output_pickleFileNames(expt_config['expt_name'], 
                                                   subGroupName)
            
        #%% Load all position data; repair
        # load all position data and determine general parameters 
        # such as fps and scale
        all_position_data, datasets = \
            load_all_position_data(allCSVfileNames, expt_config, 
                                          CSVcolumns, dataPath, params)
        Nfish = get_Nfish(datasets)
        
        # Repair (recalculate) head positions, based on indexes 1-3 
        #(i.e. the 2nd, third, and fourth positions), linearly interpolating
        all_position_data = repair_head_positions(all_position_data, CSVcolumns)
    
        # Repair (recalculate) angles
        datasets = repair_heading_angles(all_position_data, datasets, CSVcolumns)

        # Calculate fish lengths (necessary for double-length repair)
        for j in range(len(datasets)):
            # Get the length of each fish in each frame (sum of all segments)
            # Nframes x Nfish array
            datasets[j]["fish_length_array_mm"] = \
                get_fish_lengths(all_position_data[j], 
                                 datasets[j]["image_scale"], CSVcolumns)

        # Repair double-length fish
        if Nfish==2:
            all_position_data, datasets = \
                repair_double_length_fish(all_position_data, datasets, CSVcolumns, 
                                          lengthFactor = [1.5, 2.5], tol=0.001)
            # Re-calculate fish lengths 
            for j in range(len(datasets)):
                datasets[j]["fish_length_array_mm"] = \
                    get_fish_lengths(all_position_data[j], 
                                     datasets[j]["image_scale"], CSVcolumns)


    elif loading_option == 'load_from_pickle':
        # Load positions, datasets dictionary, etc., from pickle files.  
        # May contain analysis, but this will be redone
        all_position_data, variable_tuple = load_and_assign_from_pickle()
        (datasets, CSVcolumns, expt_config, params, N_datasets, Nfish,
         basePath, dataPath, subGroupName) = variable_tuple
        
        # allow revision of base path
        print(f'Base Path loaded: {basePath}')
        new_basePath = input(f'Enter the new base path; default {basePath}: ')
        if new_basePath != '':
            basePath = new_basePath
            if subGroupName is not None:
                dataPath = os.path.join(basePath, subGroupName)
            else:
                dataPath = basePath
        
        # allow revision of experiment name
        new_expt_name = input(f'Enter the experiment name; default {expt_config["expt_name"]} (unchanged): ')
        if new_expt_name != '':
            expt_config['expt_name'] = new_expt_name
        # allow revision of output subfolder
        new_output_subFolder = input(f'Enter the output subfolder name; default {params["output_subFolder"]} (unchanged, will overwrite!): ')
        if new_output_subFolder != '':
            params["output_subFolder"] = new_output_subFolder
        
        # Output pickle file name
        pickleFileNames = get_output_pickleFileNames(expt_config['expt_name'], 
                                                   subGroupName)
    else:
        raise ValueError("Error: Bad Loading Option.")
        
    t_load = perf_counter()
    print(f'\n\nData loaded, repaired; elapsed time: {t_load - t0:.2f} seconds.\n')

    #%% Identify bad tracking
            
    # Identify bad-tracking frames for each dataset
    # Call get_bad_headTrack_frames and get_bad_bodyTrack_frames
    # for each datasets[j] and put results in a 
    # dictionary that includes durations of events, etc.
    datasets = get_badTracking_frames_dictionary(all_position_data, datasets, 
                                                 params, CSVcolumns, tol=0.001)
    
    #%% Re-link all fish based on whole-body distance minimizing
    print('\nRe-linking...')
    all_position_data, datasets = relink_fish_ids_all_datasets(all_position_data,
                                                     datasets, CSVcolumns)
    # Re-calculate fish lengths 
    for j in range(len(datasets)):
        datasets[j]["fish_length_array_mm"] = \
            get_fish_lengths(all_position_data[j], 
                             datasets[j]["image_scale"], CSVcolumns)
        
    #%% Time-shift one of the fish
    time_shift_fish_idx = None # set to None to avoid shifting
    if time_shift_fish_idx is not None:
        print('\n\n***************************')
        caution_check = input(f'\nARE YOU SURE you want to time-shift fish {time_shift_fish_idx}? (y/n): ')
        if caution_check=='y':
            valid_idx = (np.isin(time_shift_fish_idx, np.arange(0, Nfish))) and \
                        (type(time_shift_fish_idx)==int)
            if valid_idx==True:
                print(f'\n\n** Time-shifting fish {time_shift_fish_idx} by half the duration**')
                for j in range(len(datasets)):
                    frameshift = np.floor(datasets[j]["Nframes"]/2.0).astype(int)
                    all_position_data[j], datasets[j] = \
                        timeShift(all_position_data[j], datasets[j], 
                                  CSVcolumns, fishIdx = time_shift_fish_idx, 
                                  frameShift = frameshift)
                    # re-calculate fish lengths. (Could just shift...)
                    datasets[j]["fish_length_array_mm"] = \
                        get_fish_lengths(all_position_data[j], 
                                         datasets[j]["image_scale"], CSVcolumns)
                # Re-calculate bad tracking frames
                # This could be avoided by shifting these, but this is easier.
                datasets = get_badTracking_frames_dictionary(all_position_data, 
                                                             datasets, 
                                                             params, CSVcolumns, 
                                                             tol=0.001)
            else:
                print('Invalid index; *NOT* flipping')
                input('Press enter to indicate acknowlegement: ')
                

    #%% Calculate statistics of continuous spans of good tracking
    for j in range(len(datasets)):
        datasets[j]["good_tracking_spans"] = calc_good_tracking_spans(datasets[j], 
                                                                      verbose = False)

    #%% Identify close-to-edge frames for each dataset, for rejecting behaviors
    # Call get_edge_frames for each datasets[j] and put results in a 
    # dictionary that includes durations of edge events, etc.
    datasets = get_edgeRejection_frames_dictionary(datasets, params, 
                                          expt_config['arena_radius_mm'])
    
    t_assessTracks = perf_counter()
    print(f'\n\nBad tracking identified, re-linked; elapsed time: {t_assessTracks - t0:.2f} seconds.\n')
    
    #%% Analysis: Basic characterizations
    
    # For each dataset, get simple coordinate characterizations
    # (polar coordinates, radial alignment) 
    datasets = get_coord_characterizations(all_position_data, 
                                           datasets, CSVcolumns, expt_config, params)
        
    #%% Analysis: single fish characterizations

    # For each dataset, characterizations that involve single fish
    # (e.g. fish length, bending, speed)
    datasets = get_single_fish_characterizations(all_position_data, 
                                                 datasets, CSVcolumns,
                                                 expt_config, params)

    t_single = perf_counter()
    print(f'\n\nSingle fish characterizations done; elapsed time: {t_single - t0:.2f} seconds.\n')

    #%% Analysis: multi-fish characterizations

    # For each dataset, perform “basic” two-fish characterizations 
    # such as inter-fish distance, if Nfish > 1. 
    if Nfish==2:
        datasets = get_basic_two_fish_characterizations(all_position_data, datasets,
                                                     CSVcolumns, expt_config, params)

    # For each dataset, compute inter-bout-interval (IBI) properties
    # (per-fish IBI means of position, heading, turning angle; pair quantities
    # if Nfish==2). Stored in datasets[j]["IBI_properties"].
    datasets = get_IBI_properties(datasets)

    # For each dataset, identify social behaviors
    if Nfish > 1:
        behavior_keys = ['perp_noneSee', 'perp_oneSees', 
                         'perp_bothSee', 'perp_larger_fish_sees',
                         'perp_smaller_fish_sees', 
                         'contact_any', 'contact_head_head', 'contact_head_body', 
                         'contact_larger_fish_head', 'contact_smaller_fish_head',
                         'contact_tail_tail',
                         'contact_inferred', 'tail_rubbing_AP', 'tail_rubbing_P',
                         'maintain_proximity_1', 'maintain_proximity_2',
                         'approaching_Fish0', 'approaching_Fish1',
                         'approaching_any', 'approaching_all',
                         'fleeing_Fish0', 'fleeing_Fish1',
                         'fleeing_any', 'fleeing_all', ]
        for j in range(N_datasets):
            
            print('Identifying two-fish behaviors for Dataset: ', 
                  datasets[j]["dataset_name"])
            
            # Initialize empty dictionary
            pair_behavior_frames = {key: np.array([], dtype=int) for key in behavior_keys}
            
            pair_behavior_frames = extract_pair_behaviors(pair_behavior_frames,
                                                          all_position_data[j],
                                                          datasets[j], 
                                                          params, CSVcolumns)

            # All frames with social (two fish) behaviors (Combine all arrays)
            all_social_frames = np.unique(np.concatenate(
                                        list(pair_behavior_frames.values())))
            behavior_keys.append('anyPairBehavior')
            pair_behavior_frames['anyPairBehavior'] = all_social_frames

            # For each behavior, a dictionary containing frames, frames with
            # "bad" elements removed, and a 2xN array of initial frames 
            # and durations
            # For most behaviors, will exclude bad tracking frames,
            # but contact_any (because of contact_inferred) will keep these
            for b_key in behavior_keys:
                keys_to_keep_badTrFrames = ['contact_any', 'contact_inferred']
                if b_key in keys_to_keep_badTrFrames:
                    tuple_of_frames_to_reject = (datasets[j]["edge_frames"]["raw_frames"],)
                else :
                    tuple_of_frames_to_reject = (datasets[j]["edge_frames"]["raw_frames"],
                     datasets[j]["bad_bodyTrack_frames"]["raw_frames"])
                datasets[j][b_key] = make_frames_dictionary(pair_behavior_frames[b_key],
                                               tuple_of_frames_to_reject,
                                               behavior_name = b_key,
                                               Nframes=datasets[j]['Nframes'])

            # Behaviors organized by relative orientation rank, not fish ID
            # Note that this doesn't introduce new frames
            # As written, need to put after datasets[j] behavior dictionary 
            # creation, so redundant dictionary-making code
            rel_orient_keys = ['approaching_Fish', 'fleeing_Fish', 
                               'isActive_Fish', 'isBending_Fish', 'isMoving_Fish',
                               'Cbend_Fish', 'Jbend_Fish', 'Rbend_Fish']
            tuple_of_frames_to_reject = (datasets[j]["edge_frames"]["raw_frames"],
                                         datasets[j]["bad_bodyTrack_frames"]["raw_frames"])
            for r_key in rel_orient_keys:
                make_rel_orient_rank_key(datasets[j], r_key)
                # Note that this makes the "raw frames" key. I'm going to destroy
                # this to make the "proper" dictionary!
                for jj in range(Nfish):
                    rel_orient_key_final = f'{r_key}_lowRelOrient{jj}'
                    # Make frames dictionary
                    datasets[j][rel_orient_key_final] = make_frames_dictionary(
                                                   datasets[j][rel_orient_key_final]['raw_frames'],
                                                   tuple_of_frames_to_reject,
                                                   behavior_name = rel_orient_key_final,
                                                   Nframes=datasets[j]['Nframes'])
            
    t_pair = perf_counter()
    print(f'\n\nPair characterizations done; elapsed time: {t_pair - t0:.2f} seconds.\n')
    
            
    #%% Outputs

    # Write pickle files containing position info and {datasets and other variables}
    if pickleFileNames[2].lower() != 'none':
        if loading_option == 'load_from_CSV':
            # Loaded from CSV, so create pickle file of position information
            # Dictionary to save, for position pickle file, in dataPath
            variables_dict = {'all_position_data': all_position_data}
            write_pickle_file(variables_dict, dataPath = dataPath, 
                              outputFolderName = '', 
                              pickleFileName = pickleFileNames[0])
        # For any loading option, save the calculated info
        # Dictionary to save, for datasets pickle file
        variables_dict = {
            'datasets': datasets,
            'CSVcolumns': CSVcolumns,
            'expt_config': expt_config,
            'params': params,
            'basePath': basePath,
            'dataPath': dataPath, 
            'subGroupName': subGroupName
        }
        # list_for_pickle = [datasets, CSVcolumns, expt_config, params]
        write_pickle_file(variables_dict, dataPath = dataPath, 
                          outputFolderName = params['output_subFolder'], 
                          pickleFileName = pickleFileNames[1])
    
    # Write the output files (CSV, Excel, and YAML (parameters))
    write_CSV_Excel_YAML(expt_config, params, dataPath, datasets)
    
    # Return to original directory
    os.chdir(cwd)

    t_write = perf_counter()
    print(f'\n\nOutputs written; elapsed time: {t_write - t0:.2f} seconds.\n')

    
    return datasets, all_position_data, CSVcolumns
    
if __name__ == '__main__':
    datasets, all_position_data, CSVcolumns = main()
