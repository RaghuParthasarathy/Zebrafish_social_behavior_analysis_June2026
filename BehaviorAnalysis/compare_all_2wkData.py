# -*- coding: utf-8 -*-
# compare_all_2wkData.py

"""
Author:   Raghuveer Parthasarathy
Created on Wed Oct 15 08:59:05 2025
Last modified April 15, 2026 -- Raghuveer Parthasarathy

Description
-----------

Load pickle files of two week old zebrafish behavior data. 
Make plots, for comparison.

Datasets and colors:
    Pairs, Light (color darkorange)
    Pairs, Light, Time-shifted Fish0 (color saddlebrown)
    Pairs, Dark (color cornflowerblue)
    Pairs, Dark, Time-shifted Fish0 (color darkblue)
    Single, Light (gold)
    Single, Dark (slategrey)


Instructions:

Modify the expBaseStrm et., for each dataset of interest.

"""

import os
import numpy as np
from IO_toolkit import load_and_assign_from_pickle, \
    combine_images_to_tiff, plot_2D_heatmap
from behavior_plots import (make_pair_fish_plots, make_bending_angle_plots,
    make_pair_1D_v_distance_plots, make_turning_angle_plots,
    make_relative_orientation_plots, make_single_fish_plots,
    slice_2D_histogram, make_2D_histogram)
from toolkit import get_fps 
from behavior_correlations import plot_behaviorCorrelation,  \
    calc_corr_asymm, plot_corr_asymm, calcDeltaFramesEvents, bin_deltaFrames, calc_pAB, calcBehavCorrAllSets

#%%

def load_expt_data(pickleFileName1, pickleFileName2, exptName, 
                           plot_color, all_expts):
    """
    Load experimental data from pickle files. Add to the all_expts dictionary
    """
    
    all_position_data, variable_tuple = load_and_assign_from_pickle(pickleFileName1, 
                                                                    pickleFileName2)
    (datasets, CSVcolumns, expt_config, params, N_datasets, Nfish,
             basePath, dataPath, subGroupName) = variable_tuple
    
    
    all_expts[exptName] = {
        "datasets" : variable_tuple[0],
        "all_position_data" : all_position_data.copy(),
        "CSVcolumns" : variable_tuple[1],
        "expt_config" : variable_tuple[2],
        "params" : variable_tuple[3],
        "N_datasets" : variable_tuple[4],
        "Nfish" : variable_tuple[5],
        "basePath" : variable_tuple[6],
        "dataPath" : variable_tuple[7],
        "subGroupName" : variable_tuple[8],
        "plot_color" : plot_color
        }
    
    del all_position_data
    del variable_tuple
    
    return all_expts


#%%

def _resolve_pickle_path(directory, desired_filename):
    """
    Return os.path.join(directory, desired_filename) if the file exists.
    Otherwise, search directory for a .pickle file whose name is the longest
    prefix of desired_filename, to handle OS path-length truncation.
    Raises FileNotFoundError if no suitable file is found.
    """
    full_path = os.path.join(directory, desired_filename)
    if os.path.isfile(full_path):
        return full_path

    # File not found; search for longest-prefix match (truncated filename)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')

    desired_stem = os.path.splitext(desired_filename)[0]
    best_match = None
    best_len = 0
    for fname in os.listdir(directory):
        if not fname.endswith('.pickle'):
            continue
        candidate_stem = os.path.splitext(fname)[0]
        if desired_stem.startswith(candidate_stem) and len(candidate_stem) > best_len:
            best_match = fname
            best_len = len(candidate_stem)

    if best_match is None:
        raise FileNotFoundError(
            f"No .pickle file found matching '{desired_filename}' in: {directory}")

    print(f"Note: using truncated filename '{best_match}' "
          f"in place of '{desired_filename}'")
    return os.path.join(directory, best_match)


def get_pickle_filenames(expBaseStr, cond_str, CSVPath, parentPath):
    """
    Construct and return (pickleFileName1, pickleFileName2) for an experiment.

    AnalysisPath is derived as f'{expBaseStr}_{cond_str}_Analysis'.
    If the expected filename does not exist (e.g. due to OS path-length
    truncation), the folder is searched for the longest-prefix match.

    Parameters
    ----------
    expBaseStr : str
        Base string for file paths and names.
    cond_str : str
        Condition string (e.g. 'Light_Cond_2').
    CSVPath : str
        Subfolder under parentPath containing the data.
    parentPath : str
        Root path for all data.

    Returns
    -------
    pickleFileName1 : str
        Path to the position data pickle file.
    pickleFileName2 : str
        Path to the datasets pickle file.
    """
    AnalysisPath = f'{expBaseStr}_{cond_str}_Analysis'

    dir1 = os.path.join(parentPath, CSVPath, cond_str)
    fname1 = f'{expBaseStr}_{cond_str}_positionData.pickle'
    pickleFileName1 = _resolve_pickle_path(dir1, fname1)

    dir2 = os.path.join(parentPath, CSVPath, cond_str, AnalysisPath)
    fname2 = f'{expBaseStr}_{cond_str}_datasets.pickle'
    pickleFileName2 = _resolve_pickle_path(dir2, fname2)

    return pickleFileName1, pickleFileName2


def load_all_expt_data(expt_list, parentPath):
    """
    Load all experimental data from a list of experiment parameter dicts.

    Parameters
    ----------
    expt_list : list of dict
        Each dict must have keys: expBaseStr, cond_str, exptName,
        plot_color, CSVPath.
    parentPath : str
        Root path for all data.

    Returns
    -------
    all_expts : dict
        Dictionary of loaded experiment data, keyed by exptName.
    """
    all_expts = {}
    for expt in expt_list:
        pickleFileName1, pickleFileName2 = get_pickle_filenames(
            expt['expBaseStr'], expt['cond_str'], expt['CSVPath'], parentPath)
        all_expts = load_expt_data(pickleFileName1, pickleFileName2,
                                   expt['exptName'], expt['plot_color'], all_expts)
    return all_expts


if __name__ == '__main__':

    #%%

    parentPath = r'C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs'


    #%% Experiment list

    _PAIRS_PATH   = r'2 week old - Sept2025 control pairs in dark vs light New Tracking'
    _SINGLES_PATH = r'2 week old - Sept2025 control single in dark vs light New Tracking'

    expt_list = [
        {'expBaseStr': 'TwoWk_Sept2025',
         'cond_str':   'Light_Cond_2',
         'exptName':   'TwoWk_Light',
         'plot_color': 'darkorange',
         'CSVPath':    _PAIRS_PATH},
        {'expBaseStr': 'TwoWk_Sept2025_TS0',
         'cond_str':   'Light_Cond_2',
         'exptName':   'TwoWk_Light_TIMESHIFT0',
         'plot_color': 'saddlebrown',
         'CSVPath':    _PAIRS_PATH},
        {'expBaseStr': 'TwoWk_Sept2025',
         'cond_str':   'Dark_Cond_1',
         'exptName':   'TwoWk_Dark',
         'plot_color': 'cornflowerblue',
         'CSVPath':    _PAIRS_PATH},
        {'expBaseStr': 'TwoWk_Sept2025_TS0',
         'cond_str':   'Dark_Cond_1',
         'exptName':   'TwoWk_Dark_TIMESHIFT0',
         'plot_color': 'darkblue',
         'CSVPath':    _PAIRS_PATH},
        {'expBaseStr': 'TwoWkSingle',
         'cond_str':   'Light_Cond_2',
         'exptName':   'TwoWk_Light_Single',
         'plot_color': 'gold',
         'CSVPath':    _SINGLES_PATH},
        {'expBaseStr': 'TwoWkSingle',
         'cond_str':   'Dark_Cond_1',
         'exptName':   'TwoWk_Dark_Single',
         'plot_color': 'slategrey',
         'CSVPath':    _SINGLES_PATH},
    ]

    all_expts = load_all_expt_data(expt_list, parentPath)

    #%% Make single fish plots (a lot!)

    makeSingleFishPlots = False
    if makeSingleFishPlots:
    
        closeFigures = True
        if closeFigures:
            print('Single fish plots: Closing Figure Windows.')
    
        for exptName in all_expts.keys():
            make_single_fish_plots(all_expts[exptName]['datasets'], 
                               exptName = exptName,
                               color = all_expts[exptName]['plot_color'], 
                               outputFileNameBase = f'JUNK{exptName} single_properties',
                               outputFileNameExt = 'png',
                               closeFigures = closeFigures,
                               writeCSVs = False)

    #%% Make all pair plots (a lot!)
    # Only for Nfish ==2

    closeFigures = True
    if closeFigures:
        print('Pair plots: Closing Figure Windows.')

    distance_type =  'head_head_distance' # or head_head_distance
    if distance_type == 'closest_distance':
        shortDistanceStr = 'ClDist'
    elif distance_type == 'head_head_distance':
        shortDistanceStr = 'HHDist'
    else:
        raise ValueError('Invalide distance type')

    for exptName in all_expts.keys():
        if all_expts[exptName]['Nfish'] == 2:

            """
            make_pair_fish_plots(all_expts[exptName]['datasets'], 
                                 exptName = exptName,
                                 distance_type = distance_type,
                                 color = all_expts[exptName]['plot_color'], 
                                 outputFileNameBase = f'{exptName}_pair_properties', 
                                 outputFileNameExt = 'png',
                                 closeFigures = closeFigures,
                                 writeCSVs = False)
            """
            saved_relOrientation_outputs = make_relative_orientation_plots(
                                 all_expts[exptName]['datasets'], 
                                 exptName = exptName,
                                 distance_type = distance_type,
                                 color = all_expts[exptName]['plot_color'], 
                                 outputFileNameBase = f'{exptName}', 
                                 outputFileNameExt = 'svg',
                                 closeFigures = closeFigures,
                                 outputCSVFileName = f'relOrient_{exptName}.csv',
                                 makeSlicePlots = False)
            all_expts[exptName]["relOrient_2Dhist_mean"] = saved_relOrientation_outputs[0]
            all_expts[exptName]["relOrient_2Dhist_sem"] = saved_relOrientation_outputs[1]
            all_expts[exptName]["relOrient_2Dhist_X"] = saved_relOrientation_outputs[2]
            all_expts[exptName]["relOrient_2Dhist_Y"] = saved_relOrientation_outputs[3]
            """
            saved_pair_bending_outputs = make_bending_angle_plots(
                                 all_expts[exptName]['datasets'], 
                                 exptName = exptName,
                                 distance_type = distance_type,
                                 bending_threshold_deg = all_expts[exptName]['params']['bend_min_deg'],
                                 color = all_expts[exptName]['plot_color'], 
                                 outputFileNameBase = f'{exptName}', 
                                 outputFileNameExt = 'png',
                                 closeFigures = closeFigures,
                                 writeCSVs = False)
            all_expts[exptName]["bend_2Dhist_mean"] = saved_pair_bending_outputs[0]
            all_expts[exptName]["bend_2Dhist_sem"] = saved_pair_bending_outputs[1]
            all_expts[exptName]["bend_2Dhist_X"] = saved_pair_bending_outputs[2]
            all_expts[exptName]["bend_2Dhist_Y"] = saved_pair_bending_outputs[3]
            """

            saved_pair_turning_outputs = make_turning_angle_plots(
                                 all_expts[exptName]['datasets'], 
                                 exptName = exptName,
                                 distance_type = distance_type,
                                 color = all_expts[exptName]['plot_color'], 
                                 cmap = 'berlin',
                                 outputFileNameBase = f'{exptName}', 
                                 outputFileNameExt = 'svg',
                                 closeFigures = closeFigures,
                                 outputCSVFileName = f'turning_{exptName}.csv',
                                 makeSlicePlots = False)
            all_expts[exptName]["turn_2Dhist_mean"] = saved_pair_turning_outputs[0]
            all_expts[exptName]["turn_2Dhist_sem"] = saved_pair_turning_outputs[1]
            all_expts[exptName]["turn_2Dhist_std"] = saved_pair_turning_outputs[2]
            all_expts[exptName]["turn_2Dhist_X"] = saved_pair_turning_outputs[3]
            all_expts[exptName]["turn_2Dhist_Y"] = saved_pair_turning_outputs[4]   

    #%% Make pair plots of behavior v distance
    # Only for Nfish ==2

    makePairDistancePlots = False
    if makePairDistancePlots:
        closeFigures = True
        if closeFigures:
            print('Pair plots of behavior v distance: Closing Figure Windows.')
        
        distanceKey = 'closest_distance_mm'      # can make it be 'head_head_distance_mm'
        for exptName in all_expts.keys():
            if all_expts[exptName]['Nfish'] == 2:
                make_pair_1D_v_distance_plots(all_expts[exptName]['datasets'], 
                                              exptName = exptName,
                                              distanceKey=distanceKey,
                                              bin_range=(0.0, 50.0), Nbins=20,
                                              color = all_expts[exptName]['plot_color'],
                                              outputFileNameBase = f'{exptName} behavior', 
                                              outputFileNameExt = 'png',
                                              closeFigures = closeFigures,
                                              writeCSVs = False)


    #%% Make combination (multipage) images from individual PNGs

    makeCombinationImages= False
    if makeCombinationImages:
        fileNameStringsToCombine = ['speed', 'angularSpeed', 'radialpos', 'heading_angle',
                                    'radialAlignment_angle', 'boutSpeed',
                                    'speedAutocorr', 'distance_head_head', 
                                    'distance_closest', 'rel_heading_angle',
                                    'rel_orientation', 'rel_orientation_sum', 
                                    'rel_orientation_abs_sum',
                                    'orientation_distance_2D', 'IBI_v_dist_and_radialpos',
                                    'IBI_v_dist_r_interior', 'IBI_v_dist_r_edge',
                                    'bendAngle_distance_orientation_2D',
                                    'bendAngle_v_orientation_small', 
                                    'bendAngle_v_orientation_small_asymm',
                                    'bendAngle_v_orientation_middle',
                                    'bendAngle_v_orientation_middle_asymm',
                                    'movingSpeed_v_HHdistance_orientation_2D',
                                    'movingSpeed_v_HHdistance_lowOrientation',
                                    'speedCrosscorr', 'speedCrosscorrDistBinned']
                                
        excludeStrings = ['angular', '', 'IBI', '', 
                          '', '', '', '',  
                          '', '', 'sum', 'abs', '', 
                          '', '', '', '', '', 'asymm', '', 'asymm', '',
                          '', '', 'DistBinned', '']
    
        if len(fileNameStringsToCombine) != len(excludeStrings):
            raise ValueError('List lengths not equal for combine_images strings.')
        for s, excl_s in zip(fileNameStringsToCombine, excludeStrings):
            print(f'Combining {s}')
            combine_images_to_tiff(filenamestring = s, 
                                   path = r'C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\Code_current', 
                                   ext = 'png', exclude_string=excl_s)
    

    #%% Differences of turning angle or bending angle or relative orientation. 

    # distance_type should be defined earlier
    if distance_type == 'closest_distance':
        distanceStr = 'Closest Distance'
        ylabelStr = 'Closest Distance (mm)'
    elif distance_type == 'head_head_distance':
        distanceStr = 'HH Distance'
        ylabelStr = 'Head-Head Distance (mm)'
    else:
        raise ValueError('Invalide distance type')
    
    calcDifferences = 'relOrient' #'turning' # 'turning', 'bending', or 'relOrient' 
                             #or 'none' [anything else]

    if (calcDifferences == 'turning') or (calcDifferences == 'bending') \
        or (calcDifferences == 'relOrient') :
    
        if calcDifferences == 'turning':
            optionString = 'Turning'
            keyString = 'turn'
            mask_by_sem_limit_degrees = 2.0 # show points with s.e.m. < this
            colorRange = (-1.5*np.pi/180.0, 1.5*np.pi/180.0)
            xlim = (-np.pi, np.pi) # for relative orientation
            zlim = (-4*np.pi/180, 4*np.pi/180)
            # Mesh is the same for all datasets
            X = all_expts['TwoWk_Light']["turn_2Dhist_X"]
            Y = all_expts['TwoWk_Light']["turn_2Dhist_Y"]
            unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi]
            xlabelStr = 'Relative Orientation (deg)'
            clabelStr= f'Mean {optionString} Angle (degrees)'
        elif calcDifferences == 'bending':
            optionString = 'Bending'
            keyString = 'bend'
            mask_by_sem_limit_degrees = 6.0 # show points with s.e.m. < this
            colorRange = (-6*np.pi/180.0, 6*np.pi/180.0)
            zlim = (-15*np.pi/180, 15*np.pi/180)
            # Mesh is the same for all datasets
            X = all_expts['TwoWk_Light']["bend_2Dhist_X"]
            Y = all_expts['TwoWk_Light']["bend_2Dhist_Y"]
            unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi]
            xlim = (-np.pi, np.pi) # for relative orientation
            xlabelStr = 'Relative Orientation (deg)'
            clabelStr= f'Mean {optionString} Angle (degrees)'
        elif calcDifferences == 'relOrient':
            optionString = 'RelOrient'
            keyString = 'relOrient'
            mask_by_sem_limit_degrees = None
            colorRange = (-0.0075,0.0075)
            xlim = (0.0, np.pi)
            zlim = (-0.0075, 0.0075)
            # Mesh is the same for all datasets
            X = all_expts['TwoWk_Light']["relOrient_2Dhist_X"]
            Y = all_expts['TwoWk_Light']["relOrient_2Dhist_Y"]
            unit_scaling_for_plot = [180.0/np.pi, 1.0, 1.0]
            xlabelStr = 'Abs. Relative Orientation (deg)'
            clabelStr= f'ΔP({optionString})'
        else:
            raise ValueError('Calc differences must be bending, turning, or relOrient')
        
        cmap = 'berlin' # 'RdYlBu_r'
        plot_type = 'heatmap'
        xlabelStr = 'Relative Orientation (degrees)'
        outputExtension = '.svg' # for saving images

        outputFileNameBase = f'Difference in {optionString} Angle, ({distanceStr}), '
        mask_by_sem_limit = (mask_by_sem_limit_degrees * np.pi/180.0
                             if mask_by_sem_limit_degrees is not None else None)

        # Each tuple: (title_label, filename_label, exptKey_A, exptKey_B, slice_color)
        comparisons = [
            ('Light - TimeShift Light', 'Light - TS0Light', 'TwoWk_Light', 'TwoWk_Light_TIMESHIFT0', 'peru'),
            ('Light - Dark',            'Light - Dark',     'TwoWk_Light', 'TwoWk_Dark',             'darkseagreen'),
            ('Dark - TimeShift Dark',   'Dark - TS0Dark',   'TwoWk_Dark',  'TwoWk_Dark_TIMESHIFT0',  'blue'),
        ]

        for cmp_title, cmp_fname, keyA, keyB, slice_color in comparisons:
            d = (all_expts[keyA][f"{keyString}_2Dhist_mean"] -
                 all_expts[keyB][f"{keyString}_2Dhist_mean"])
            sem_A = all_expts[keyA][f"{keyString}_2Dhist_sem"]
            sem_B = all_expts[keyB][f"{keyString}_2Dhist_sem"]
            d_unc = np.sqrt(sem_A**2 + sem_B**2) if (sem_A is not None and sem_B is not None) else None

            if mask_by_sem_limit_degrees is None:
                titleStr = f'{cmp_title}: {optionString} Angle Probability'
            else:
                titleStr = f'{cmp_title}: {optionString} Angle; unc. < {mask_by_sem_limit_degrees:.1f} deg'

            outputCSVFileName = outputFileNameBase + cmp_fname + '.csv'
            print(outputCSVFileName)
            plot_2D_heatmap(d, X, Y, Z_unc=d_unc,
                            titleStr=titleStr, xlabelStr=xlabelStr, ylabelStr=ylabelStr,
                            clabelStr=clabelStr, colorRange=colorRange, cmap=cmap,
                            unit_scaling_for_plot=unit_scaling_for_plot,
                            mask_by_sem_limit=mask_by_sem_limit,
                            outputFileName=outputFileNameBase + cmp_fname + outputExtension,
                            outputCSVFileName = outputCSVFileName,
                            closeFigure=False)

            makeSlicePlots = False
            if makeSlicePlots:
                for d_range in [(0.0, 5.0), (5.0, 15.0)]:
                    if d_range[0] == 0.0:
                        title_dist = f'{cmp_title}: {optionString} Angle for d < {d_range[1]:.2f} mm'
                    else:
                        title_dist = f'{cmp_title}: {optionString} Angle for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
                    slice_2D_histogram(d, X, Y, d_unc,
                                    slice_axis='x', other_range=d_range,
                                    titleStr=title_dist, xlabelStr=xlabelStr,
                                    zlabelStr=clabelStr, ylabelStr=ylabelStr,
                                    zlim=zlim, xlim=xlim,
                                    plot_z_zero_line=True, plot_vert_zero_line=True,
                                    unit_scaling_for_plot=unit_scaling_for_plot,
                                    color=slice_color,
                                    outputFileName=outputFileNameBase + f'{cmp_fname} {d_range[0]:.1f}-{d_range[1]:.1f} mm' + outputExtension,
                                    closeFigure=False)

    #%% Correlations of pair behaviors
    # Note that this is slow!

    calculateCorrelations = False

    if calculateCorrelations:
    
        behavior_key_list = ['perp_noneSee', 'perp_oneSees', 'perp_bothSee', 
                             'contact_any', 'tail_rubbing_AP', 'tail_rubbing_P', 
                             'maintain_proximity', 'approaching_Fish0', 'approaching_Fish1', 
                             'fleeing_Fish0', 'fleeing_Fish1', 
                             'approaching_Fish_lowRelOrient0',  'approaching_Fish_lowRelOrient1', 
                             'fleeing_Fish_lowRelOrient0',  'fleeing_Fish_lowRelOrient1', 
                             'isActive_Fish_lowRelOrient0', 'isActive_Fish_lowRelOrient1', 
                             'isBending_Fish_lowRelOrient0', 'isBending_Fish_lowRelOrient1', 
                             'isMoving_Fish_lowRelOrient0', 'isMoving_Fish_lowRelOrient1', 
                             'Cbend_Fish_lowRelOrient0', 'Cbend_Fish_lowRelOrient1', 
                             'Jbend_Fish_lowRelOrient0', 'Jbend_Fish_lowRelOrient1', 
                             'Rbend_Fish_lowRelOrient0',  'Rbend_Fish_lowRelOrient1']
        behavior_key_list_subset1 = ['maintain_proximity', 'approaching_Fish_lowRelOrient0', 'approaching_Fish_lowRelOrient1', 'fleeing_Fish_lowRelOrient0', 'fleeing_Fish_lowRelOrient1', 'isBending_Fish_lowRelOrient0', 'isBending_Fish_lowRelOrient1', 'isMoving_Fish_lowRelOrient0', 'isMoving_Fish_lowRelOrient1']
        behavior_key_list_subset2 = ['perp_noneSee', 'perp_oneSees', 'perp_bothSee', 
                                     'contact_any', 'tail_rubbing_AP', 'tail_rubbing_P',
                                     'maintain_proximity', 'Cbend_Fish_lowRelOrient0', 
                                     'Cbend_Fish_lowRelOrient1']
        behavior_key_list_subset = behavior_key_list_subset1 + behavior_key_list_subset2
    
        binWidthFrames = 1 # bin size for delays (number of frames)
        halfFrameRange = 50 # max frame delay to consider
    
        for exptName in all_expts.keys():
            if all_expts[exptName]['Nfish'] == 2:

                print(f'\n\n -- {exptName} -- \n')
                this_datasets = datasets=all_expts[exptName]['datasets']
                fps = get_fps(this_datasets, fpstol = 1e-6)
                # behavior_key_list = get_behavior_key_list(datasets=all_expts[exptName]['datasets'])
                min_duration_behavior = 'maintain_proximity'
                min_duration_fr = 13
                behav_corr = calcDeltaFramesEvents(datasets = this_datasets, 
                                                   behavior_key_list = behavior_key_list, 
                                                   max_delta_frame = 150, 
                                                   min_duration_behavior = min_duration_behavior , 
                                                   min_duration_fr = min_duration_fr)
                behav_corr, binCenters = bin_deltaFrames(behav_corr, behavior_key_list, 
                                                         binWidthFrames = binWidthFrames, halfFrameRange = halfFrameRange, deleteDeltaFrames = True)
                behav_corr = calc_pAB(behav_corr, behavior_key_list, binCenters)
                behav_corr_allSets = calcBehavCorrAllSets(behav_corr, behavior_key_list, 
                                                          binCenters)
                exType = 'Light '  # e.g. 'Light ' include space at the end 
                behaviorA = 'maintain_proximity'
                behaviorB = ''
                plot_behaviorCorrelation(behav_corr_allSets['DeltaCorr'], binCenters, 
                                         behavior_key_list_subset1, behaviorA, behaviorB, 
                                         titleString = f'{exptName}' + r'$\Delta$P', 
                                         fps = fps, 
                                         plotShadedUnc = True, 
                                         outputFileName = f'{exptName}_behav_after_{behaviorA}_1.png')
                plot_behaviorCorrelation(behav_corr_allSets['DeltaCorr'], binCenters, 
                                         behavior_key_list_subset2, behaviorA, behaviorB, 
                                         titleString = f'{exptName}' + r'$\Delta$P', 
                                         fps = fps, 
                                         plotShadedUnc = True, 
                                         outputFileName = f'{exptName}_behav_after_{behaviorA}_2.png')
                behaviorA = 'contact_any'
                behaviorB = ''
                plot_behaviorCorrelation(behav_corr_allSets['DeltaCorr'], binCenters, 
                                         behavior_key_list_subset1, behaviorA, behaviorB, 
                                         titleString = f'{exptName}' + r'$\Delta$P', 
                                         fps = fps, 
                                         plotShadedUnc = True, 
                                         outputFileName = f'{exptName}_behav_after_{behaviorA}_1.png')
                plot_behaviorCorrelation(behav_corr_allSets['DeltaCorr'], binCenters, 
                                         behavior_key_list_subset2, behaviorA, behaviorB, 
                                         titleString = f'{exptName}' + r'$\Delta$P', 
                                         fps = fps, 
                                         plotShadedUnc = True, 
                                         outputFileName = f'{exptName}_behav_after_{behaviorA}_2.png')
                corr_asymm = calc_corr_asymm(behav_corr_allSets['DeltaCorr'], 
                                             behavior_key_list_subset, binCenters, 
                                             maxFrameDelay = None, normalization = 'none')
                plot_corr_asymm(corr_asymm, crange=None, 
                                titleString = f'{exptName}' + r'corr_asymm_$\Delta$P', 
                                outputFileName = f'{exptName}_corr_asymm.png')
    else:
        print('\n\nNot calculating correlations.')