# -*- coding: utf-8 -*-
# behavior_plots.py
"""
Author:   Raghuveer Parthasarathy
Created: June 5, 2026 (functions moved from other modules)
Last modified: June 5, 2026 -- Raghuveer Parthasarathy

Description
-----------
Plotting and visualization functions for zebrafish behavioral analysis.

Contains high-level plot orchestrators (make_*_plots) extracted from
behavior_identification.py and behavior_identification_single.py, and
make_2D_histogram / slice_2D_histogram extracted from IO_toolkit.py.

Generic plot primitives (plot_probability_distr, plot_2D_heatmap, etc.)
remain in IO_toolkit.py and are imported from there.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import pandas as pd
import os
from toolkit import (combine_all_values_constrained, get_effective_dims,
                     wrap_to_pi, make_frames_dictionary, dilate_frames,
                     get_values_subset, calculate_value_corr_all,
                     calculate_value_corr_all_binned, fit_gaussian_mixture,
                     behaviorFrameCount_all)
from IO_toolkit import (plot_2D_heatmap, plot_2Darray_linePlots,
                        plot_probability_distr, plot_function_allSets,
                        plot_waterfall_binned_crosscorr,
                        get_plot_and_CSV_filenames, calculate_property_1Dbinned,
                        simple_write_CSV)
from behavior_identification import (calculate_bout_property_binned_by_distance,
                                     calculate_IBI_binned_by_2D_keys,
                                     calculate_interfish_bout_lags)
from behavior_identification_single import (average_bout_trajectory_allSets,
                                            average_bout_trajectory_oneSet)


def plot_property_1Dbinned(binned_mean, bin_centers, binned_mean_each_dataset,
                           plot_each_dataset=True, plot_sem_band=False,
                           titleStr=None, xlabelStr=None, ylabelStr=None,
                           color='black', xlim=None, ylim=None,
                           unit_scaling_for_plot=None,
                           outputFileName=None, closeFigure=False,
                           outputCSVFileName=None):
    """
    Plot the output of calculate_property_1Dbinned() as an errorbar plot.

    Inputs
    ------
    binned_mean : (Nbins, 3) array — columns are [mean, std, sem] per bin
    bin_centers : 1D array of bin centre values
    binned_mean_each_dataset : (Ndatasets, Nbins) array
    plot_each_dataset : bool — draw a semi-transparent line per dataset
    plot_sem_band : bool — draw a shaded ±s.e.m. band around the mean
    titleStr, xlabelStr, ylabelStr : str — axis and title labels
    color : plot colour
    xlim, ylim : (min, max) axis limits in raw (unscaled) units
    unit_scaling_for_plot : [x_scale, y_scale] — multiply axis values for display
    outputFileName : str or None — save figure to this path
    closeFigure : bool — close figure after creating
    outputCSVFileName : str or None — write bin_centers + stats to CSV
    """
    if unit_scaling_for_plot is None:
        unit_scaling_for_plot = [1.0, 1.0]

    Ndatasets = binned_mean_each_dataset.shape[0]

    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(bin_centers * unit_scaling_for_plot[0],
                 binned_mean[:, 0] * unit_scaling_for_plot[1],
                 binned_mean[:, 2] * unit_scaling_for_plot[1],
                 fmt='o-', capsize=7, markersize=12, linewidth=2,
                 color=color, ecolor=color)

    if plot_sem_band:
        plt.fill_between(bin_centers * unit_scaling_for_plot[0],
                         (binned_mean[:, 0] - binned_mean[:, 2]) * unit_scaling_for_plot[1],
                         (binned_mean[:, 0] + binned_mean[:, 2]) * unit_scaling_for_plot[1],
                         color=color, alpha=0.4, label='s.e.m.')

    if plot_each_dataset:
        alpha_each = np.max((0.7 / Ndatasets, 0.15))
        for i in range(Ndatasets):
            plt.plot(bin_centers * unit_scaling_for_plot[0],
                     binned_mean_each_dataset[i, :] * unit_scaling_for_plot[1],
                     color=color, alpha=alpha_each)

    if xlabelStr is None:
        xlabelStr = 'x'
    if ylabelStr is None:
        ylabelStr = 'y'
    if titleStr is None:
        titleStr = 'Binned property'

    plt.xlabel(xlabelStr, fontsize=14)
    plt.ylabel(ylabelStr, fontsize=14)
    plt.title(titleStr, fontsize=16)
    plt.grid(True, alpha=0.3)

    if xlim is not None:
        plt.xlim([v * unit_scaling_for_plot[0] for v in xlim])
    if ylim is not None:
        plt.ylim([v * unit_scaling_for_plot[1] for v in ylim])

    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')

    if closeFigure:
        plt.close(fig)
    else:
        plt.show()

    if outputCSVFileName is not None:
        header_strings = [xlabelStr.replace(',', '_'),
                          f'{ylabelStr} mean', f'{ylabelStr} std', f'{ylabelStr} s.e.m.']
        for i in range(Ndatasets):
            header_strings.append(f'{ylabelStr}_Dataset_{i+1}')
        list_to_output = [binned_mean[:, 0], binned_mean[:, 1], binned_mean[:, 2]]
        for i in range(Ndatasets):
            list_to_output.append(binned_mean_each_dataset[i, :])
        simple_write_CSV(bin_centers, list_to_output,
                         filename=outputCSVFileName,
                         header_strings=header_strings)


def make_pair_fish_plots(datasets, exptName = '',
                         distance_type = 'closest_distance',
                         color = 'black',
                         plot_type_2D = 'heatmap',
                         outputFileNameBase = 'pair_fish', 
                         outputFileNameExt = 'png',
                         closeFigures = False,
                         writeCSVs = False):
    """
    Makes several useful "pair" plots -- i.e. plots of characteristics 
    of pairs of fish.
    Note that there are lots of parameter values that are hard-coded; this
    function is probably more useful to read than to run, pasting and 
    modifying its code.
    Bending angle plots extracted and moved to make_bending_angle_plots()
    
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        distance_type : (string) : either 'closest_distance' or 'head_head_distance',
                        used to make labels for some of the plots
        color: plot color (uses alpha for indiv. dataset colors)
        plot_type_2D : str, 'heatmap' or 'line_plots'
                    Which plotting function make_2D_histogram() will use
                    ('heatmap' or 'line_plots')
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        writeCSVs : (bool) Used by various functions; if true, output plotted 
                            points to a CSV file. See code for filenames

    Outputs:
        None

    """
        
    verifyPairs = True
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            verifyPairs = False
    if verifyPairs==False:
        raise ValueError('Error in make_pair_fish_plots; Nfish must be 2 !')

    # distance_type is not always used, but if it is, these may be useful:
    if distance_type == 'closest_distance':
        distanceKey = 'closest_distance_mm'
        distanceStr = 'Closest Distance'
        distancelabelStr = 'Closest Distance (mm)'
        shortDistanceStr = 'ClDist'
    elif distance_type == 'head_head_distance':
        distanceKey = 'head_head_distance_mm'
        distanceStr = 'HH Distance'
        distancelabelStr = 'Head-Head Distance (mm)'
        shortDistanceStr = 'HHDist'
    else:
        raise ValueError('Invalide distance type')
    
    """
    # head-head distance histogram
    head_head_mm_all = combine_all_values_constrained(datasets, 
                                                     keyName='head_head_distance_mm', 
                                                     dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_distance_head_head', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    plot_probability_distr(head_head_mm_all, bin_width = 0.5, 
                           bin_range = [0, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           xlim = (-1.0, 50.0), ylim = (-0.005, 0.05),
                           xlabelStr = 'Head-head distance (mm)', 
                           titleStr = f'{exptName}: head-head distance (mm)',
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    # closest distance histogram
    closest_distance_mm_all = combine_all_values_constrained(datasets, 
                                                     keyName='closest_distance_mm', 
                                                     dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_distance_closest', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    plot_probability_distr(closest_distance_mm_all, bin_width = 0.5, 
                           bin_range = [0, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           xlim = (-1.0, 50.0), ylim = (-0.005, 0.15),
                           xlabelStr = 'Closest distance (mm)', 
                           titleStr = f'{exptName}: closest distance (mm)',
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    # Relative heading angle histogram
    relative_heading_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_heading_angle', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_rel_heading_angle', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/30
    plot_probability_distr(relative_heading_angle_all, bin_width = bin_width,
                           bin_range=[None, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           polarPlot = True,
                           titleStr = f'{exptName}: Relative Heading Angle',
                           ylim = (0, 0.6),
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    # Relative orientation angle histogram
    relative_orientation_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_orientation', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_rel_orientation', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/30
    plot_probability_distr(relative_orientation_angle_all, 
                                  bin_width = bin_width,
                                  bin_range=[None, None], 
                                  color = color,
                                  yScaleType = 'linear',
                                  plot_each_dataset = False,
                                  plot_sem_band = True,
                                  polarPlot = True,
                                  titleStr = f'{exptName}: Relative Orientation Angle',
                                  ylim = (0, 0.6),
                                  outputFileName = outputFileName,
                                  closeFigure = closeFigures,
                                  outputCSVFileName = outputCSVFileName)
    
    # Relative orientation angle histogram constrained by inter-fish distance
    dRange=(0.0, 15.0)
    relative_orientation_angle_all_constr = combine_all_values_constrained(
        datasets, 
                                                 keyName='relative_orientation',
                                                 keyIdx = None,
                                                 use_abs_value = False,
                                                 constraintKey=distanceKey,
                                                 constraintRange=dRange,
                                                 constraintIdx = None,
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames( \
        f'_rel_orientation_{shortDistanceStr}_{dRange[0]:.1f}_{dRange[1]:.1f}mm', 
        outputFileNameBase, outputFileNameExt, writeCSVs)
    bin_width = np.pi/30
    plot_probability_distr(relative_orientation_angle_all_constr, 
                                  bin_width = bin_width,
                                  bin_range=[None, None], 
                                  color = color,
                                  yScaleType = 'linear',
                                  plot_each_dataset = False,
                                  plot_sem_band = True,
                                  polarPlot = True,
                                  titleStr = f'{exptName}: Rel. Orient. Angle,' +
                                     f'{shortDistanceStr}_{dRange[0]:.1f}_{dRange[1]:.1f}mm',
                                  ylim = (0, 0.4),
                                  outputFileName = outputFileName,
                                  closeFigure = closeFigures,
                                  outputCSVFileName = outputCSVFileName)
    
    
    # Sum of relative orientation angles histogram
    relative_orientation_sum_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_orientation_sum', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_rel_orientation_sum', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/60
    plot_probability_distr(relative_orientation_sum_all, bin_width = bin_width,
                           bin_range=[None, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           polarPlot = False,
                           titleStr = f'{exptName}: Sum of Relative Orientation Angles',
                           xlabelStr = 'Sum of Rel. Orient. Angles (rad)',
                           ylim = (0, 0.6), xlim = (-6.3, 6.3),
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    # Sum of absolute value of relative orientation angles histogram
    relative_orientation_abs_sum_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_orientation_abs_sum', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_rel_orientation_abs_sum', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/60
    plot_probability_distr(relative_orientation_abs_sum_all, bin_width = bin_width,
                           bin_range=[None, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           polarPlot = False,
                           titleStr = f'{exptName}: Sum of Relative Orientation Angles',
                           xlabelStr = 'Sum of Rel. Orient. Angles (rad)',
                           ylim = (0, 0.6), xlim = (0.0, 6.3),
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)
    """
    """
    # 2D histogram of heading alignment and head-head distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_heading_distance_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    # use heatmap as plot type; for 2D histogram slicing along Y is not helpful
    make_2D_histogram(datasets, keyNames = ('head_head_distance_mm', 
                                            'relative_heading_angle'), 
                          keyIdx = (None, None), 
                          dilate_minus1=False, 
                          bin_ranges=((0.0, 50.0), (0.0, 3.142)), 
                          Nbins=(20,20),
                          colorRange = (0.0, 0.007),
                          titleStr = f'{exptName}: heading angle and hh distance', 
                          cmap = 'viridis',
                          plot_type = 'heatmap',
                          outputFileName = outputFileName,
                          closeFigure = closeFigures)
    """

    # 2D histogram of abs(relative orientation) and distance (whichever 
    # distance type specified earier)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + \
            f'_orientation_{shortDistanceStr}_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    # use heatmap as plot type; for 2D histogram slicing along Y is not helpful
    make_2D_histogram(datasets, keyNames = (distanceKey, 'relative_orientation'), 
                          keyIdx = (None, None), 
                          use_abs_value = (False, True),
                          dilate_minus1=False, bin_ranges=((0.0, 50.0), 
                                                           (0.0, 3.142)), 
                          Nbins=(20,20), 
                          unit_scaling_for_plot = [1.0, 180.0/np.pi, 1.0],
                          xlabelStr = distancelabelStr, 
                          ylabelStr = 'Rel. Orientation (deg)',
                          cmap = 'viridis',
                          colorRange = (0.0, 0.0075),
                          clabelStr = 'Probability',
                          titleStr = f'{exptName}: abs(Rel. orient.) and {shortDistanceStr}', 
                          plot_type = 'heatmap',
                          outputFileName = outputFileName,
                          closeFigure = closeFigures)

    """
    # Speed of the "other" fish vs. time for bouts
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_boutSpeed_other' + '.' + outputFileNameExt
    else:
        outputFileName = None
    average_bout_trajectory_allSets(datasets, keyName = "speed_array_mm_s", 
                                    keyIdx = 'other', t_range_s=(-1.0, 2.0), 
                                    titleStr = f'{exptName}: Bout Speed, other fish', makePlot=True,
                                    outputFileName = outputFileName,
                                    closeFigure = closeFigures)
    # Speed vs. time for bouts, distance constraint
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_boutSpeed_close' + '.' + outputFileNameExt
    else:
        outputFileName = None
    max_d = 5.0 # mm, for constraint
    average_bout_trajectory_allSets(datasets, keyName = "speed_array_mm_s", 
                                    keyIdx = None, t_range_s=(-1.0, 2.0), 
                                    constraintKey='head_head_distance_mm', 
                                    constraintRange=(0, 5.0), 
                                    titleStr = f'Bout Speed, d < {max_d:.1f} mm', 
                                    makePlot=True, 
                                    outputFileName = outputFileName,
                                    closeFigure = closeFigures)
    """
        
    """
    # 2D histogram of C- and J-bend frequencies (combined) vs head-head distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_CJ_distance_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    keyList = ['Cbend_any', 'Jbend_any']
    datasets = behaviorFrameCount_all(datasets, keyList, 'CJcombined')
    make_2D_histogram(datasets, keyNames = ('head_head_distance_mm', 
                      'CJcombined'), Nbins=(15,10), 
                      constraintKey='CJcombined', constraintRange=(0.5,100), 
                      colorRange=(0, 0.002), cmap = 'viridis', 
                      titleStr = f'{exptName}: CJ probability', 
                      outputFileName = outputFileName,
                      closeFigure = closeFigures)
    
    # Inter-bout interval (IBI) binned by inter-fish distance.
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_distance' + '.' + outputFileNameExt
    else:
        outputFileName = None
    calculate_IBI_binned_by_distance(
        datasets=datasets, 
        distance_key='head_head_distance_mm',bin_distance_min=0, 
        bin_distance_max=50.0, bin_width=5.0, dilate_minus1=False,
        outlier_std = 3.0,
        makePlot = True, ylim = (0.2, 0.55), titleStr = exptName, 
        plotColor = color,
        outputFileName = outputFileName,
        closeFigure = closeFigures)

    # Inter-bout interval (IBI) binned by radial position.
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_r' + '.' + outputFileNameExt
    else:
        outputFileName = None
    calculate_IBI_binned_by_distance(
        datasets=datasets, 
        distance_key='radial_position_mm',bin_distance_min=0, 
        bin_distance_max=25.0, bin_width=5.0, dilate_minus1=False,
        makePlot = True, ylim = (0.2, 0.55), titleStr = exptName, 
        plotColor = color,
        outputFileName = f'{exptName}_IBI_v_r.png',
        closeFigure = closeFigures)
    """
    
    """
    # Inter-bout interval (IBI) binned by inter-fish distance *and* 
    # radial position
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_dist_and_radialpos' + '.' + outputFileNameExt
    else:
        outputFileName = None
    binned_IBI, X, Y, binned_IBI_each_dataset, binned_IBI_each_fish = \
        calculate_IBI_binned_by_2D_keys(datasets=datasets, 
                                     key1='head_head_distance_mm',
                                     key2='radial_position_mm',
                                     bin_ranges=((0.0, 50.0), (0.0, 25.0)), 
                                     Nbins=(12, 12),
                                     dilate_minus1=False,
                                     outlier_std=3.0,
                                     makePlot=True, 
                                     titleStr = f'{exptName}: IBI vs. head-head distance, radial pos.', 
                                     cmap='viridis_r',
                                     colorRange = (0.0, 0.6),
                                     plot_type = plot_type_2D,
                                     outputFileName=outputFileName,
                                     closeFigure=closeFigures)
        
    # Slice along IBI binned by distance and r, for r < 22 mm
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_dist_r_interior' + '.' + outputFileNameExt
    else:
        outputFileName = None
    r_range = (15.0, 22.0)
    titleStr = f'{exptName}: Average IBI for {r_range[0]:.1f} r < {r_range[1]:.1f} mm'
    xlabelStr = 'Closest Distance (mm)'
    ylabelStr = 'Radial position (mm)'
    zlabelStr = 'Average IBI (s)'
    xlim = (0.0, 50.0)
    zlim = (0.0, 0.6)
    color = color
    slice_2D_histogram(binned_IBI[:,:,0], X, Y, binned_IBI[:,:,2], 
                       slice_axis = 'x', other_range = r_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim,
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)

    # Slice along IBI binned by distance and r, for r >= 22 mm
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_dist_r_edge' + '.' + outputFileNameExt
    else:
        outputFileName = None
    r_range = (22.0, np.inf)
    titleStr = f'{exptName}: Average IBI for r >= {r_range[0]:.1f} mm'
    xlabelStr = 'Closest Distance (mm)'
    ylabelStr = 'Radial position (mm)'
    zlabelStr = 'Average IBI (s)'
    xlim = (0.0, 50.0)
    zlim = (0.0, 0.6)
    color = color
    slice_2D_histogram(binned_IBI[:,:,0], X, Y, binned_IBI[:,:,2], 
                       slice_axis = 'x', other_range = r_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)

    
    # Average above-threshold speed versus distance and relative orientation
    # Hard-code speed threshold 
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_movingSpeed_v_HHdistance_orientation_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    speed_threshold = 9.0 # mm/s 
    keyNames = ('head_head_distance_mm', 'relative_orientation')
    keyIdx = (None, None)
    use_abs_value = (False, True)
    keyNameC = 'speed_array_mm_s'
    keyIdxC = None
    constraintKey= 'speed_array_mm_s'
    constraintRange=(speed_threshold, np.inf)
    constraintIdx = None
    use_abs_value_constraint = False
    bin_ranges= ((0.0, 50.0), (0.0, 3.142))
    Nbins = (20, 12)
    titleStr = f'{exptName}: Avg speed when > {constraintRange[0]:.1f} mm/s'
    xlabelStr = 'Head-Head Distance (mm)'
    ylabelStr = 'Relative Orientation (rad)'
    colorRange = (constraintRange[0], 40.0)
    hist_speed, X, Y, hist_speed_sem, _ = make_2D_histogram(
        datasets,
        keyNames = keyNames, keyIdx = keyIdx, use_abs_value = use_abs_value,
        keyNameC = keyNameC, keyIdxC =keyIdxC, 
        constraintKey = constraintKey, constraintRange = constraintRange, 
        constraintIdx = constraintIdx, 
        use_abs_value_constraint = use_abs_value_constraint,
        dilate_minus1=True, bin_ranges = bin_ranges, Nbins = Nbins, 
        titleStr = titleStr,
        clabelStr= 'Mean above-threshold speed (mm/s)',
        xlabelStr = xlabelStr, ylabelStr = ylabelStr, 
        colorRange = colorRange, 
        plot_type = plot_type_2D,
        outputFileName = outputFileName, 
        closeFigure = closeFigures)

    # Slice along above-threshold speed versus distance and relative orientation binned by distance and orientation, 
    # orientation axis, constrain orientation angle to be < 60 degres
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_movingSpeed_v_HHdistance_lowOrientation' + '.' + outputFileNameExt
    else:
        outputFileName = None
    orientation_range = (0.0, np.pi/3.0)
    titleStr = f'{exptName}: Avg speed when > {speed_threshold:.1f} mm/s, ' + \
        f'for |rel. orient| < {180.0*orientation_range[1]/np.pi:.3f} rad'
    xlabelStr = 'Head-Head Distance (mm)'
    ylabelStr = 'Rel. Orient. Angle (rad)'
    zlabelStr = 'Average speed (mm/s)'
    xlim = (0.0, 50.0)
    zlim = (0.0, 90.0)
    color = color
    slice_2D_histogram(hist_speed, X, Y, hist_speed_sem, 
                       slice_axis = 'x', other_range = orientation_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = False,
                       plot_vert_zero_line = False,
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)


    # Speed cross-correlation function
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_speedCrosscorr', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    speed_cc_all, t_lag = \
        calculate_value_corr_all(datasets, keyName = 'speed_array_mm_s',
                                 corr_type='cross', dilate_minus1 = True, 
                                 t_max = 2.0, t_window = 5.0, fpstol = 1e-6)
    plot_function_allSets(speed_cc_all, t_lag, xlabelStr='time (s)', 
                          ylabelStr='Speed Cross-correlation', 
                          titleStr=f'{exptName}: Speed Cross-correlation', 
                          ylim = (-0.03, 0.2),
                          color = color,
                          plot_each_dataset = True, 
                          average_in_dataset = True,
                          outputFileName = outputFileName,
                          closeFigure = closeFigures,
                          outputCSVFileName = outputCSVFileName)
    
    # Waterfall plot of speed cross-correlations binned by distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_speedCrosscorrDistBinned' + '.' + outputFileNameExt
    else:
        outputFileName = None
    binned_crosscorr_all, bin_centers, t_lag, bin_counts_all = \
        calculate_value_corr_all_binned(datasets, keyName='speed_array_mm_s', 
                                        binKeyName = 'closest_distance_mm', 
                                        bin_value_min = 0.0, bin_value_max = 50.0, 
                                        bin_width=5.0, t_max=2.0, t_window=5.0,
                                        dilate_minus1=True)
    plot_waterfall_binned_crosscorr(binned_crosscorr_all, bin_centers, t_lag,
                                    bin_counts_all=bin_counts_all, 
                                    xlabelStr='Time lag (s)',
                                    titleStr=f'{exptName}: Closest Distance-Binned Cross-correlation',
                                    outputFileName=outputFileName,
                                    closeFigure = closeFigures)
    
    """
    return None


def make_pair_1D_v_distance_plots(datasets, exptName = '', 
                                  distanceKey='closest_distance_mm', 
                                  bin_range=(0.0, 50.0), Nbins=20,
                                  color = 'black',
                                  outputFileNameBase = 'pair_', 
                                  outputFileNameExt = 'png',
                                  closeFigures = False,
                                  writeCSVs = False):
    """
    Makes several useful plots of something vs. inter-fish distance
    
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        distanceKey : (string) which distance key to use, 
                      'closest_distance_mm' or 'head_head_distance_mm'
        bin_range : tuple,  (min, max) for binning
        Nbins : int,  Number of bins
        color: plot color (uses alpha for indiv. dataset colors)
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        writeCSVs : (bool) Used by various functions; if true, output plotted 
                            points to a CSV file. See code for filenames

    Outputs:
        None

    """
        
    verifyPairs = True
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            verifyPairs = False
    if verifyPairs==False:
        raise ValueError('Error in make_pair_1D_v_distance_plots; Nfish must be 2 !')
    
    if distanceKey=='closest_distance_mm':
        xlabelStr = 'Closest distance (mm)'
        distance_abbrev = 'CL'
    elif distanceKey=='head_head_distance_mm':
        xlabelStr = 'Head-Head distance (mm)'
        distance_abbrev = 'HH'
    else: 
        raise ValueError('Invalid distance key.')
        
    # Perpendicular one-sees
    keyName = 'perp_oneSees'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(
                                            f'_{keyName}_v_{distance_abbrev}distance',
                                            outputFileNameBase,
                                            outputFileNameExt, writeCSVs)
    binned_mean, bin_centers, binned_mean_each_dataset, _ = calculate_property_1Dbinned(
                                          datasets,
                                          keyName=keyName,
                                          key_is_a_behavior=True,
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1=False)
    plot_property_1Dbinned(binned_mean, bin_centers, binned_mean_each_dataset,
                           titleStr=titleStr, xlabelStr=xlabelStr,
                           ylabelStr=ylabelStr, color=color,
                           outputFileName=outputFileName, closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)

    # J-Bend
    keyName = 'Jbend_any'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(
                                            f'_{keyName}_v_{distance_abbrev}distance',
                                            outputFileNameBase,
                                            outputFileNameExt, writeCSVs)
    binned_mean, bin_centers, binned_mean_each_dataset, _ = calculate_property_1Dbinned(
                                          datasets,
                                          keyName=keyName,
                                          key_is_a_behavior=True,
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1=False)
    plot_property_1Dbinned(binned_mean, bin_centers, binned_mean_each_dataset,
                           titleStr=titleStr, xlabelStr=xlabelStr,
                           ylabelStr=ylabelStr, color=color,
                           outputFileName=outputFileName, closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)

    # C-Bend
    keyName = 'Cbend_any'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(
                                            f'_{keyName}_v_{distance_abbrev}distance',
                                            outputFileNameBase,
                                            outputFileNameExt, writeCSVs)
    binned_mean, bin_centers, binned_mean_each_dataset, _ = calculate_property_1Dbinned(
                                          datasets,
                                          keyName=keyName,
                                          key_is_a_behavior=True,
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1=False)
    plot_property_1Dbinned(binned_mean, bin_centers, binned_mean_each_dataset,
                           titleStr=titleStr, xlabelStr=xlabelStr,
                           ylabelStr=ylabelStr, color=color,
                           outputFileName=outputFileName, closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)

    # R-Bend
    keyName = 'Rbend_any'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(
                                            f'_{keyName}_v_{distance_abbrev}distance',
                                            outputFileNameBase,
                                            outputFileNameExt, writeCSVs)
    binned_mean, bin_centers, binned_mean_each_dataset, _ = calculate_property_1Dbinned(
                                          datasets,
                                          keyName=keyName,
                                          key_is_a_behavior=True,
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1=False)
    plot_property_1Dbinned(binned_mean, bin_centers, binned_mean_each_dataset,
                           titleStr=titleStr, xlabelStr=xlabelStr,
                           ylabelStr=ylabelStr, color=color,
                           outputFileName=outputFileName, closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)

    # isActive (moving or bending)
    keyName = 'isActive_any'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(
                                            f'_{keyName}_v_{distance_abbrev}distance',
                                            outputFileNameBase,
                                            outputFileNameExt, writeCSVs)
    binned_mean, bin_centers, binned_mean_each_dataset, _ = calculate_property_1Dbinned(
                                          datasets,
                                          keyName=keyName,
                                          key_is_a_behavior=True,
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1=False)
    plot_property_1Dbinned(binned_mean, bin_centers, binned_mean_each_dataset,
                           titleStr=titleStr, xlabelStr=xlabelStr,
                           ylabelStr=ylabelStr, color=color,
                           outputFileName=outputFileName, closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)


    return None


def make_bending_angle_plots(datasets, exptName = '', distance_type = None,
                             bending_threshold_deg = 0.0,
                             color = 'black',
                             plot_type_2D = 'heatmap',
                             cmap = 'RdYlBu_r', 
                             outputFileNameBase = 'bending_angle', 
                             outputFileNameExt = 'png',
                             closeFigures = False,
                             writeCSVs = False):
    
    """
    Makes several useful plots of bending angle properties for 
    of pairs of fish.
    
    Removed from make_pair_fish_plots()
        
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        bending_threshold_deg : (float) for a plot of mean bending angle
                constrained to abs(angle) > threshold, use this threshold.
                Input in degrees. 
        distance_type : string, either closest_distance or head_head_distance
                Default is None; user should think about this!
        color: plot color (uses alpha for indiv. dataset colors)
        plot_type_2D : str, 'heatmap' or 'line_plots'
                    Which plotting function make_2D_histogram() will use
                    ('heatmap' or 'line_plots')
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        writeCSVs : (bool) Used by various functions; if true, output plotted 
                            points to a CSV file. See code for filenames

    Outputs:
        saved_pair_outputs : list, containing
            0 : bend_2Dhist_mean, mean 2D bending angle histogram
            1 : bend_2Dhist_std, std dev for 2D bending angle histogram
            2: bin positions ("X") for head_head_distance_mm for 2D bending angle histogram
            3: bin positions ("Y") for relative orientation for 2D bending angle histogram

    To do:
        Redundant code for slicing, symmetrization. Probably not worth cleaning up.
        
    """
    
    # Make sure of distance measure being used
    if not (distance_type==None or distance_type == 'closest_distance' or \
            distance_type == 'head_head_distance'):
        print('\nDistance measure must be "closest_distance" or "head_head_distance".\n')
        distance_type = None
    if distance_type == None:
        distance_type_choice = 0
        while not ((distance_type_choice  == 1) or (distance_type_choice  == 2)):
            distance_type_choice = int(input('Choose distance measure ' + 
                                             '\n  (1) closest_distance ' + 
                                             '\n  (2) head_head_distance' +
                                             '\nEnter "1" or "2": '))
        if distance_type_choice==1:
            distance_type = 'closest_distance'
        else:
            distance_type = 'head_head_distance'

    # Strings for file output, labels
    if distance_type == 'closest_distance':
        distance_file_string = 'closestDistance'
        distanceLabelStr = 'Closest Distance (mm)'
    elif distance_type == 'head_head_distance':
        distance_file_string = 'headHeadDistance'
        distanceLabelStr = 'Head-Head Distance (mm)'
    else:
        raise ValueError('Invalid distance type')
    
        
    saved_pair_outputs = []

    
    verifyPairs = True
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            verifyPairs = False
    if verifyPairs==False:
        raise ValueError('Error in make_pair_fish_plots; Nfish must be 2 !')


    # 2D plot of mean bending angle vs. relative orientation and distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + \
            f'_bendAngle_{distance_file_string}_orientation_2D' + \
                '.' + outputFileNameExt
    else:
        outputFileName = None
    mask_by_sem_limit_degrees = 2.0 # show points with s.e.m. < this
    use_abs_value = (False, False)
    titleStr = f'{exptName}: Bend Angle; unc. < {mask_by_sem_limit_degrees:.1f} deg'
    # Save the output 2D histograms, for use later.
    bend_2Dhist_mean, X, Y, bend_2Dhist_sem, _ = make_2D_histogram(
        datasets,
        keyNames = ('relative_orientation', f'{distance_type}_mm'),
        keyIdx = (None, None), 
        use_abs_value = use_abs_value,
        keyNameC = 'bend_angle', keyIdxC = None,
        colorRange = (-12*np.pi/180.0, 12*np.pi/180.0),
        dilate_minus1= False, 
        bin_ranges = ((-np.pi, np.pi), (0.0, 30.0)), Nbins = (19,15), 
        titleStr = titleStr,
        clabelStr= 'Mean Bending Angle (degrees)',
        xlabelStr = 'Relative Orientation (degrees)',
        ylabelStr = distanceLabelStr, 
        mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
        unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
        cmap = cmap,
        plot_type = plot_type_2D,
        outputFileName = outputFileName,
        closeFigure = closeFigures)
    saved_pair_outputs.append(bend_2Dhist_mean)
    saved_pair_outputs.append(bend_2Dhist_sem)
    saved_pair_outputs.append(X)
    saved_pair_outputs.append(Y)

    # Slice bend angle binned by distance and orientation, along the
    # orientation axis, distance slice: distance < 2.5 mm
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + f'_bendAngle_v_orientation_small_{distance_file_string}' + '.' + outputFileNameExt
    else:
        outputFileName = None
    d_range = (0.0, 2.5)
    xlabelStr = 'Relative Orientation (deg)'
    titleStr = f'{exptName}: Bend Angle for d < {d_range[1]:.2f} mm'
    zlabelStr = 'Mean Bending Angle (degrees)'
    xlim = (-np.pi, np.pi)
    zlim = (-15*np.pi/180, 15*np.pi/180)
    color = color
    slice_2D_histogram(bend_2Dhist_mean, X, Y, bend_2Dhist_sem, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = distanceLabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True,
                       plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)

    # Symmetrize the above bending angle / relative orientation graph,
    # taking theta[theta > 0] - theta[theta < 0]
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + f'_bendAngle_v_orientation_small_{distance_file_string}_asymm' + '.' + outputFileNameExt
    else:
        outputFileName = None
    midXind = int((X.shape[0] - 1)/2.0)
    if np.abs(X[midXind, 0]) > 1e-6:
        print('"X" array is not centered at zero. Will not symmetrize.')
    else:
        bend_2Dhist_mean_symm = 0.5*(bend_2Dhist_mean[midXind:,:] - 
                                     np.flipud(bend_2Dhist_mean[:(midXind+1),:]))
        bend_2Dhist_sem_symm = np.sqrt(bend_2Dhist_sem[midXind:,:]**2 +
                                       np.flipud(bend_2Dhist_sem[:(midXind+1),:])**2)/np.sqrt(2)
        X_symm = X[midXind:,:]
        Y_symm = Y[midXind:,:]
        slice_2D_histogram(bend_2Dhist_mean_symm, X_symm, Y_symm,
                           bend_2Dhist_sem_symm, 
                           slice_axis = 'x', other_range = d_range, 
                           titleStr = titleStr, xlabelStr = f'|{xlabelStr}|', 
                           zlabelStr = zlabelStr + ' toward Other',
                           ylabelStr = distanceLabelStr, zlim = zlim, 
                           xlim = (0.0, xlim[1]), 
                           plot_z_zero_line = True,
                           plot_vert_zero_line = False,
                           unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                           color = color, outputFileName=outputFileName,
                           closeFigure=closeFigures)

    # Slice along bend angle binned by distance and orientation, 
    # orientation axis, constrain distance: 5 mm < distance < 15 mm
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + f'_bendAngle_v_orientation_middle_{distance_file_string}' + '.' + outputFileNameExt
    else:
        outputFileName = None
    d_range = (5.0, 15.0)
    xlabelStr = 'Relative Orientation (deg)'
    titleStr = f'{exptName}: Bend Angle for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
    zlabelStr = 'Mean Bending Angle (degrees)'
    xlim = (-np.pi, np.pi)
    zlim = (-15*np.pi/180, 15*np.pi/180)
    color = color
    slice_2D_histogram(bend_2Dhist_mean, X, Y, bend_2Dhist_sem, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = distanceLabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True,
                       plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)

    # Symmetrize the above bending angle / relative orientation graph,
    # taking theta[theta > 0] - theta[theta < 0]
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + f'_bendAngle_v_orientation_middle_{distance_file_string}_asymm' + '.' + outputFileNameExt
    else:
        outputFileName = None
    midXind = int((X.shape[0] - 1)/2.0)
    if np.abs(X[midXind, 0]) > 1e-6:
        print('"X" array is not centered at zero. Will not symmetrize.')
    else:
        bend_2Dhist_mean_symm = 0.5*(bend_2Dhist_mean[midXind:,:] - 
                                     np.flipud(bend_2Dhist_mean[:(midXind+1),:]))
        bend_2Dhist_sem_symm = np.sqrt(bend_2Dhist_sem[midXind:,:]**2 +
                                       np.flipud(bend_2Dhist_sem[:(midXind+1),:])**2)/np.sqrt(2)
        X_symm = X[midXind:,:]
        Y_symm = Y[midXind:,:]
        slice_2D_histogram(bend_2Dhist_mean_symm, X_symm, Y_symm,
                           bend_2Dhist_sem_symm, 
                           slice_axis = 'x', other_range = d_range, 
                           titleStr = titleStr, xlabelStr = f'|{xlabelStr}|', 
                           zlabelStr = zlabelStr + ' toward Other',
                           ylabelStr = distanceLabelStr, zlim = zlim, 
                           xlim = (0.0, xlim[1]), 
                           plot_z_zero_line = True,
                           plot_vert_zero_line = False,
                           unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                           color = color, outputFileName=outputFileName,
                           closeFigure=closeFigures)

    
    # 2D plot of mean bending angle vs. relative orientation and distance,
    # Constrained to bending angle > minimum threshold for "bending"
    # presumably params['bend_min_deg']
    if bending_threshold_deg > 0.0:
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + \
                f'_bendAngle_above{bending_threshold_deg:.0f}deg_{distance_file_string}_orientation_2D' + \
                    '.' + outputFileNameExt
        else:
            outputFileName = None
        mask_by_sem_limit_degrees = 8.0 # show points with s.e.m. < this
        use_abs_value = (False, False)
        titleStr = f'{exptName}: Bend Angle >{bending_threshold_deg:.0f}deg; unc. < {mask_by_sem_limit_degrees:.1f}deg'
        # Save the output 2D histograms, for use later.
        bend_2Dhist_mean, X, Y, bend_2Dhist_sem, _ = make_2D_histogram(
            datasets,
            keyNames = ('relative_orientation', f'{distance_type}_mm'),
            keyIdx = (None, None), 
            use_abs_value = use_abs_value,
            keyNameC = 'bend_angle', keyIdxC = None,
            colorRange = (-45*np.pi/180.0, 45*np.pi/180.0),
            dilate_minus1= False, 
            constraintKey = 'bend_angle', 
            constraintRange = ((np.pi/180.0)*bending_threshold_deg, np.inf), 
            constraintIdx = None, use_abs_value_constraint = True,
            bin_ranges = ((-np.pi, np.pi), (0.0, 30.0)), Nbins = (19,15), 
            titleStr = titleStr,
            clabelStr= 'Mean Bending Angle (degrees)',
            xlabelStr = 'Relative Orientation (degrees)',
            ylabelStr = distanceLabelStr, 
            mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
            unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
            cmap = 'RdYlBu_r', 
            plot_type = plot_type_2D,
            outputFileName = outputFileName,
            closeFigure = closeFigures)    
    
    return saved_pair_outputs


def make_turning_angle_plots(datasets, exptName = '', distance_type = None,
                             turning_threshold_deg = 0.0,
                             color = 'black',
                             plot_type_2D = 'heatmap',
                             Nbins = (19,15),
                             mask_by_sem_limit_degrees = 2.0,
                             colorRange = (-5.0*np.pi/180.0, 5.0*np.pi/180.0),
                             cmap = 'RdYlBu_r',
                             outputFileNameBase = 'turning_angle',
                             outputFileNameExt = 'png',
                             closeFigures = False,
                             outputCSVFileName = None,
                             makeSlicePlots = False):
    
    """
    Makes several useful plots of Turning angle properties for 
    of pairs of fish.
        
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        turning_threshold_deg : (float) for a plot of mean turning angle
                constrained to abs(angle) > threshold, use this threshold.
                Input in degrees. 
        distance_type : string, either closest_distance or head_head_distance
                Default is None; user should think about this!
        color: plot color (uses alpha for indiv. dataset colors)
        plot_type_2D : str, 'heatmap' or 'line_plots'
                    Which plotting function make_2D_histogram() will use
                    ('heatmap' or 'line_plots')
        Nbins : tuple, Number of bins in the rel orientation and distance axes
        mask_by_sem_limit_degrees : (float) only plot 2D-histogram bins whose
            s.e.m. is below this value (degrees); also used in the title. 2.0 by default.
        colorRange : (vmin, vmax) tuple for the heatmap color scale, in radians
            (converted to degrees for display). Default +/- 5 degrees.
        cmap : colormap to use for heatmap
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        outputCSVFileName : str or None ; send to make_2D_histogram(); if not None,
            output the 2D histogram values as a CSV file.
        makeSlicePlots : (bool) if true, make plots that are "slices" of the 2D histogram

    Outputs:
        saved_pair_outputs : list, containing
            0 : turn_2Dhist_mean, mean 2D turning angle histogram
            1 : turn_2Dhist_sem, standard error of the mean for 2D turning angle histogram
            2 : turn_2Dhist_std, std dev for 2D turning angle histogram
            3: bin positions ("X") for relative orientation for 2D turning angle histogram
            4: bin positions ("Y") for head_head_distance_mm for 2D turning angle histogram

    To do:
        Redundant code for slicing, symmetrization. Probably not worth cleaning up.
        
    """
    
    # Make sure of distance measure being used
    if not (distance_type==None or distance_type == 'closest_distance' or \
            distance_type == 'head_head_distance'):
        print('\nDistance measure must be "closest_distance" or "head_head_distance".\n')
        distance_type = None
    if distance_type == None:
        distance_type_choice = 0
        while not ((distance_type_choice  == 1) or (distance_type_choice  == 2)):
            distance_type_choice = int(input('Choose distance measure ' + 
                                             '\n  (1) closest_distance ' + 
                                             '\n  (2) head_head_distance' +
                                             '\nEnter "1" or "2": '))
        if distance_type_choice==1:
            distance_type = 'closest_distance'
        else:
            distance_type = 'head_head_distance'

    # Strings for file output, labels
    if distance_type == 'closest_distance':
        distance_file_string = 'closestDistance'
        distanceLabelStr = 'Closest Distance (mm)'
    elif distance_type == 'head_head_distance':
        distance_file_string = 'headHeadDistance'
        distanceLabelStr = 'Head-Head Distance (mm)'
    else:
        raise ValueError('Invalid distance type')
    
        
    saved_pair_outputs = []

    verifyPairs = True
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            verifyPairs = False
    if verifyPairs==False:
        raise ValueError('Error in make_turning_angle_plots; Nfish must be 2 !')

    # 2D plot of mean turning angle vs. relative orientation and distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + \
            f'_turnAngle_{distance_file_string}_orientation_2D' + \
                '.' + outputFileNameExt
    else:
        outputFileName = None
    use_abs_value = (False, False)
    titleStr = f'{exptName}: Turn Angle; unc. < {mask_by_sem_limit_degrees:.1f} deg'
    # Save the output 2D histograms, for use later.
    turn_2Dhist_mean, X, Y, turn_2Dhist_sem, turn_2Dhist_std = make_2D_histogram(
        datasets,
        keyNames = ('relative_orientation', f'{distance_type}_mm'),
        keyIdx = (None, None), 
        use_abs_value = use_abs_value,
        keyNameC = 'turning_angle_rad', keyIdxC = None,
        colorRange = colorRange,
        dilate_minus1= False,
        bin_ranges = ((-np.pi, np.pi), (0.0, 50.0)), Nbins = Nbins,
        titleStr = titleStr,
        clabelStr= 'Mean Turning Angle (degrees)',
        xlabelStr = 'Relative Orientation (degrees)',
        ylabelStr = distanceLabelStr, 
        mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
        unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
        cmap = cmap, 
        plot_type = plot_type_2D,
        outputFileName = outputFileName,
        closeFigure = closeFigures,
        outputCSVFileName = outputCSVFileName)
    saved_pair_outputs.append(turn_2Dhist_mean)
    saved_pair_outputs.append(turn_2Dhist_sem)
    saved_pair_outputs.append(turn_2Dhist_std)
    saved_pair_outputs.append(X)
    saved_pair_outputs.append(Y)

    if makeSlicePlots:
        # Slice turning angle binned by distance and orientation, along the
        # orientation axis, distance slice: distance < 5.0 mm
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + f'_turnAngle_v_orientation_small_{distance_file_string}' + '.' + outputFileNameExt
        else:
            outputFileName = None
        d_range = (0.0, 5.0)
        xlabelStr = 'Relative Orientation (deg)'
        titleStr = f'{exptName}: Turn Angle for d < {d_range[1]:.2f} mm'
        zlabelStr = 'Mean Turning Angle (degrees)'
        xlim = (-np.pi, np.pi)
        zlim = (-5*np.pi/180, 5*np.pi/180)
        color = color
        slice_2D_histogram(turn_2Dhist_mean, X, Y, turn_2Dhist_sem, 
                        slice_axis = 'x', other_range = d_range, 
                        titleStr = titleStr, xlabelStr = xlabelStr, 
                        zlabelStr = zlabelStr,
                        ylabelStr = distanceLabelStr, zlim = zlim, xlim = xlim, 
                        plot_z_zero_line = True,
                        plot_vert_zero_line = True,
                        unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                        color = color, outputFileName=outputFileName,
                        closeFigure=closeFigures)

        """
        # Symmetrize the above turning angle / relative orientation graph,
        # taking theta[theta > 0] - theta[theta < 0]
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + f'_turnAngle_v_orientation_small_{distance_file_string}_asymm' + '.' + outputFileNameExt
        else:
            outputFileName = None
        midXind = int((X.shape[0] - 1)/2.0)
        if np.abs(X[midXind, 0]) > 1e-6:
            print('"X" array is not centered at zero. Will not symmetrize.')
        else:
            turn_2Dhist_mean_symm = 0.5*(turn_2Dhist_mean[midXind:,:] - 
                                        np.flipud(turn_2Dhist_mean[:(midXind+1),:]))
            turn_2Dhist_sem_symm = np.sqrt(turn_2Dhist_sem[midXind:,:]**2 +
                                        np.flipud(turn_2Dhist_sem[:(midXind+1),:])**2)/np.sqrt(2)
            X_symm = X[midXind:,:]
            Y_symm = Y[midXind:,:]
            slice_2D_histogram(turn_2Dhist_mean_symm, X_symm, Y_symm,
                            turn_2Dhist_sem_symm, 
                            slice_axis = 'x', other_range = d_range, 
                            titleStr = titleStr, xlabelStr = f'|{xlabelStr}|', 
                            zlabelStr = zlabelStr + ' toward Other',
                            ylabelStr = distanceLabelStr, zlim = zlim, 
                            xlim = (0.0, xlim[1]), 
                            plot_z_zero_line = True,
                            plot_vert_zero_line = False,
                            unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                            color = color, outputFileName=outputFileName,
                            closeFigure=closeFigures)
        """
        
        # Slice along turn angle binned by distance and orientation, 
        # orientation axis, constrain distance: 5 mm < distance < 15 mm
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + f'_turnAngle_v_orientation_middle_{distance_file_string}' + '.' + outputFileNameExt
        else:
            outputFileName = None
        d_range = (5.0, 15.0)
        xlabelStr = 'Relative Orientation (deg)'
        titleStr = f'{exptName}: Turning Angle for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
        zlabelStr = 'Mean Turning Angle (degrees)'
        xlim = (-np.pi, np.pi)
        zlim = (-5*np.pi/180, 5*np.pi/180)
        color = color
        slice_2D_histogram(turn_2Dhist_mean, X, Y, turn_2Dhist_sem, 
                        slice_axis = 'x', other_range = d_range, 
                        titleStr = titleStr, xlabelStr = xlabelStr, 
                        zlabelStr = zlabelStr,
                        ylabelStr = distanceLabelStr, zlim = zlim, xlim = xlim, 
                        plot_z_zero_line = True,
                        plot_vert_zero_line = True,
                        unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                        color = color, outputFileName=outputFileName,
                        closeFigure=closeFigures)

        """
        # Symmetrize the above turning angle / relative orientation graph,
        # taking theta[theta > 0] - theta[theta < 0]
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + f'_turnAngle_v_orientation_middle_{distance_file_string}_asymm' + '.' + outputFileNameExt
        else:
            outputFileName = None
        midXind = int((X.shape[0] - 1)/2.0)
        if np.abs(X[midXind, 0]) > 1e-6:
            print('"X" array is not centered at zero. Will not symmetrize.')
        else:
            turn_2Dhist_mean_symm = 0.5*(turn_2Dhist_mean[midXind:,:] - 
                                        np.flipud(turn_2Dhist_mean[:(midXind+1),:]))
            turn_2Dhist_sem_symm = np.sqrt(turn_2Dhist_sem[midXind:,:]**2 +
                                        np.flipud(turn_2Dhist_sem[:(midXind+1),:])**2)/np.sqrt(2)
            X_symm = X[midXind:,:]
            Y_symm = Y[midXind:,:]
            slice_2D_histogram(turn_2Dhist_mean_symm, X_symm, Y_symm,
                            turn_2Dhist_sem_symm, 
                            slice_axis = 'x', other_range = d_range, 
                            titleStr = titleStr, xlabelStr = f'|{xlabelStr}|', 
                            zlabelStr = zlabelStr + ' toward Other',
                            ylabelStr = distanceLabelStr, zlim = zlim, 
                            xlim = (0.0, xlim[1]), 
                            plot_z_zero_line = True,
                            plot_vert_zero_line = False,
                            unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                            color = color, outputFileName=outputFileName,
                            closeFigure=closeFigures)
        """
    
    """
    # 2D plot of mean turning angle vs. relative orientation and distance,
    # Constrained to turning angle > minimum threshold for "turning"
    if turning_threshold_deg > 0.0:
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + \
                f'_turnAngle_above{turning_threshold_deg:.0f}deg_{distance_file_string}_orientation_2D' + \
                    '.' + outputFileNameExt
        else:
            outputFileName = None
        mask_by_sem_limit_degrees = 8.0 # show points with s.e.m. < this
        use_abs_value = (False, False)
        titleStr = f'{exptName}: Turn Angle >{turning_threshold_deg:.0f}deg; unc. < {mask_by_sem_limit_degrees:.1f}deg'
        # Save the output 2D histograms, for use later.
        turn_2Dhist_mean, X, Y, turn_2Dhist_sem, _ = make_2D_histogram(
            datasets,
            keyNames = ('relative_orientation', f'{distance_type}_mm'),
            keyIdx = (None, None), 
            use_abs_value = use_abs_value,
            keyNameC = 'turning_angle_rad', keyIdxC = None,
            colorRange = (-45*np.pi/180.0, 45*np.pi/180.0),
            dilate_minus1= False, 
            constraintKey = 'turning_angle_rad', 
            constraintRange = ((np.pi/180.0)*turning_threshold_deg, np.inf), 
            constraintIdx = None, use_abs_value_constraint = True,
            bin_ranges = ((-np.pi, np.pi), (0.0, 30.0)), Nbins = (19,15), 
            titleStr = titleStr,
            clabelStr= 'Mean Turning Angle (degrees)',
            xlabelStr = 'Relative Orientation (degrees)',
            ylabelStr = distanceLabelStr, 
            mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
            unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
            cmap = 'RdYlBu_r', 
            plot_type = plot_type_2D,
            outputFileName = outputFileName,
            closeFigure = closeFigures)    
    """
    
    return saved_pair_outputs


def make_interbout_turning_angle_plots(datasets, exptName = '',
                                       distance_type = 'head_head_distance',
                                       plot_type_2D = 'heatmap',
                                       Nbins = (19, 15),
                                       mask_by_sem_limit_degrees = 2.0,
                                       colorRange = (-5.0*np.pi/180.0, 5.0*np.pi/180.0),
                                       cmap = 'RdYlBu_r',
                                       outputFileNameBase = 'IBI_turning_angle',
                                       outputFileNameExt = 'png',
                                       closeFigures = False,
                                       outputCSVFileName = None):
    """
    Inter-bout-interval (IBI) analogue of make_turning_angle_plots().

    Plots the mean IBI-to-IBI turning angle (datasets[j]["IBI_properties"]
    ["turning_angle_IBI"]) binned by the IBI-level relative orientation and
    inter-fish distance. Pools the per-IBI arrays across fish and datasets, then
    bins/plots them via bin_and_plot_2D(). Requires Nfish==2 (uses the pair
    sub-keys of IBI_properties). Non-finite IBI entries are dropped before
    binning.

    Inputs
    ------
    datasets : list of dataset dictionaries, each with an "IBI_properties" key
               (from get_IBI_properties) containing the per-fish ragged arrays
               "turning_angle_IBI", "relative_orientation_mean", and
               "head_head_distance_mm_mean" / "closest_distance_mm_mean".
    exptName : experiment name, appended to titles
    distance_type : 'head_head_distance' or 'closest_distance'
    plot_type_2D : 'heatmap' or 'line_plots'
    Nbins : (n_relorient_bins, n_distance_bins)
    mask_by_sem_limit_degrees : only plot bins whose s.e.m. is below this value
                        (degrees); also used in the title. 2.0 by default.
    colorRange : (vmin, vmax) tuple for the heatmap color scale, in radians
                        (converted to degrees for display). Default +/- 5 degrees.
    cmap : colormap for the heatmap
    outputFileNameBase : base figure filename; None to skip saving
    outputFileNameExt : figure file extension (e.g. 'png', 'svg')
    closeFigures : if True, close the figure after creating it
    outputCSVFileName : if not None, write the 2D mean (and parallel '_unc'
                        s.e.m.) to CSV via bin_and_plot_2D()

    Outputs
    -------
    saved_pair_outputs : list, containing (matching make_turning_angle_plots)
        0 : turn_2Dhist_mean, mean IBI 2D turning-angle histogram
        1 : turn_2Dhist_sem, standard error of the mean in each bin
        2 : turn_2Dhist_std, standard deviation in each bin
        3 : X, bin-center positions for relative orientation (meshgrid)
        4 : Y, bin-center positions for distance (meshgrid)
    """
    # Require Nfish==2 (the pair sub-keys exist only then)
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            raise ValueError('Error in make_interbout_turning_angle_plots; '
                             'Nfish must be 2 !')

    if distance_type == 'closest_distance':
        distance_file_string = 'closestDistance'
        distanceLabelStr = 'Closest Distance (mm)'
    elif distance_type == 'head_head_distance':
        distance_file_string = 'headHeadDistance'
        distanceLabelStr = 'Head-Head Distance (mm)'
    else:
        raise ValueError('Invalid distance type')
    distance_key = f'{distance_type}_mm_mean'

    # Pool the per-IBI arrays across fish and datasets into flat 1D arrays
    turning_all = []
    rel_orient_all = []
    distance_all = []
    for dataset in datasets:
        ibi = dataset["IBI_properties"]
        for k in range(dataset["Nfish"]):
            turning_all.append(ibi["turning_angle_IBI"][k])
            rel_orient_all.append(ibi["relative_orientation_mean"][k])
            distance_all.append(ibi[distance_key][k])
    turning_all = np.concatenate(turning_all)
    rel_orient_all = np.concatenate(rel_orient_all)
    distance_all = np.concatenate(distance_all)

    # Drop any non-finite entries (e.g. an IBI with no good frames)
    finite = (np.isfinite(turning_all) & np.isfinite(rel_orient_all)
              & np.isfinite(distance_all))
    turning_all = turning_all[finite]
    rel_orient_all = rel_orient_all[finite]
    distance_all = distance_all[finite]

    # 2D plot of mean IBI turning angle vs. relative orientation and distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + \
            f'_IBIturnAngle_{distance_file_string}_orientation_2D' + \
            '.' + outputFileNameExt
    else:
        outputFileName = None
    titleStr = f'{exptName}: IBI Turn Angle; unc. < {mask_by_sem_limit_degrees:.1f} deg'

    turn_2Dhist_mean, X, Y, turn_2Dhist_sem, turn_2Dhist_std = bin_and_plot_2D(
        rel_orient_all, distance_all, valuesC_all=turning_all,
        bin_ranges=((-np.pi, np.pi), (0.0, 50.0)), Nbins=Nbins,
        titleStr=titleStr,
        clabelStr='Mean IBI Turning Angle (degrees)',
        xlabelStr='Relative Orientation (degrees)',
        ylabelStr=distanceLabelStr,
        colorRange=colorRange,
        unit_scaling_for_plot=[180.0/np.pi, 1.0, 180.0/np.pi],
        mask_by_sem_limit=mask_by_sem_limit_degrees*np.pi/180.0,
        cmap=cmap,
        plot_type=plot_type_2D,
        outputFileName=outputFileName,
        outputCSVFileName=outputCSVFileName,
        closeFigure=closeFigures)

    saved_pair_outputs = [turn_2Dhist_mean, turn_2Dhist_sem, turn_2Dhist_std,
                          X, Y]
    return saved_pair_outputs


def make_relative_orientation_plots(datasets, exptName = '',
                                    distance_type = None,
                             color = 'black',
                             plot_type_2D = 'heatmap',
                             cmap = 'RdYlBu_r', 
                             outputFileNameBase = 'rel_orient', 
                             outputFileNameExt = 'png',
                             closeFigures = False,
                             outputCSVFileName = None,
                             makeSlicePlots = False):
    
    """
    Makes useful plots of relative orientation properties for 
    of pairs of fish.
        
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        distance_type : string, either closest_distance or head_head_distance
                Default is None; user should think about this!
        color: plot color (uses alpha for indiv. dataset colors)
        plot_type_2D : str, 'heatmap' or 'line_plots'
                    Which plotting function make_2D_histogram() will use
                    ('heatmap' or 'line_plots')
        cmap : colormap to use for heatmap
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        writeCSVs : (bool) Used by various functions; if true, output plotted 
                            points to a CSV file. See code for filenames
        outputCSVFileName : str or None ; send to make_2D_histogram(); if not None,
                    output the 2D histogram values as a CSV file. 
        makeSlicePlots : (bool) if true, make plots that are "slices" of the 2D histogram

    Outputs:
        saved_pair_outputs : list, containing
            0 : relOrient_2Dhist_mean, mean 2D relative orientation histogram
            1 : relOrient_2Dhist_sem, s.e.m. for 2D relative orientation histogram
            2: bin positions ("X") for distance for 2D relative orientation histogram
            3: bin positions ("Y") for relative orientation for 2D relative orientation histogram

    To do:
        Redundant code for slicing, symmetrization. Probably not worth cleaning up.
        
    """
    
    # Make sure of distance measure being used
    if not (distance_type==None or distance_type == 'closest_distance' or \
            distance_type == 'head_head_distance'):
        print('\nDistance measure must be "closest_distance" or "head_head_distance".\n')
        distance_type = None
    if distance_type == None:
        distance_type_choice = 0
        while not ((distance_type_choice  == 1) or (distance_type_choice  == 2)):
            distance_type_choice = int(input('Choose distance measure ' + 
                                             '\n  (1) closest_distance ' + 
                                             '\n  (2) head_head_distance' +
                                             '\nEnter "1" or "2": '))
        if distance_type_choice==1:
            distance_type = 'closest_distance'
        else:
            distance_type = 'head_head_distance'

    # Strings for file output, labels
    if distance_type == 'closest_distance':
        distanceKey = 'closest_distance_mm'
        distanceStr = 'Closest Distance'
        distanceLabelStr = 'Closest Distance (mm)'
        shortDistanceStr = 'ClDist'
        distance_file_string = 'closestDistance'
    elif distance_type == 'head_head_distance':
        distanceKey = 'head_head_distance_mm'
        distanceStr = 'HH Distance'
        distanceLabelStr = 'Head-Head Distance (mm)'
        shortDistanceStr = 'HHDist'
        distance_file_string = 'headHeadDistance'
    else:
        raise ValueError('Invalide distance type')
        
    saved_pair_outputs = []

    
    verifyPairs = True
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            verifyPairs = False
    if verifyPairs==False:
        raise ValueError('Error in make_pair_fish_plots; Nfish must be 2 !')

    # 2D plot of mean turning angle vs. relative orientation and distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + \
            f'_relative_orientation_{shortDistanceStr}_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    titleStr = f'{exptName}: Relative Orientation'
    # Save the output 2D histograms, for use later.
    # use heatmap as plot type; for 2D histogram slicing along Y is not helpful
    relOrient_2Dhist_mean, X, Y, relOrient_2Dhist_sem, _ = make_2D_histogram(
        datasets, 
        keyNames = ('relative_orientation', distanceKey), 
        keyIdx = (None, None), 
        use_abs_value = (True, False),
        dilate_minus1=False, 
        bin_ranges=((0.0, 3.142), (0.0, 30.0)), 
        Nbins=(11,15), 
        unit_scaling_for_plot = [180.0/np.pi, 1.0, 1.0],
        xlabelStr = 'Rel. Orientation (deg)',
        ylabelStr = distanceLabelStr, 
        cmap = cmap,
        colorRange = (0.0, 0.0075),
        clabelStr = 'Probability',
        titleStr = f'{exptName}: abs(Rel. orient.) and {shortDistanceStr}', 
        plot_type = plot_type_2D,
        outputFileName = outputFileName,
        outputCSVFileName = outputCSVFileName,
        closeFigure = closeFigures)

    saved_pair_outputs.append(relOrient_2Dhist_mean)
    saved_pair_outputs.append(relOrient_2Dhist_sem)
    saved_pair_outputs.append(X)
    saved_pair_outputs.append(Y)

    if makeSlicePlots:
        # Slice turning angle binned by distance and orientation, along the
        # orientation axis, distance slice: distance < 5.0 mm
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + f'_P_rel_orientation_middle_{distance_file_string}' + '.' + outputFileNameExt
            outputFileName = outputFileNameBase + f'_P_rel_orientation_small_{distance_file_string}' + '.' + outputFileNameExt
        else:
            outputFileName = None
        d_range = (0.0, 5.0)
        xlabelStr = 'Relative Orientation (deg)'
        titleStr = f'{exptName}: P(Rel. Orientation) for d < {d_range[1]:.2f} mm'
        zlabelStr = 'Probability (not normalized)'
        xlim = (-np.pi, np.pi)
        zlim = (0.0, 0.02)
        color = color
        slice_2D_histogram(relOrient_2Dhist_mean, X, Y, relOrient_2Dhist_sem, 
                        slice_axis = 'x', other_range = d_range, 
                        titleStr = titleStr, xlabelStr = xlabelStr, 
                        zlabelStr = zlabelStr,
                        ylabelStr = distanceLabelStr, zlim = zlim, xlim = xlim, 
                        plot_z_zero_line = True,
                        plot_vert_zero_line = True,
                        unit_scaling_for_plot = [180.0/np.pi, 1.0, 1.0],
                        color = color, outputFileName=outputFileName,
                        closeFigure=closeFigures)

        
        # Slice along turn angle binned by distance and orientation, 
        # orientation axis, constrain distance: 5 mm < distance < 15 mm
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + f'_P_rel_orientation_middle_{distance_file_string}' + '.' + outputFileNameExt
        else:
            outputFileName = None
        d_range = (5.0, 15.0)
        xlabelStr = 'Relative Orientation (deg)'
        titleStr = f'{exptName}: P(Rel. Orientation) for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
        zlabelStr = 'Probability (not normalized)'
        xlim = (-np.pi, np.pi)
        zlim = (0.0, 0.02)
        color = color
        slice_2D_histogram(relOrient_2Dhist_mean, X, Y, relOrient_2Dhist_sem, 
                        slice_axis = 'x', other_range = d_range, 
                        titleStr = titleStr, xlabelStr = xlabelStr, 
                        zlabelStr = zlabelStr,
                        ylabelStr = distanceLabelStr, zlim = zlim, xlim = xlim, 
                        plot_z_zero_line = True,
                        plot_vert_zero_line = True,
                        unit_scaling_for_plot = [180.0/np.pi, 1.0, 1.0],
                        color = color, outputFileName=outputFileName,
                        closeFigure=closeFigures)

    return saved_pair_outputs


def make_single_fish_plots(datasets, exptName = '', color = 'black',
                           outputFileNameBase = 'single_fish',
                           outputFileNameExt = 'png',
                           closeFigures = False,
                           writeCSVs = False):
    """
    makes several useful "single fish" plots -- i.e. 
    plots of characteristics of individual fish, which may be in multi-fish 
    experiments
    Note that there are lots of parameter values that are hard-coded; this
    function is probably more useful to read than to run, pasting and 
    modifying its code.
    
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        color: plot color (uses alpha for indiv. dataset colors)
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        writeCSVs : (bool) Used by various functions; if true, output plotted 
                            points to a CSV file. See code for filenames

    Outputs:

    """
    
    # Speed histogram
    speeds_mm_s_all = combine_all_values_constrained(datasets, 
                                                     keyName='speed_array_mm_s', 
                                                     dilate_minus1 = True)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_speed', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    plot_probability_distr(speeds_mm_s_all, bin_width = 1.0, 
                           bin_range = [0, None], 
                           ylim = (0.001, 0.5), xlim = (0.0, 60.0),
                           color = color,
                           yScaleType = 'log',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           xlabelStr = 'Speed (mm/s)', 
                           titleStr = f'{exptName}: Probability Distr.: Speed',
                           outputFileName = outputFileName,
                           closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)

    # Angular_speed histogram
    angular_speeds_rad_s_all = combine_all_values_constrained(datasets, 
                                                     keyName='angular_speed_array_rad_s', 
                                                     dilate_minus1 = True)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_angularSpeed', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    plot_probability_distr(angular_speeds_rad_s_all, bin_width = 1.0, 
                           bin_range = [0, None], 
                           color = color, 
                           yScaleType = 'log',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           ylim = (0.001, 0.5), xlim = (0.0, 40.0),
                           xlabelStr = 'Angular Speed (rad/s)', 
                           titleStr = f'{exptName}: Probability Distr.: Angular Speed',
                           outputFileName = outputFileName,
                           closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)


    # Radial position histogram
    radial_position_mm_all = combine_all_values_constrained(datasets, 
                                                     keyName='radial_position_mm', 
                                                     dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_radialpos', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    plot_probability_distr(radial_position_mm_all, bin_width = 0.5, 
                           bin_range = [0, None],
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           normalize_by_inv_bincenter = True,
                           ylim = (-0.025, 0.5),
                           xlabelStr = 'Radial position (mm)', 
                           titleStr = f'{exptName}: Probability Distr.: r',
                           outputFileName = outputFileName,
                           closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)
    
    # Heading angle histogram
    heading_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='heading_angle', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_heading_angle', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/30
    plot_probability_distr(heading_angle_all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           polarPlot = True,
                           color = color,
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           titleStr = f'{exptName}: Heading Angle',
                           ylim = (0, 0.3),
                           unit_scaling_for_plot = 1.0,
                           outputFileName = outputFileName,
                           closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)
    
    # Radial alignment angle
    radial_alignment_all = combine_all_values_constrained(datasets, 
                                                     keyName='radial_alignment_rad', 
                                                     dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_radialAlignment_angle', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/30
    plot_probability_distr(radial_alignment_all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           polarPlot = True,
                           color = color,
                           titleStr = f'{exptName}: Radial alignment angle (rad)',
                           ylim = (0, 0.6),
                           outputFileName = outputFileName,
                           closeFigure=closeFigures,
                           outputCSVFileName=outputCSVFileName)

    # Speed vs. time for bouts
    # Note that this doesn't support CSV output
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_boutSpeed', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs=False)
    average_bout_trajectory_allSets(datasets, keyName = "speed_array_mm_s", 
                                    keyIdx = None, t_range_s=(-1.0, 2.0), 
                                    titleStr = f'{exptName}: Bout Speed', 
                                    makePlot=True,
                                    color = color,
                                    outputFileName = outputFileName,
                                    closeFigure=closeFigures)

    # speed autocorrelation function
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_speedAutocorr', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs=False)
    speed_ac_all, t_lag = \
        calculate_value_corr_all(datasets, keyName = 'speed_array_mm_s',
                                 corr_type='auto', dilate_minus1 = True, 
                                 t_max = 3.0, t_window = 10.0, fpstol = 1e-6)
    plot_function_allSets(speed_ac_all, t_lag, xlabelStr='time (s)', 
                          ylabelStr='Speed autocorrelation', 
                          titleStr = f'{exptName}: Speed autocorrelation', 
                          color = color,
                          xlim = (-0.1, 2.0),
                          average_in_dataset = True,
                          outputFileName = outputFileName,
                          closeFigure=closeFigures,
                          outputCSVFileName=outputCSVFileName)


def bin_and_plot_2D(values1_all, values2_all, valuesC_all=None,
                    bin_ranges=None, Nbins=(20, 20),
                    titleStr=None, clabelStr=None,
                    xlabelStr='x', ylabelStr='y',
                    colorRange=None,
                    unit_scaling_for_plot=[1.0, 1.0, 1.0],
                    mask_by_sem_limit=None,
                    cmap='RdYlBu',
                    plot_type='heatmap',
                    outputFileName=None,
                    outputCSVFileName=None,
                    closeFigure=False):
    """
    Bin pre-pooled 1D value arrays into a 2D histogram, optionally write the
    result to CSV, and plot it (heatmap or line plots).

    This is the binning/plotting back-end shared by make_2D_histogram() (which
    builds the inputs from per-frame dataset keys) and by callers that already
    have pooled 1D arrays, e.g. per-IBI properties
    (make_interbout_turning_angle_plots()).

    If valuesC_all is None, computes a normalized occurrence histogram of
    (values1, values2). Otherwise computes the mean of valuesC in each
    (value1, value2) bin, plus its standard deviation and s.e.m.

    Parameters
    ----------
    values1_all, values2_all : 1D arrays of equal length (the two binning axes)
    valuesC_all : 1D array of the same length, or None. The quantity to average
                  in each bin; None for an occurrence histogram.
    bin_ranges : ((v1_min, v1_max), (v2_min, v2_max)) or None (auto from data)
    Nbins : (n_bins_axis1, n_bins_axis2)
    titleStr, clabelStr, xlabelStr, ylabelStr : plot labels
    colorRange : (vmin, vmax) or None
    unit_scaling_for_plot : [x, y, C] display scaling (e.g. radians->degrees)
    mask_by_sem_limit : float or None; mask bins whose s.e.m. exceeds this
    cmap : colormap name
    plot_type : 'heatmap' or 'line_plots'
    outputFileName : figure filename, or None
    outputCSVFileName : if not None, write the mean to this CSV (X as row index,
                        Y as columns) and the s.e.m. to a parallel '_unc' file
    closeFigure : if True, close the figure after creating it

    Returns
    -------
    hist : 2D array (occurrence histogram, or mean of valuesC per bin)
    X, Y : 2D meshgrid arrays of bin-center coordinates
    hist_sem : 2D array, s.e.m. per bin (None if valuesC_all is None)
    hist_std : 2D array, std dev per bin (None if valuesC_all is None)
    """
    # Determine the bin ranges
    if bin_ranges is None:
        value1_min, value1_max = np.nanmin(values1_all), np.nanmax(values1_all)
        value2_min, value2_max = np.nanmin(values2_all), np.nanmax(values2_all)
        # Expand a bit!
        d1 = (value1_max - value1_min)/Nbins[0]
        d2 = (value2_max - value2_min)/Nbins[1]
        value1_min = value1_min - d1/2.0
        value1_max = value1_max + d1/2.0
        value2_min = value2_min - d2/2.0
        value2_max = value2_max + d2/2.0
    else:
        value1_min, value1_max = bin_ranges[0]
        value2_min, value2_max = bin_ranges[1]

    if valuesC_all is None:
        # normalized occurrence histogram
        hist, xedges, yedges = np.histogram2d(values1_all, values2_all,
                                              bins=Nbins,
                                              range=[(value1_min, value1_max),
                                                     (value2_min, value2_max)])
        hist = hist / hist.sum()
        if clabelStr is None:
            clabelStr = 'Normalized Count'
        hist_std = None
        hist_sem = None
    else:
        # mean (and std, count -> s.e.m.) of valuesC in each bin
        hist, xedges, yedges, _ = binned_statistic_2d(
            values1_all, values2_all, valuesC_all,
            statistic='mean', bins=Nbins,
            range=[(value1_min, value1_max), (value2_min, value2_max)])
        hist_std, _, _, _ = binned_statistic_2d(
            values1_all, values2_all, valuesC_all,
            statistic='std', bins=Nbins,
            range=[(value1_min, value1_max), (value2_min, value2_max)])
        hist_count, _, _, _ = binned_statistic_2d(
            values1_all, values2_all, valuesC_all,
            statistic='count', bins=Nbins,
            range=[(value1_min, value1_max), (value2_min, value2_max)])
        hist_sem = hist_std / np.sqrt(hist_count)
        if clabelStr is None:
            clabelStr = 'Mean value'

    # X and Y bin-center meshgrid
    X, Y = np.meshgrid(0.5*(xedges[1:] + xedges[:-1]),
                       0.5*(yedges[1:] + yedges[:-1]), indexing='ij')

    if outputCSVFileName is not None:
        x_vals = X[:, 0]
        y_vals = Y[0, :]
        pd.DataFrame(hist, index=x_vals, columns=y_vals).to_csv(outputCSVFileName)
        if hist_sem is not None:
            base, ext = os.path.splitext(outputCSVFileName)
            pd.DataFrame(hist_sem, index=x_vals, columns=y_vals).to_csv(
                base + '_unc' + ext)

    if titleStr is None:
        titleStr = '2D histogram'

    # Choose plotting function based on plot_type
    # For line_plots, ignore the colorRange
    if plot_type.lower() == 'line_plots':
        plot_2Darray_linePlots(hist, X, Y, Z_unc=hist_sem,
                              titleStr=titleStr,
                              xlabelStr=xlabelStr, ylabelStr=ylabelStr,
                              clabelStr=clabelStr,
                              colorRange=None, cmap=cmap,
                              unit_scaling_for_plot=unit_scaling_for_plot,
                              mask_by_sem_limit=mask_by_sem_limit,
                              outputFileName=outputFileName,
                              closeFigure=closeFigure)
    else:  # default to heatmap
        plot_2D_heatmap(hist, X, Y, Z_unc=hist_sem,
                       titleStr=titleStr,
                       xlabelStr=xlabelStr, ylabelStr=ylabelStr,
                       clabelStr=clabelStr,
                       colorRange=colorRange, cmap=cmap,
                       unit_scaling_for_plot=unit_scaling_for_plot,
                       mask_by_sem_limit=mask_by_sem_limit,
                       outputFileName=outputFileName, closeFigure=closeFigure)

    return hist, X, Y, hist_sem, hist_std


def make_2D_histogram(datasets,
                      keyNames = ('speed_array_mm_s', 'head_head_distance_mm'), 
                      keyIdx = (None, None), 
                      use_abs_value = (False, False),
                      keyNameC = None, keyIdxC = None,
                      constraintKey=None, constraintRange=None, 
                      constraintIdx = 0,
                      use_abs_value_constraint = False, 
                      dilate_minus1=True, bin_ranges=None, Nbins=(20,20),
                      titleStr = None, clabelStr = None,
                      xlabelStr = None, ylabelStr = None,
                      colorRange = None, 
                      unit_scaling_for_plot = [1.0, 1.0, 1.0],
                      mask_by_sem_limit = None,
                      cmap = 'RdYlBu', 
                      plot_type = 'heatmap',
                      outputFileName = None,
                      outputCSVFileName = None,
                      closeFigure = False):
    """
    Create a 2D histogram plot of the values from two keys in the given 
    datasets. Combine all the values across datasets. Or can plot the mean 
    value of a quantitative property (third key) binned by these two keys.

    If keyNameC is None: plots a normalized 2D histogram (occurrence count)
    If keyNameC is provided: plots mean value of keyC in each (keyA, keyB) bin

    Uses combine_all_values_constrained() to pick a subset of the keys or to 
    apply a constraint to the values to plot (optional)
    
    Parameters:
    datasets (list): List of dictionaries containing the analysis data.
    keyNames (tuple of 2 str): Key names for the first and second values to plot.
    keyIdx : tuple of 2 items, integer or string or None indicating subset of 
             value array, using get_values_subset(datasets[j][keyName], keyIdx)
            If keyIdx is:
                None: If datasets[j][keyName] is a multidimensional array, 
                   return the full array (minus bad frames, constraints)
                an integer: extract that column
                   (e.g. datasets[12]["speed_array_mm_s"][:,keyIdx])
                    the string "phi_low" or "phi_high", call get_keyIdx_array()
                       to get an array of integers corresponding to the index
                       of the low or high relative orientation fish
                    a string "min", "max", or "mean", apply this
                       operation along axis==1 (e.g. for fastest fish)
    use_abs_value : (bool, bool) default False, False (for each property)
                    If True, return absolute value of the quantitative 
                    property. Probably should always be false
    keyNameC (str or None): Key name for the third variable whose mean will be
                           plotted as a heatmap. If None, plots occurrence histogram.
    keyIdxC : integer or string or None, same indexing as keyIdx but for keyNameC
    constraintKey (str): Key name for the constraint, 
        or None to use no constraint. Apply the same constraint to both keys.
        see combine_all_values_constrained()
    constraintRange (np.ndarray): Numpy array with two elements specifying the constraint range, or None to use no constraint.
    constraintIdx : integer or string.
                    If the constraint is a multidimensional array, will use
                    dimension constraintIdx if constraintIdx is an integer 
                    or the "operation" if constraintIdx is a string,
                    "min", "max", or "mean",
                    If None, won't apply constraint    
                    see combine_all_values_constrained()
    use_abs_value_constraint : bool, default False
                    If True, use absolute value of the quantitative 
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).    
    dilate_minus1 (bool): If True, dilate the bad frames -1; see above.
    bin_ranges (tuple): Optional tuple of two lists, specifying the (min, max) range for the bins of value1 and value2.
    Nbins (tuple): Optional tuple of two integers, number of bins for value1 and value2
    titleStr : title string; if None use Key names
    clabelStr : string for color bar label. If None use 'Normalized Count' 
                or Mean {keyNameC} 
    xlabelStr, ylabelStr : x and y axis labels. If None, use key names
    colorRange : Optional tuple of (vmin, vmax) for the histogram "v axis" range
    unit_scaling_for_plot : List of None or float for x, y, optional "C"; 
                   for plots only, multiply values by unit_scaling_for_plot,  
                   for example to convert radians to degrees.
                   if keyNameC is None, ignore 3rd item
    mask_by_sem_limit : float or None. 
                   If not None, and if keyNameC is not None, only plot 
                   the 2D mesh at points whose s.e.m. of the third variable
                   is less than this value, to ignore noisy points. 
    cmap : string, colormap. Default 'RdYlBu' (rather than usual Python viridis)
    plot_type : str, 'heatmap' or 'line_plots'
                   Determines which plotting function to use.
    outputFileName : if not None, save the figure with this filename
                     (include extension)
    outputCSVFileName : str or None
        If not None, write hist as a wide-format CSV with X axis values as
        the row index and Y axis values as column headers. If hist_sem is
        also available (keyNameC is not None), write it to a parallel file
        with '_unc' inserted before the extension. Written before the plot,
        regardless of plot_type.
    closeFigure : (bool) if True, close figure after creating it.

    Returns:
        hist : 2D array, normalized histogram if keyNameC is None; mean values
                in each bin if keyNameC is used
        X, Y : 2D array of X, Y values from meshgrid
        hist_sem : 2D array, std error of the mean in each bin if keyNameC
                   is used; else None
        hist_std : 2D array, std deviation in each bin if keyNameC
                   is used; else None
    """
    if len(keyNames) != 2:
        raise ValueError("There must be two keys for the 2D histogram!") 
        
    # For scaling values for plot only, e.g. radians to degrees.
    if len(unit_scaling_for_plot) == 2:
        unit_scaling_for_plot.append(1.0)
    if (keyNameC is None) and (abs(unit_scaling_for_plot[2]-1.0)<1e-6):
        print('No keyNameC and unit_scaling_for_plot[2] ' + \
              f'is {unit_scaling_for_plot[2]:.6e} is not 1; forcing equal to 1.0')
        unit_scaling_for_plot[2] = 1.0
        
    # Get the values for each key with the constraint applied
    values1 = combine_all_values_constrained(datasets, keyNames[0], 
                                             keyIdx=keyIdx[0],
                                             use_abs_value = use_abs_value[0],
                                             constraintKey=constraintKey, 
                                             constraintRange=constraintRange, 
                                             constraintIdx=constraintIdx,
                                             use_abs_value_constraint = use_abs_value_constraint,
                                             dilate_minus1=dilate_minus1)
    values2 = combine_all_values_constrained(datasets, keyNames[1], 
                                             keyIdx=keyIdx[1],
                                             use_abs_value = use_abs_value[1],
                                             constraintKey=constraintKey, 
                                             constraintRange=constraintRange, 
                                             constraintIdx=constraintIdx,
                                             use_abs_value_constraint = use_abs_value_constraint,
                                             dilate_minus1 = dilate_minus1)
    
    # Get values for keyC if provided
    if keyNameC is not None:
        valuesC = combine_all_values_constrained(datasets, keyNameC, 
                                                 keyIdx=keyIdxC,
                                                 use_abs_value = False,
                                                 constraintKey=constraintKey, 
                                                 constraintRange=constraintRange, 
                                                 constraintIdx=constraintIdx,
                                                 use_abs_value_constraint = use_abs_value_constraint,
                                                 dilate_minus1=dilate_minus1)
    
    # Flatten the values and handle different dimensions
    values1_all = []
    values2_all = []
    valuesC_all = [] if keyNameC is not None else None
    
    for idx, (v1, v2) in enumerate(zip(values1, values2)):
        M1 = get_effective_dims(v1)
        M2 = get_effective_dims(v2)
        
        if keyNameC is not None:
            vC = valuesC[idx]
            MC = get_effective_dims(vC)
        
        if M1 is None or M2 is None or ((M1 != M2) and min(M1, M2) > 1):
            print(f'M values: {M1}, {M2}')
            raise ValueError("Values for the two keys are not commensurate. 2D histogram cannot be created.")
        
        if keyNameC is not None and (MC is None or (MC != M1 and MC != M2 and MC != 1)):
            print(f'M values: {M1}, {M2}, {MC}')
            raise ValueError("Values for keyC are not commensurate with keyA and keyB.")
        
        if M1 > 1 and M2 == 1:
            Nfish = M1
            values2_all.append(np.repeat(v2.flatten(), Nfish))
            values1_all.append(v1.flatten())
            if keyNameC is not None:
                if MC > 1:
                    valuesC_all.append(vC.flatten())
                else:
                    valuesC_all.append(np.repeat(vC.flatten(), Nfish))
        elif M2 > 1 and M1 == 1:
            Nfish = M2
            values1_all.append(np.repeat(v1.flatten(), Nfish))
            values2_all.append(v2.flatten())
            if keyNameC is not None:
                if MC > 1:
                    valuesC_all.append(vC.flatten())
                else:
                    valuesC_all.append(np.repeat(vC.flatten(), Nfish))
        else:
            values1_all.append(v1.flatten())
            values2_all.append(v2.flatten())
            if keyNameC is not None:
                valuesC_all.append(vC.flatten())
    
    # Concatenate the flattened values
    values1_all = np.concatenate(values1_all)
    values2_all = np.concatenate(values2_all)
    if keyNameC is not None:
        valuesC_all = np.concatenate(valuesC_all)
    
    # Resolve default labels that depend on the key names, then delegate the
    # binning, optional CSV export, and plotting to bin_and_plot_2D().
    if xlabelStr is None:
        xlabelStr = keyNames[0]
    if ylabelStr is None:
        ylabelStr = keyNames[1]
    if clabelStr is None:
        clabelStr = 'Normalized Count' if keyNameC is None else f'Mean {keyNameC}'
    if titleStr is None:
        if keyNameC is None:
            titleStr = f'2D Histogram of {keyNames[0]} vs {keyNames[1]}'
        else:
            titleStr = f'Mean {keyNameC} vs {keyNames[0]} and {keyNames[1]}'

    return bin_and_plot_2D(values1_all, values2_all, valuesC_all=valuesC_all,
                           bin_ranges=bin_ranges, Nbins=Nbins,
                           titleStr=titleStr, clabelStr=clabelStr,
                           xlabelStr=xlabelStr, ylabelStr=ylabelStr,
                           colorRange=colorRange,
                           unit_scaling_for_plot=unit_scaling_for_plot,
                           mask_by_sem_limit=mask_by_sem_limit,
                           cmap=cmap, plot_type=plot_type,
                           outputFileName=outputFileName,
                           outputCSVFileName=outputCSVFileName,
                           closeFigure=closeFigure)


def slice_2D_histogram(z_mean, X, Y, z_unc, slice_axis='x', other_range=None, 
                       titleStr = None, 
                       xlabelStr = 'x', ylabelStr = 'y', zlabelStr = 'z',
                       xlim = None, ylim = None, zlim = None,
                       plot_z_zero_line = False, plot_vert_zero_line = False,
                       unit_scaling_for_plot = [1.0, 1.0, 1.0],
                       color = 'black', 
                       outputFileName = None,
                       closeFigure = False):
    """
    Perform weighted average of 2D data "z" along one axis 
    (either x or y), possibly limited to some range along the other axis,
    and make an errorbar plot.
    Ignores NaNs in sums, averages; treats std. dev. = 0 as NaN
    
    Parameters:
    -----------
    z_mean : 2D numpy array
        Mean values at each (x, y) position
    X : 2D numpy array
        X-coordinates from meshgrid
    Y : 2D numpy array
        Y-coordinates from meshgrid
    z_unc : 2D numpy array
        Uncertainty values at each (x, y) position
        Note that if these are mean values, should use s.e.m. for proper weighting
    slice_axis : str, either 'x' or 'y'
        Axis along which to plot the slice
    other_range : tuple or None
        If not None, (min, max) range for the axis being averaged over
    titleStr : string or None
        If not None, title string; If None, "Slice along {whatever} axis"
    xlabelStr, ylabelStr, zlabelStr : string for x axis label, y, z.
        Note that only two of these will be used on the axis, the other in
        the title (if slicing)
    plot_z_zero_line : bool . if true, dotted line at z = 0
    plot_vert_zero_line : bool . if true, dotted line at (x or y) = 0
    unit_scaling_for_plot : List of None or float for x, y, z; 
                   for plots only, multiply values by unit_scaling_for_plot,  
                   for example to convert radians to degrees.
                   Note that one of x or y will be irrelevant, but I include 
                   all three anyway
    color: plot color
    xlim, ylim, zlim : (optional) tuple of min, max {x,y,z}-axis limits
        Note that only two of these will be used.
        Note that this should not incorporate unit_scaling_for_plot -- 
        if angles are in radians, for example, leave the limits in radians 
        and unit_scaling_for_plot will convert values *and* axis limits
        to degrees
    outputFileName : if not None, save the figure with this filename 
                     (include extension)
    closeFigure : (bool) if True, close a figure after creating it.
    
    Returns:
    --------
    axis_values : 1D numpy array
        Values along the slice axis (x or y values)
    weighted_mean : 1D numpy array
        Weighted mean at each position along slice axis
    weighted_std : 1D numpy array
        Weighted standard deviation at each position along slice axis
    """
    
    # Calculate weights (inverse variance weighting)
    if z_unc is not None:
        weights = 1.0 / (z_unc ** 2)
    else:
        weights = np.ones_like(z_mean)
    
    # Handle inf values (where z_unc= 0)
    weights = np.where(np.isfinite(weights), weights, np.nan)

    # For the plot    
    if zlabelStr is None:
        for_plot_ylabelStr = 'z (weighted mean)'
    else:
        for_plot_ylabelStr = zlabelStr

    if slice_axis == 'x':
        # Average over y, plot along x
        axis_values = X[:, 0]  # x values
        
        # Determine which y indices to include
        if other_range is not None:
            y_values = Y[0, :]
            mask = (y_values >= other_range[0]) & (y_values <= other_range[1])
        else:
            mask = np.ones(Y.shape[1], dtype=bool)
        
        # Apply mask to data
        z_mean_masked = z_mean[:, mask]
        weights_masked = weights[:, mask]
        
        if z_mean_masked.shape[1] == 1:
            # Only one "bin" along other axis
            weighted_mean = z_mean_masked.flatten()
            weighted_std = z_unc[:, mask].flatten()
        else:
            # Calculate weighted mean along y-axis (axis=1)
            sum_weights = np.nansum(weights_masked, axis=1)
            weighted_mean = np.nansum(weights_masked * z_mean_masked, axis=1) / sum_weights
            
            # Calculate weighted standard deviation
            # Formula: sqrt(sum(w_i * (x_i - mu)^2) / sum(w_i))
            weighted_variance = np.nansum(weights_masked * (z_mean_masked - weighted_mean[:, np.newaxis])**2, axis=1) / sum_weights
            weighted_std = np.sqrt(weighted_variance)
        
        # For labels
        if xlabelStr is None:
            for_plot_xlabelStr = 'x'
        else:
            for_plot_xlabelStr = xlabelStr
        if ylabelStr is None:
            slice_label = 'y'
        else:
            slice_label = 'ylabelStr'
        if titleStr is None:
            titleStr = f'Slice along {for_plot_xlabelStr}-axis' + \
                     (f' ({slice_label} ∈ [{other_range[0]}, {other_range[1]}])' \
                      if other_range else '')
        # For plot limits
        if xlim is None:
            for_plot_xlim = None
        else:
            for_plot_xlim = xlim
            
        # for scaling x or y scale (converting units)
        unit_scaling_for_plot_xy = unit_scaling_for_plot[0]
        
    elif slice_axis == 'y':
        # Average over x, plot along y
        axis_values = Y[0, :]  # y values
        
        # Determine which x indices to include
        if other_range is not None:
            x_values = X[:, 0]
            mask = (x_values >= other_range[0]) & (x_values <= other_range[1])
        else:
            mask = np.ones(X.shape[0], dtype=bool)
        
        # Apply mask to data
        z_mean_masked = z_mean[mask, :]
        weights_masked = weights[mask, :]

        if z_mean_masked.shape[0] == 1:
            # Only one "bin" along other axis
            weighted_mean = z_mean_masked.flatten()
            weighted_std = z_unc[mask, :].flatten()
        else:
            # Calculate weighted mean along x-axis (axis=0)
            sum_weights = np.sum(weights_masked, axis=0)
            weighted_mean = np.sum(weights_masked * z_mean_masked, axis=0) / sum_weights
            
            # Calculate weighted standard deviation
            weighted_variance = np.sum(weights_masked * (z_mean_masked - weighted_mean[np.newaxis, :])**2, axis=0) / sum_weights
            weighted_std = np.sqrt(weighted_variance)
        
        # For labels
        if xlabelStr is None:
            for_plot_xlabelStr = 'y'
        else:
            for_plot_xlabelStr = ylabelStr
        if ylabelStr is None:
            slice_label = 'x'
        else:
            slice_label = 'xlabelStr'
        if titleStr is None:
            titleStr = f'Slice along {for_plot_xlabelStr}-axis' + \
                     (f' ({slice_label} ∈ [{other_range[0]}, {other_range[1]}])' \
                      if other_range else '')

        # For plot limits
        if ylim is None:
            for_plot_xlim = None
        else:
            for_plot_xlim = ylim
        # for scaling x or y scale (converting units)
        unit_scaling_for_plot_xy = unit_scaling_for_plot[1]
        
    else:
        raise ValueError("slice_axis must be either 'x' or 'y'")

    # Plot
    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(axis_values*unit_scaling_for_plot_xy, 
                 weighted_mean*unit_scaling_for_plot[2], 
                 yerr=weighted_std*unit_scaling_for_plot[2], 
                 fmt='o-', capsize=5, capthick=2, markersize=12,
                 linewidth = 2, color = color)
    plt.xlabel(for_plot_xlabelStr, fontsize=14)
    plt.ylabel(for_plot_ylabelStr, fontsize=14)
    plt.title(titleStr, fontsize=16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    if xlim is not None:
        plt.xlim(tuple([item * unit_scaling_for_plot_xy for item in for_plot_xlim]))
    if zlim is not None:
        plt.ylim(tuple([item * unit_scaling_for_plot[2] for item in zlim]))
    if plot_z_zero_line:
        plt.axhline(y=0.0, color='black', alpha = 0.7, linestyle=':')
    if plot_vert_zero_line:
        plt.axvline(x=0.0, color='black', alpha = 0.7, linestyle=':')

    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')

    plt.show()
    if closeFigure:
        print(f'Closing figure {titleStr}')
        plt.close(fig)
    
    return axis_values, weighted_mean, weighted_std


