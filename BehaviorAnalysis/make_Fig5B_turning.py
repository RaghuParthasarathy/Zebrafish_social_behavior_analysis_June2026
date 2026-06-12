# -*- coding: utf-8 -*-
# make_Fig5B_turning.py

"""
Author:   Raghuveer Parthasarathy
Created on April 15, 2026
Last modified June 11, 2026 -- Raghuveer Parthasarathy

Description
-----------

For making "Figure 5B", turning angle v distance and relative orientation

Can either plot the frame-by-frame mean turning angle, or the inter-bout-interval
(IBI) to IBI mean turning angle. Toggle with `which_turning_plot` in main().

Based on compare_all_2wkData.py
Calls load_all_expt_data() from compare_all_2wkData.py
to load pickle files of two week old zebrafish behavior data.
Then calls other functions to make plots.

Datasets and colors:
    Pairs, Light (color darkorange)
    Pairs, Light, Time-shifted Fish0 (color saddlebrown)
    Pairs, Dark (color cornflowerblue)
    Pairs, Dark, Time-shifted Fish0 (color darkblue)
    Single, Light (gold)
    Single, Dark (slategrey)


Instructions:

Modify the expBaseStr et., for each dataset of interest.

"""

import os
import numpy as np
from behavior_plots import make_turning_angle_plots, \
    make_interbout_turning_angle_plots, slice_2D_histogram
from IO_toolkit import plot_2D_heatmap

from compare_all_2wkData import load_all_expt_data


def make_turning_difference_plots(all_expts, comparisons, distance_type,
                                  cmap, xlabelStr, outputExtension,
                                  keyString='turn', optionString='Turning',
                                  mask_by_sem_limit_degrees=2.0,
                                  colorRange = (-8.0*np.pi/180.0, 8.0*np.pi/180.0),
                                  makeSlicePlots=False):
    """
    Make between-condition difference heatmaps (and optionally 1D slices) of the
    2D turning-angle histograms stored in all_expts. Shared by the frame-by-frame
    and inter-bout turning-angle paths.

    Assumes each experiment in all_expts already has the keys
    "{keyString}_2Dhist_mean", "{keyString}_2Dhist_sem",
    "{keyString}_2Dhist_X", "{keyString}_2Dhist_Y"
    (as written by make_turning_angle_plots / make_interbout_turning_angle_plots).
    The (X, Y) bin-center mesh is assumed identical across experiments.

    Inputs
    ------
    all_expts : dict of experiment data, keyed by experiment name.
    comparisons : list of tuples
        (title_label, filename_label, exptKey_A, exptKey_B, slice_color);
        each plots experiment A minus experiment B.
    distance_type : 'closest_distance' or 'head_head_distance' (for the y label).
    cmap : colormap.
    xlabelStr : x-axis label (relative orientation).
    outputExtension : figure file extension, including the dot (e.g. '.svg').
    keyString : prefix of the stored histogram keys (default 'turn').
    optionString : label used in titles and output filenames (default 'Turning').
    mask_by_sem_limit_degrees : mask bins whose s.e.m. exceeds this (degrees);
        None to show all bins.
    colorRange : (vmin, vmax) color scale (radians) for the difference
        histograms. 
    makeSlicePlots : if True, also make 1D slices of the difference along
        distance ranges (not used for the IBI path).
    """
    if distance_type == 'closest_distance':
        distanceStr = 'Closest Distance'
        ylabelStr = 'Closest Distance (mm)'
    elif distance_type == 'head_head_distance':
        distanceStr = 'HH Distance'
        ylabelStr = 'Head-Head Distance (mm)'
    else:
        raise ValueError('Invalid distance type')

    xlim = (-np.pi, np.pi)  # for relative orientation
    zlim = (-4*np.pi/180, 4*np.pi/180)
    unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi]
    clabelStr = f'Mean {optionString} Angle (degrees)'

    # The bin-center mesh is the same for all datasets; use any one.
    ref_key = next(iter(all_expts))
    X = all_expts[ref_key][f"{keyString}_2Dhist_X"]
    Y = all_expts[ref_key][f"{keyString}_2Dhist_Y"]

    outputFileNameBase = f'Difference in {optionString} Angle, ({distanceStr}), '
    mask_by_sem_limit = (mask_by_sem_limit_degrees * np.pi/180.0
                         if mask_by_sem_limit_degrees is not None else None)

    for cmp_title, cmp_fname, keyA, keyB, slice_color in comparisons:
        d = (all_expts[keyA][f"{keyString}_2Dhist_mean"] -
             all_expts[keyB][f"{keyString}_2Dhist_mean"])
        sem_A = all_expts[keyA][f"{keyString}_2Dhist_sem"]
        sem_B = all_expts[keyB][f"{keyString}_2Dhist_sem"]
        d_unc = (np.sqrt(sem_A**2 + sem_B**2)
                 if (sem_A is not None and sem_B is not None) else None)

        if mask_by_sem_limit_degrees is None:
            titleStr = f'{cmp_title}: {optionString} Angle Probability'
        else:
            titleStr = (f'{cmp_title}: {optionString} Angle; '
                        f'unc. < {mask_by_sem_limit_degrees:.1f} deg')

        plot_2D_heatmap(d, X, Y, Z_unc=d_unc,
                        titleStr=titleStr, xlabelStr=xlabelStr, ylabelStr=ylabelStr,
                        clabelStr=clabelStr, colorRange=colorRange, cmap=cmap,
                        unit_scaling_for_plot=unit_scaling_for_plot,
                        mask_by_sem_limit=mask_by_sem_limit,
                        outputFileName=outputFileNameBase + cmp_fname + outputExtension,
                        closeFigure=False,
                        outputCSVFileName=outputFileNameBase + cmp_fname + '.csv')

        if makeSlicePlots:
            for d_range in [(0.0, 5.0), (5.0, 15.0)]:
                if d_range[0] == 0.0:
                    title_dist = (f'{cmp_title}: {optionString} Angle '
                                  f'for d < {d_range[1]:.2f} mm')
                else:
                    title_dist = (f'{cmp_title}: {optionString} Angle '
                                  f'for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm')
                slice_2D_histogram(d, X, Y, d_unc,
                                   slice_axis='x', other_range=d_range,
                                   titleStr=title_dist, xlabelStr=xlabelStr,
                                   zlabelStr=clabelStr, ylabelStr=ylabelStr,
                                   zlim=zlim, xlim=xlim,
                                   plot_z_zero_line=True, plot_vert_zero_line=True,
                                   unit_scaling_for_plot=unit_scaling_for_plot,
                                   color=slice_color,
                                   outputFileName=outputFileNameBase
                                   + f'{cmp_fname} {d_range[0]:.1f}-{d_range[1]:.1f} mm'
                                   + outputExtension,
                                   closeFigure=False)


def for_frame_by_frame_turning_angle_plots(all_expts, distance_type, comparisons,
                                           Nbins=(19, 25), cmap='berlin',
                                           plot_type='heatmap',
                                           mask_by_sem_limit_degrees=2.0,
                                           colorRange=(-5.0*np.pi/180.0, 5.0*np.pi/180.0),
                                           xlabelStr='Relative Orientation (degrees)',
                                           outputExtension='.svg',
                                           closeFigures=True):
    """
    Frame-by-frame mean turning-angle "Figure 5B" plots.

    For each experiment in all_expts, makes a 2D histogram of the frame-to-frame
    mean turning angle (turning_angle_rad) binned by relative orientation and
    inter-fish distance, via make_turning_angle_plots(), and stores the outputs
    under the keys turn_2Dhist_mean / _sem / _std / _X / _Y in all_expts. Then
    makes the between-condition difference heatmaps via
    make_turning_difference_plots().

    Inputs
    ------
    all_expts : dict of loaded experiment data, keyed by experiment name; each
        entry must contain "datasets" and "plot_color".
    distance_type : 'closest_distance' or 'head_head_distance'.
    comparisons : list of (title, filename, exptKey_A, exptKey_B, slice_color)
        tuples for the difference plots (A minus B).
    Nbins : (n_relorient_bins, n_distance_bins) for the 2D histogram.
    cmap : colormap.
    plot_type : 2D plot type for make_turning_angle_plots ('heatmap' or 'line_plots').
    mask_by_sem_limit_degrees : only show bins whose s.e.m. is below this (degrees);
        applied to both the per-experiment histograms and the difference plots.
    colorRange : (vmin, vmax) color scale (radians) for the per-experiment
        histograms. To use a different range for the difference plots, manually
        alter the call to make_turning_difference_plots() in this function.
    xlabelStr : x-axis label for the difference plots.
    outputExtension : figure file extension, including the dot (e.g. '.svg').
    closeFigures : if True, close figures after creating them.
    """
    if closeFigures:
        print('Frame-by-frame turning angle plots: closing figure windows.')

    for exptName in all_expts.keys():
        saved_pair_turning_outputs = make_turning_angle_plots(
            all_expts[exptName]['datasets'],
            exptName=exptName,
            distance_type=distance_type,
            color=all_expts[exptName]['plot_color'],
            Nbins=Nbins,
            mask_by_sem_limit_degrees=mask_by_sem_limit_degrees,
            colorRange=colorRange,
            cmap=cmap,
            plot_type_2D=plot_type,
            outputFileNameBase=f'{exptName}',
            outputFileNameExt=outputExtension.lstrip('.'),
            closeFigures=closeFigures,
            outputCSVFileName=f'{exptName}.csv',
            makeSlicePlots=False)
        all_expts[exptName]["turn_2Dhist_mean"] = saved_pair_turning_outputs[0]
        all_expts[exptName]["turn_2Dhist_sem"] = saved_pair_turning_outputs[1]
        all_expts[exptName]["turn_2Dhist_std"] = saved_pair_turning_outputs[2]
        all_expts[exptName]["turn_2Dhist_X"] = saved_pair_turning_outputs[3]
        all_expts[exptName]["turn_2Dhist_Y"] = saved_pair_turning_outputs[4]

    # Between-condition difference plots (slices off here)
    make_turning_difference_plots(all_expts, comparisons, distance_type,
                                  cmap=cmap, xlabelStr=xlabelStr,
                                  outputExtension=outputExtension,
                                  keyString='turn', optionString='Turning',
                                  mask_by_sem_limit_degrees=mask_by_sem_limit_degrees,
                                  colorRange=colorRange,
                                  makeSlicePlots=False)


def for_interbout_turning_angle_plots(all_expts, distance_type, comparisons,
                                      Nbins=(11, 15), cmap='berlin',
                                      plot_type='heatmap',
                                      mask_by_sem_limit_degrees=2.0,
                                      colorRange=(-5.0*np.pi/180.0, 5.0*np.pi/180.0),
                                      xlabelStr='Relative Orientation (degrees)',
                                      outputExtension='.svg',
                                      closeFigures=True):
    """
    Inter-bout-interval (IBI) mean turning-angle "Figure 5B" plots.

    For each experiment in all_expts, makes a 2D histogram of the IBI-to-IBI mean
    turning angle (IBI_properties["turning_angle_IBI"]) binned by relative
    orientation and inter-fish distance, via make_interbout_turning_angle_plots(),
    and stores the outputs under the same keys (turn_2Dhist_mean / _sem / _std /
    _X / _Y) as the frame-by-frame path so the difference-plot helper is shared.
    Then makes the between-condition difference heatmaps. No slice plots are made
    for the IB turning angle.

    Inputs are the same as for_frame_by_frame_turning_angle_plots(), except the
    default Nbins is coarser because IBI data is much sparser than frame data
    (one value per IBI rather than per frame).
    """
    if closeFigures:
        print('Inter-bout turning angle plots: closing figure windows.')

    for exptName in all_expts.keys():
        saved_pair_turning_outputs = make_interbout_turning_angle_plots(
            all_expts[exptName]['datasets'],
            exptName=exptName,
            distance_type=distance_type,
            Nbins=Nbins,
            mask_by_sem_limit_degrees=mask_by_sem_limit_degrees,
            colorRange=colorRange,
            cmap=cmap,
            plot_type_2D=plot_type,
            outputFileNameBase=f'{exptName}_IB',
            outputFileNameExt=outputExtension.lstrip('.'),
            closeFigures=closeFigures,
            outputCSVFileName=f'{exptName}_IB.csv')
        all_expts[exptName]["turn_2Dhist_mean"] = saved_pair_turning_outputs[0]
        all_expts[exptName]["turn_2Dhist_sem"] = saved_pair_turning_outputs[1]
        all_expts[exptName]["turn_2Dhist_std"] = saved_pair_turning_outputs[2]
        all_expts[exptName]["turn_2Dhist_X"] = saved_pair_turning_outputs[3]
        all_expts[exptName]["turn_2Dhist_Y"] = saved_pair_turning_outputs[4]

    # Between-condition difference plots (no slices for the IB turning angle)
    make_turning_difference_plots(all_expts, comparisons, distance_type,
                                  cmap=cmap, xlabelStr=xlabelStr,
                                  outputExtension=outputExtension,
                                  keyString='turn', optionString='IB Turning',
                                  mask_by_sem_limit_degrees=mask_by_sem_limit_degrees,
                                  colorRange=colorRange,
                                  makeSlicePlots=False)


def main():
    """
    Main function for loading data and choosing the turning-angle analysis to run.
    Shared inputs (distance_type, comparisons, cmap, plot_type, xlabelStr,
    outputExtension) are defined here and passed to whichever plotting function is
    selected by `which_turning_plot`.
    """

    parentPath = r'C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs'

    #%% Experiment list

    _PAIRS_PATH = r'2 week old - Sept2025 control pairs in dark vs light New Tracking'

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
    ]

    all_expts = load_all_expt_data(expt_list, parentPath)

    #%% Shared inputs, used by either turning-angle function

    distance_type = 'head_head_distance'  # or 'closest_distance'

    # Each tuple: (title_label, filename_label, exptKey_A, exptKey_B, slice_color)
    comparisons = [
        ('Light - TimeShift Light', 'Light - TS0Light', 'TwoWk_Light', 'TwoWk_Light_TIMESHIFT0', 'peru'),
        ('Light - Dark',            'Light - Dark',     'TwoWk_Light', 'TwoWk_Dark',             'darkseagreen'),
        ('Dark - TimeShift Dark',   'Dark - TS0Dark',   'TwoWk_Dark',  'TwoWk_Dark_TIMESHIFT0',  'blue'),
    ]

    cmap = 'berlin'  # 'RdYlBu_r'
    plot_type = 'heatmap'
    xlabelStr = 'Relative Orientation (degrees)'
    outputExtension = '.svg'  # for saving images
    mask_by_sem_limit_degrees = 8.0  # only show bins with s.e.m. below this (deg)
    # Color scale (radians) for the per-experiment turning-angle histograms.
    # (The between-condition difference plots use their own range, set in
    #  make_turning_difference_plots.)
    colorRange = (-8.0*np.pi/180.0, 8.0*np.pi/180.0)

    #%% Choose which turning-angle plot to make

    which_turning_plot = 'inter_bout'  # 'frame_by_frame' or 'inter_bout'

    if which_turning_plot == 'frame_by_frame':
        for_frame_by_frame_turning_angle_plots(
            all_expts, distance_type, comparisons,
            Nbins=(19, 25), cmap=cmap, plot_type=plot_type,
            mask_by_sem_limit_degrees=mask_by_sem_limit_degrees,
            colorRange=colorRange,
            xlabelStr=xlabelStr, outputExtension=outputExtension)
    elif which_turning_plot == 'inter_bout':
        for_interbout_turning_angle_plots(
            all_expts, distance_type, comparisons,
            Nbins=(11, 13), cmap=cmap, plot_type=plot_type,
            mask_by_sem_limit_degrees=mask_by_sem_limit_degrees,
            colorRange=colorRange,
            xlabelStr=xlabelStr, outputExtension=outputExtension)
    else:
        raise ValueError('Invalid turning plot option.')


if __name__ == '__main__':
    main()
