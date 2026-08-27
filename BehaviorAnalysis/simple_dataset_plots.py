# -*- coding: utf-8 -*-
# simple_dataset_plots.py
"""

Author:   Raghuveer Parthasarathy
Date: June 23, 2026

Last modified June 23, 2026 -- Raghu Parthasarathy

Description
-----------

Code to plot various dataset properties.
Similar to functions called in make_pair_fish_plots(), and other code.
Probably redundant, but might be useful.

"""

import os
import matplotlib.pyplot as plt

from behavior_plots import make_pair_fish_plots

from IO_toolkit import load_and_assign_from_pickle, plot_probability_distr
from toolkit import combine_all_values_constrained

def run_head_head_distance(DATASETS, dataset_key, closeFigures=False):
    cfg = DATASETS[dataset_key]

    pickleFileName1 = cfg["pickle1"]
    pickleFileName2 = cfg["pickle2"]
    color = cfg["color"]
    exptName = cfg["label"]

    # Load data
    all_position_data, variable_tuple = load_and_assign_from_pickle(
        pickleFileName1=pickleFileName1,
        pickleFileName2=pickleFileName2
    )

    (datasets, CSVcolumns, expt_config, params,
     N_datasets, Nfish, basePath, dataPath,
     subGroupName) = variable_tuple

    # Output names auto-generated
    outputFileName = f"{exptName}_HHDistance.png"
    outputCSVFileName = f"{exptName}_distance_head_head.csv"

    # Compute metric
    head_head_mm_all = combine_all_values_constrained(
        datasets,
        keyName='head_head_distance_mm',
        dilate_minus1=False
    )

    # Plot
    plot_probability_distr(
        head_head_mm_all,
        bin_width=0.5,
        bin_range=[0, None],
        color=color,
        yScaleType='linear',
        plot_each_dataset=False,
        plot_sem_band=True,
        xlim=(-1.0, 50.0),
        ylim=(0.0, 0.05),
        xlabelStr='Head-head distance (mm)',
        titleStr=f'{exptName}: head-head distance (mm)',
        outputFileName=outputFileName,
        closeFigure=closeFigures,
        outputCSVFileName=outputCSVFileName
    )

def run_radial_distribution(DATASETS, dataset_key, closeFigures=False):
    cfg = DATASETS[dataset_key]

    pickleFileName1 = cfg["pickle1"]
    pickleFileName2 = cfg["pickle2"]
    color = cfg["color"]
    exptName = cfg["label"]

    # Load data
    all_position_data, variable_tuple = load_and_assign_from_pickle(
        pickleFileName1=pickleFileName1,
        pickleFileName2=pickleFileName2
    )

    (datasets, CSVcolumns, expt_config, params,
     N_datasets, Nfish, basePath, dataPath,
     subGroupName) = variable_tuple

    # Output names
    outputFileName = f"{exptName}_RadialDistribution.png"
    outputCSVFileName = f"{exptName}_radial_distribution.csv"

    # Compute radial values
    radial_position_mm_all = combine_all_values_constrained(
        datasets,
        keyName='radial_position_mm',
        dilate_minus1=False
    )

    # Plot
    plot_probability_distr(
        radial_position_mm_all,
        bin_width=0.5,
        bin_range=[0, None],
        color=color,
        yScaleType='linear',
        plot_each_dataset=False,
        plot_sem_band=True,
        normalize_by_inv_bincenter=True,   # ✅ critical for radial distributions
        ylim=(-0.025, 0.5),
        xlabelStr='Radial position (mm)',
        titleStr=f'{exptName}: Probability Distr.: r',
        outputFileName=outputFileName,
        closeFigure=closeFigures,
        outputCSVFileName=outputCSVFileName
    )


def plot_multiple_head_head_distance(DATASETS, dataset_keys, closeFigures=False):
    plt.figure()

    for key in dataset_keys:
        cfg = DATASETS[key]

        pickleFileName1 = cfg["pickle1"]
        pickleFileName2 = cfg["pickle2"]
        color = cfg["color"]
        exptName = cfg["label"]

        # Load
        all_position_data, variable_tuple = load_and_assign_from_pickle(
            pickleFileName1=pickleFileName1,
            pickleFileName2=pickleFileName2
        )

        (datasets, CSVcolumns, expt_config, params,
         N_datasets, Nfish, basePath, dataPath,
         subGroupName) = variable_tuple

        # Compute
        head_head_mm_all = combine_all_values_constrained(
            datasets,
            keyName='head_head_distance_mm',
            dilate_minus1=False
        )

        # Plot (overlay — do NOT save yet)
        plot_probability_distr(
            head_head_mm_all,
            bin_width=0.5,
            bin_range=[0, None],
            color=color,
            yScaleType='linear',
            plot_each_dataset=False,
            plot_sem_band=True,
            xlim=(-1.0, 50.0),
            ylim=(0.0, 0.05),
            xlabelStr='Head-head distance (mm)',
            titleStr='Head-head distance comparison',
            outputFileName=None,              # <-- important: don't overwrite per dataset
            closeFigure=False,
            outputCSVFileName=None,
            label=exptName   # ✅ requires your function to pass label to plt.plot
        )

    # Add legend and save once
    plt.legend()
    outputFileName = "_vs_".join(dataset_keys) + "_HHDistance.png"
    plt.savefig(outputFileName)

    if closeFigures:
        plt.close()

def main():
    """
    Main function for loading data and calling analysis functions.
    """

    mainPathName = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs"

    DATASETS = {
        "pairs_light": {
            "pickle1": os.path.join(mainPathName,
                r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"),
            "pickle2": os.path.join(mainPathName,
                r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"),
            "color": "darkorange",
            "label": "Pairs_Light"
        },

        "pairs_light_ts": {
            "pickle1": os.path.join(mainPathName,
                r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_positionData.pickle"),
            "pickle2": os.path.join(mainPathName,
                r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_Analysis\TwoWk_Sept2025_TS0_Light_Cond_2_.pickle"),
            "color": "saddlebrown",
            "label": "Pairs_Light_TS"
        },

        "pairs_dark": {
            "pickle1": os.path.join(mainPathName,
                r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_positionData.pickle"),
            "pickle2": os.path.join(mainPathName,
                r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_Analysis\TwoWk_Sept2025_Dark_Cond_1_datasets.pickle"),
            "color": "cornflowerblue",
            "label": "Pairs_Dark"
        },

        "pairs_dark_ts": {
            "pickle1": os.path.join(mainPathName,
                r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_TS0_Dark_Cond_1_positionData.pickle"),
            "pickle2": os.path.join(mainPathName,
                r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_TS0_Dark_Cond_1_Analysis\TwoWk_Sept2025_TS0_Dark_Cond_1_dat.pickle"),
            "color": "darkblue",
            "label": "Pairs_Dark_TS"
        },

        "single_light": {
            "pickle1": r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle",
            "pickle2": r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle",
            "color": "gold",
            "label": "Single_Light"
        },

        "single_dark": {
            "pickle1": r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_positionData.pickle",
            "pickle2": r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_Analysis\TwoWkSingle_Dark_Cond_1_datasets.pickle",
            "color": "slategrey",
            "label": "Single_Dark"
        }
    }


    """
    run_head_head_distance(DATASETS, "pairs_light")
    run_head_head_distance(DATASETS, "pairs_dark")
    run_head_head_distance(DATASETS, "single_light")
    run_head_head_distance(DATASETS, "pairs_light_ts")
    run_head_head_distance(DATASETS, "pairs_dark_ts")
    """
    
    """
    Multiple graphs; won't work unless I make plot_probability_distr() have a 
    "label" input

    plot_multiple_head_head_distance(DATASETS, [
     "pairs_light",
     "pairs_dark"
    ])
    """

    run_radial_distribution(DATASETS, "single_light")
    run_radial_distribution(DATASETS, "single_dark")
    run_radial_distribution(DATASETS, "pairs_light")
    run_radial_distribution(DATASETS, "pairs_dark")

if __name__ == '__main__':
    main()
