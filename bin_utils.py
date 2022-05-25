import numpy as np
import pandas as pd


def collapse_bins(bin_boundaries, bin_counts, smoothing=0, min_bin_num=0, smooth_before_binning=False, population_percents=[]):
    this_bin_boundaries = list(bin_boundaries)
    this_bin_counts = list(bin_counts)
    if smooth_before_binning:
        this_bin_counts = [bc + smoothing * population_percents[bc_idx] for bc_idx, bc in enumerate(this_bin_counts)]
    while True:
        if len(this_bin_counts) <= 1 or all(p >= min_bin_num for p in this_bin_counts):
            break
        min_idx = this_bin_counts.index(min(this_bin_counts))
        del_idx = min_idx
        if min_idx == 0:
            min_adjacent_idx = min_idx + 1
            del_idx = min_idx + 1
        elif min_idx == len(this_bin_counts) - 1:
            min_adjacent_idx = min_idx - 1
        else:
            if this_bin_counts[min_idx - 1] >= this_bin_counts[min_idx + 1]:
                min_adjacent_idx = min_idx + 1
            else:
                min_adjacent_idx = min_idx - 1

            if min_adjacent_idx >= min_idx:
                del_idx = min_idx + 1

        this_p = this_bin_counts[min_idx]
        this_bin = this_bin_boundaries[del_idx]
        this_bin_counts[min_adjacent_idx] += this_bin_counts[min_idx]

        del this_bin_counts[min_idx]
        del this_bin_boundaries[del_idx]
        if smooth_before_binning:
            del population_percents[min_idx]
    return this_bin_boundaries, this_bin_counts


def get_population_percents(population_data, dem, population_table_cols, user_dem_bins, bins):
    if len(bins) < 3:
        return np.array([1])

    if dem in ('gender', 'education'):
        return population_data[population_table_cols].tolist()
    else:
        percents = []
        base_idx = 0

        for b in bins[1:]:
            bin_idx = user_dem_bins.index(b)
            bin_total = np.sum(population_data[population_table_cols[base_idx:bin_idx]])
            percents.append(bin_total)
            base_idx = bin_idx

        # normalize array sum to 100
        percents = percents / np.sum(percents)

        return percents


def get_bins(user_data, population_data, dem, population_table_cols, user_dem_bins, smoothing=0, min_bin_num=0, smooth_before_binning=False):
    # get bins
    bins = user_dem_bins
    values = user_data[dem]
    bins_counts, _ = np.histogram(values, bins=bins)
    if min_bin_num > 0:
        bins, bins_counts = collapse_bins(bins, bins_counts, smoothing, min_bin_num)

    # get percentages
    user_percents = np.array([x / len(values) for x in bins_counts])
    population_percents = get_population_percents(population_data, dem, population_table_cols, user_dem_bins, bins)

    # recalculate if smooth before binning
    if smooth_before_binning:
        bins, bins_counts = collapse_bins(bins, bins_counts, smoothing, min_bin_num, smooth_before_binning, population_percents)
        population_percents = get_population_percents(population_data, dem, population_table_cols, user_dem_bins, bins)

    return bins, user_percents, population_percents

