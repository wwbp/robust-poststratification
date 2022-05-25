import numpy as np
import pandas as pd
import quantipy as qp


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


def get_population_percents(population_data, population_table_cols, user_dem_bins, bins):
    if len(bins) < 3:
        return np.array([1])

    if len(population_table_cols) == 2:
        # binary demographic
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
    population_percents = get_population_percents(population_data, population_table_cols, user_dem_bins, bins)

    # recalculate if smooth before binning
    if smooth_before_binning:
        bins, bins_counts = collapse_bins(bins, bins_counts, smoothing, min_bin_num, smooth_before_binning, population_percents)
        population_percents = get_population_percents(population_data, population_table_cols, user_dem_bins, bins)

    return bins, user_percents, population_percents


def create_banded_dataset(user_data, population_data, demographics, smoothing, min_bin_num, smooth_before_binning, user_dem_bins, population_dem_cols):
    dataset = qp.DataSet(name="_".join(demographics)+'_dataset', dimensions_comp=False)

    bands = {}
    user_dem_percents = {}
    population_dem_percents = {}

    for dem in demographics:
        bins, user_percents, population_percents = get_bins(
            user_data=user_data,
            population_data=population_data,
            dem=dem,
            population_table_cols=population_dem_cols[dem],
            user_dem_bins=user_dem_bins[dem],
            smoothing=smoothing,
            min_bin_num=min_bin_num,
            smooth_before_binning=smooth_before_binning,
        )

        band = [(bins[i], bins[i + 1] - 1) for i in range(len(bins[:-1]))]
        band[-1] = (bins[-2], bins[-1])
        bands[dem] = band

        user_dem_percents[dem] = {i+1:perc for i, perc in enumerate(user_percents)}
        population_dem_percents[dem] = {i+1:perc for i, perc in enumerate(population_percents)}

    dataset.from_components(user_data[['user_id'] + demographics])

    for band in bands:
        dataset.band(band, bands[band])

    return dataset, user_dem_percents, population_dem_percents


def rakeonvar(df, dem, bin_marginals):
    for bin_class, bin_perc in bin_marginals.items():
        # 1. subset df where column=dem and column value=bin_class
        subset_df = df[(df[dem] == bin_class)]
        index_array = (df[dem] == bin_class)

        # 2. multiply by census prob, divide by sum of "perc"
        if sum(subset_df['perc']) == 0:
            data = subset_df['perc']
        else:
            data = subset_df['perc'] * (bin_perc / sum(subset_df['perc']))
        
        # 3. replace the `perc` column in df with the updated subset_df[perc]
        df.loc[index_array, 'perc'] = data

    return df


def rake(df, population_marginals):
    convcrit = 0.01
    pct_still = 1 - convcrit
    diff_error = 999999
    diff_error_old = 99999999999
    max_iterations = 1000
    initial_weights = df['perc'].copy()

    # run the raking: for each iteration, rake over each key in census_marginals
    for iteration in range(1, max_iterations+1):
        old_weights = df['perc'].copy()

        if not diff_error < pct_still * diff_error_old:
            break

        for dem, bin_marginals in population_marginals.items():
            rakeonvar(df, dem, bin_marginals)

        diff_error_old = diff_error
        diff_error = sum(abs(df['perc'] - old_weights))

    if iteration == max_iterations:
        print('Convergence did not occur in %s iterations' % iteration)

    df['perc'] = df['perc'].div(initial_weights,0)
    return df
