from itertools import product

import numpy as np
import pandas as pd
import quantipy as qp

import rake_utils
import bin_utils


def create_weights(user_data, bins, user_percents, population_percents, dem, smoothing=0, smooth_before_binning=False, uninformed_smoothing=False):
    total_twitter = user_data.shape[0]
    ii = len(bins) - 2
    if len(population_percents) == 1 and population_percents[0] == 1:
        user_data['weight'] = 1
        return user_data

    for bin_entry in bins[1:][::-1]:
        twitter_bin_n = round(user_percents[ii] * total_twitter, 0)
        if uninformed_smoothing:
            percentage_twitter_bin = (twitter_bin_n + 1) / (total_twitter + len(user_percents))
        elif smooth_before_binning:
            percentage_twitter_bin = twitter_bin_n / total_twitter
        else:
            percentage_twitter_bin = (twitter_bin_n + smoothing * population_percents[ii]) / (total_twitter + smoothing)

        if percentage_twitter_bin > 0:
            w = population_percents[ii] / percentage_twitter_bin
        else:
            w = 0

        if ii == len(bins) - 2:
            user_data['weight'] = np.where(user_data[dem] < bin_entry, w, None)
        else:
            user_data['weight'] = np.where(user_data[dem] < bin_entry, w, user_data['weight'])
        ii -= 1

    # fill missing entries and renormalize
    user_data['weight'].fillna(user_data['weight'].mean(), inplace=True)
    user_data['weight'] = user_data['weight'] / user_data['weight'].sum() * len(user_data)

    return user_data


def create_weights_single(user_data, population_data, dem, smoothing, min_bin_num, smooth_before_binning, uninformed_smoothing, user_bins, population_cols):
    bins, user_percents, population_percents = bin_utils.get_bins(
        user_data=user_data,
        population_data=population_data,
        dem=dem,
        population_table_cols=population_cols,
        user_dem_bins=user_bins,
        smoothing=smoothing,
        min_bin_num=min_bin_num,
        smooth_before_binning=smooth_before_binning,
    )
    user_weights = create_weights(
        user_data=user_data,
        bins=bins,
        user_percents=user_percents,
        population_percents=population_percents,
        dem=dem,
        smoothing=smoothing,
        smooth_before_binning=smooth_before_binning,
        uninformed_smoothing=uninformed_smoothing,
    )

    return user_weights


def create_banded_dataset(user_data, population_data, demographics, smoothing, min_bin_num, smooth_before_binning, user_dem_bins, population_dem_cols):
    dataset = qp.DataSet(name="_".join(demographics)+'_dataset', dimensions_comp=False)

    all_targets = {}
    all_bands = {}
    all_population_dems = {}
    all_sample_dems = {}

    for dem in demographics:
        bins, user_percents, population_percents = bin_utils.get_bins(
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
        all_bands[dem] = band

        all_sample_dems[dem] = {i+1:perc for i, perc in enumerate(user_percents)}
        all_population_dems[dem] = {i+1:perc for i, perc in enumerate(population_percents)}
        all_targets[dem + '_banded'] = {i+1:perc for i, perc in enumerate(population_percents)}

    dataset.from_components(user_data[['user_id'] + demographics])

    for dem in all_bands:
        dataset.band(dem, all_bands[dem])

    return dataset, all_population_dems, all_targets


def create_weights_rake(user_data, population_data, demographics, smoothing, min_bin_num, smooth_before_binning, uninformed_smoothing, user_dem_bins, population_dem_cols):
    dataset, all_population_dems, all_targets = create_banded_dataset(
        user_data=user_data,
        population_data=population_data,
        demographics=demographics,
        smoothing=smoothing,
        min_bin_num=min_bin_num,
        smooth_before_binning=smooth_before_binning,
        user_dem_bins=user_dem_bins,
        population_dem_cols=population_dem_cols,
    )

    data = dataset.data()
    dataframe_data = {}
    columns = demographics + ['perc']

    bands = [dem + '_banded' for dem in demographics]
    group_dict = data.groupby(bands).agg(['count'])['user_id'].to_dict()['count']
    sorted_keys = list(product(*[all_population_dems[d].keys() for d in demographics]))
    total_twitter_users = float(data.shape[0])
    naive_percentages = {key: np.prod([all_population_dems[d][k] for d, k in zip(demographics, key)]) / (100 ** len(demographics)) for key in sorted_keys}

    for i, key in enumerate(sorted_keys):
        if key in group_dict:
            if uninformed_smoothing:
                num = group_dict[key] + 1
                den = float(total_twitter_users + len(naive_percentages))
            else:
                num = group_dict[key] + smoothing * naive_percentages[key]
                den = float(total_twitter_users + smoothing)

            dataframe_data[i] = list(key) + [num / den * 100]
        elif smoothing > 0 or uninformed_smoothing:
            if uninformed_smoothing:
                num = 1
                den = float(len(naive_percentages))
            else:
                num = smoothing * naive_percentages[key]
                den = float(smoothing)
            dataframe_data[i] = list(key) + [num / den * 100]
        else:
            dataframe_data[i] = list(key) + [0]

    rake_df = pd.DataFrame.from_dict(dataframe_data, orient='index')
    rake_df.columns = columns
    
    rake_utils.rake(rake_df, all_population_dems)
    rake_df.columns = [d + '_banded' for d in demographics] + ['perc']
    user_weights = pd.merge(data, rake_df, on=[d + '_banded' for d in demographics])
    user_weights.rename(columns={'perc': 'weight'}, inplace=True)

    return user_weights


def create_weights_naive(user_data, population_data, demographics, smoothing, min_bin_num, smooth_before_binning, uninformed_smoothing, user_dem_bins, population_dem_cols):
    dataset, all_population_dems, all_targets = create_banded_dataset(
        user_data=user_data,
        population_data=population_data,
        demographics=demographics,
        smoothing=smoothing,
        min_bin_num=min_bin_num,
        smooth_before_binning=smooth_before_binning,
        user_dem_bins=user_dem_bins,
        population_dem_cols=population_dem_cols,
    )

    data = dataset.data()
    data['naive_banded'] = 0
    combined_targets = {'naive_banded': {}}
    bands = [dem + '_banded' for dem in demographics]
    group_dict = data.groupby(bands).agg(['count'])['user_id'].to_dict()
    sorted_keys = list(product(*[all_population_dems[d].keys() for d in demographics]))

    i = 0
    skipped = False
    for key in sorted_keys:
        count = group_dict['count'].get(key, 0)
        if count < 1:
            skipped = True
            continue

        prod = 1
        query = []
        for k, band in zip(key, bands):
            prod += all_targets[band][k]
            query.append(f'{band} == {k}')
        combined_targets['naive_banded'][i + 1] = prod
        data.loc[data.eval(' and '.join(query)), 'naive_banded'] = i + 1
        i += 1

    s = sum(combined_targets['naive_banded'].values())
    if skipped or round(s) != 100:
        # renormalize combined_targets
        combined_targets['naive_banded'] = {k: v * 100 / s for k, v in combined_targets['naive_banded'].items()}

    sorted_bins = sorted(list(combined_targets['naive_banded'].keys()))
    sorted_targets = [combined_targets['naive_banded'][kkey] for kkey in sorted_bins]
    sorted_sample_targets = data.groupby(['naive_banded']).agg(['count'])['user_id'].to_dict()['count']
    t = float(sum(sorted_sample_targets.values()))
    sorted_sample_targets = {kkey:v/t for kkey,v in sorted_sample_targets.items()}
    sorted_sample_targets = [sorted_sample_targets[kkey] for kkey in sorted_bins]

    user_weights = create_weights(
        user_data=data,
        bins=sorted_bins,
        user_percents=sorted_sample_targets,
        population_percents=sorted_targets,
        dem='naive_banded',
        smoothing=smoothing,
        smooth_before_binning=smooth_before_binning,
        uninformed_smoothing=uninformed_smoothing
    )

    return user_weights
