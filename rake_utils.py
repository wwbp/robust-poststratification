from itertools import product

import numpy as np
import pandas as pd
import quantipy as qp


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
        diff_error = sum(abs(df['perc']-old_weights))

    if iteration == max_iterations:
        print('Convergence did not occur in %s iterations' % iteration)

    df['perc'] = df['perc'].div(initial_weights,0)


def create_raked_df_from_four(df, dem, pop_percentages, smoothed_k, uninformed_smoothing=False):
    dataframe_data = {}
    columns = dem + ['perc']

    group_dict = df.groupby([d + '_banded' for d in dem]).agg(['count'])['user_id'].to_dict()['count']
    sorted_keys = list(product(pop_percentages[dem[0]].keys(), pop_percentages[dem[1]].keys(), pop_percentages[dem[2]].keys(), pop_percentages[dem[3]].keys()))
    total_twitter_users = float(df.shape[0])
    naive_percentages = {kkey: np.prod([pop_percentages[dem[i]][kkey[i]] for i in range(len(dem))])/(100*100*100*100) for kkey in sorted_keys}

    for i, kkey in enumerate(sorted_keys):
        if kkey in group_dict:
            if uninformed_smoothing:
                num = group_dict[kkey] + 1
                den = float(total_twitter_users + len(naive_percentages))
            else:
                num = group_dict[kkey] + smoothed_k*naive_percentages[kkey]
                den = float(total_twitter_users + smoothed_k)

            dataframe_data[i] = list(kkey) + [num/den*100]
        elif smoothed_k > 0 or uninformed_smoothing:
            if uninformed_smoothing:
                num = 1
                den = float(len(naive_percentages))
            else:
                num = smoothed_k*naive_percentages[kkey]
                den = float(smoothed_k)
            dataframe_data[i] = list(kkey) + [num/den*100]
        else:
            dataframe_data[i] = list(kkey) + [0]

    rake_df = pd.DataFrame.from_dict(dataframe_data, orient='index')
    rake_df.columns = columns

    rake(rake_df, pop_percentages)
    rake_df.columns = [d + '_banded' for d in dem] + ['perc']
    merged_df = pd.merge(df, rake_df, on=[d + '_banded' for d in dem])
    merged_df.rename(columns={'perc': 'weight'}, inplace=True)

    return merged_df


def create_raked_df_from_two(df, dem, pop_percentages, smoothed_k, uninformed_smoothing=False):
    dataframe_data = {}
    columns = dem + ['perc']

    group_dict = df.groupby([d + '_banded' for d in dem]).agg(['count'])['user_id'].to_dict()['count']
    sorted_keys = list(product(pop_percentages[dem[0]].keys(), pop_percentages[dem[1]].keys()))
    total_twitter_users = float(df.shape[0])
    naive_percentages = {kkey: np.prod([pop_percentages[dem[0]][kkey[0]], pop_percentages[dem[1]][kkey[1]]])/(100*100) for kkey in sorted_keys}

    for i, kkey in enumerate(sorted_keys):
        if kkey in group_dict:
            if uninformed_smoothing:
                num = group_dict[kkey] + 1
                den = float(total_twitter_users + len(naive_percentages))
            else:
                num = group_dict[kkey] + smoothed_k*naive_percentages[kkey]
                den = float(total_twitter_users + smoothed_k)

            dataframe_data[i] = list(kkey) + [num/den*100]
        elif smoothed_k > 0 or uninformed_smoothing:
            if uninformed_smoothing:
                num = 1
                den = float(len(naive_percentages))
            else:
                num = smoothed_k*naive_percentages[kkey]
                den = float(smoothed_k)
            dataframe_data[i] = list(kkey) + [num/den*100]
        else:
            dataframe_data[i] = list(kkey) + [0]

    rake_df = pd.DataFrame.from_dict(dataframe_data, orient='index')
    rake_df.columns = columns
    
    rake(rake_df, pop_percentages)
    rake_df.columns = [d + '_banded' for d in dem] + ['perc']
    merged_df = pd.merge(df, rake_df, on=[d + '_banded' for d in dem])
    merged_df.rename(columns={'perc': 'weight'}, inplace=True)

    return merged_df
