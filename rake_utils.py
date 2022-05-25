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
        diff_error = sum(abs(df['perc'] - old_weights))

    if iteration == max_iterations:
        print('Convergence did not occur in %s iterations' % iteration)

    df['perc'] = df['perc'].div(initial_weights,0)
