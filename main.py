import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import quantipy as qp

import utils
import weight_utils


BINS = {
    'age': [13, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 81],
    'income': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'gender': [0, 1, 5],
    'education': [0, 1, 5],
}
USER_BINS = {
    'age': [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],
    'income': [0, 10000, 15000, 25000, 35000, 50000, 75000, 100000, 150000, 200000, 1e12],
    'gender': [-100, 0, 100],
    'education': [0, 0.5, 1],
}
POPULATION_TABLE_COLS = {
    'age': [
        'total_15to19', 'total_20to24', 'total_25to29', 'total_30to34',
        'total_35to39', 'total_40to44', 'total_45to49', 'total_50to54',
        'total_55to59', 'total_60to64', 'total_65plus',
    ],
    'income': [
        'incomelt10k', 'income10kto14999', 'income15kto24999', 'income25kto34999',
        'income35kto49999', 'income50kto74999', 'income75kto99999',
        'income100kto149999', 'income150kto199999', 'incomegt200k',
    ],
    'gender': ['male_perc', 'female_perc'],
    'education': ['perc_high_school_or_higher', 'perc_bach_or_higher']
}
REDIST_BINS = {
    'age': [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],
    'income': [0, 30000, 50000, 75000, 1e12],
    'gender': [-100, 0, 100],
    'education': [0, 0.5, 1],
}
REDIST_PERCENTS = {
    'age': [0.4882, 0.3044, 0.1675, 0.4000],
    'income': [0.1989, 0.2047, 0.2428, 0.3536],
    'gender': [0.4878, 0.5122],
    'education': [0.5078, 0.4922],
}


# DEMOGRAPHICS = ['income']
# DEMOGRAPHICS = ['age', 'gender']
DEMOGRAPHICS = ['income', 'education']
# DEMOGRAPHICS = ['age', 'income', 'education']
# DEMOGRAPHICS = ['age', 'gender', 'income', 'education']

SMOOTHING = 0
MIN_BIN_NUM = 0

UNINFORMED_SMOOTHING = False
SMOOTH_BEFORE_BINNING = False
REDISTRIBUTION = True
NAIVE = False

USER_TABLE = './data/users_en_30_10pct.csv'
POPULATION_TABLE = './data/acs2015_5yr_age_gender_income_education.csv'
OUTPUT = './data/weights.csv'


def main():
    print(f'CREATING WEIGHTS FOR: {DEMOGRAPHICS}')
    print(f'    WITH SMOOTHING K = {SMOOTHING}')
    print(f'    WITH MIN BIN NUM = {MIN_BIN_NUM}')

    if UNINFORMED_SMOOTHING:
        print('    WITH UNINFORMED SMOOTHING')
    if SMOOTH_BEFORE_BINNING:
        print('    WITH SMOOTH BEFORE BINNING')
    if NAIVE:
        print('    WITH NAIVE POST-STRATIFICATION')
    if REDISTRIBUTION:
        print('    WITH REDISTRIBUTION')

    if os.path.exists(OUTPUT):
        sys.exit('ERROR: output file already exists')

    print('Loading Data')
    user_df, population_df = utils.load_data(USER_TABLE, POPULATION_TABLE)
    if REDISTRIBUTION:
        print('Performing Redistribution')
        user_df = utils.apply_redistribution(user_df, DEMOGRAPHICS, REDIST_BINS, REDIST_PERCENTS)
    print('Performing Binning')
    user_df = utils.bin_demographics(user_df, DEMOGRAPHICS, USER_BINS, BINS)

    cnty_list = user_df.index.unique().tolist()
    cnty_list.sort()
    print(f'Number of counties: {len(cnty_list)}')

    for count, cnty in enumerate(cnty_list, start=1):
        print(f'== Processing county: {cnty} [{count} / {len(cnty_list)}] ==')

        try:
            population_data = population_df.loc[cnty]
        except:
            print(f'    SKIPPING: population table does not contain county {cnty}')
            continue
        user_data = user_df.loc[cnty]

        if len(DEMOGRAPHICS) == 1:
            # single correction factor
            dem = DEMOGRAPHICS[0]
            user_weights = weight_utils.create_weights_single(
                user_data=user_data,
                population_data=population_data,
                dem=dem,
                smoothing=SMOOTHING,
                min_bin_num=MIN_BIN_NUM,
                smooth_before_binning=SMOOTH_BEFORE_BINNING,
                uninformed_smoothing=UNINFORMED_SMOOTHING,
                user_bins=BINS[dem],
                population_cols=POPULATION_TABLE_COLS[dem],
            )
        elif not NAIVE:
            # raking
            user_weights = weight_utils.create_weights_rake(
                user_data=user_data,
                population_data=population_data,
                demographics=DEMOGRAPHICS,
                smoothing=SMOOTHING,
                min_bin_num=MIN_BIN_NUM,
                smooth_before_binning=SMOOTH_BEFORE_BINNING,
                uninformed_smoothing=UNINFORMED_SMOOTHING,
                user_dem_bins=BINS,
                population_dem_cols=POPULATION_TABLE_COLS,
            )
        else:
            # naive post-stratification
            user_weights = weight_utils.create_weights_naive(
                user_data=user_data,
                population_data=population_data,
                demographics=DEMOGRAPHICS,
                smoothing=SMOOTHING,
                min_bin_num=MIN_BIN_NUM,
                smooth_before_binning=SMOOTH_BEFORE_BINNING,
                uninformed_smoothing=UNINFORMED_SMOOTHING,
                user_dem_bins=BINS,
                population_dem_cols=POPULATION_TABLE_COLS,
            )


        # write weights to output file
        # with open(OUTPUT, 'a') as f:
        #     writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        #     user_ids = user_weights['user_id'].tolist()
        #     weights = user_weights['weight'].tolist()
        #     for i in range(len(user_ids)):
        #         writer.writerow([user_ids[i], cnty, weights[i]])



if __name__ == '__main__':
    main()
