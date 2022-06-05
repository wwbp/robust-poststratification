import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

import utils
import weights


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
NAIVE_POSTSTRAT = False

USER_TABLE = './data/users_en_30_10pct.csv'
POPULATION_TABLE = './data/acs2015_5yr_age_gender_income_education.csv'
OUTPUT = './data/weights.csv'


def get_args():
    parser = argparse.ArgumentParser(description='Create Robust post-stratification weights.')

    # correction factors
    parser.add_argument('--demographics', type=str, metavar='FIELD(S)', dest='demographics', nargs='+',
        default=DEMOGRAPHICS, help='Fields to compare with.')

    # parameters
    parser.add_argument('--minimum_bin_threshold', dest='minimum_bin_threshold', type=int,
        default=MIN_BIN_NUM, help='Set the minimum bin threshold. Default {d}'.format(d=MIN_BIN_NUM))
    parser.add_argument('--smoothing_k', dest='smoothing_k', type=int,
        default=SMOOTHING, help='Set the smoothing constant. Default {d}'.format(d=SMOOTHING))

    # non-standard analyses
    parser.add_argument('--naive_poststrat', action='store_true', dest='naive_poststrat',
        default=NAIVE_POSTSTRAT, help='Apply naive post-stratification.')
    parser.add_argument('--smooth_before_binning', action='store_true', dest='smooth_before_binning',
        default=SMOOTH_BEFORE_BINNING, help='Apply smoothing before binning.')
    parser.add_argument('--uninformed_smoothing', action='store_true', dest='uninformed_smoothing',
        default=UNINFORMED_SMOOTHING, help='Apply uninformed smoothing.')
    parser.add_argument('--redistribution', action='store_true', dest='redistribution',
        default=REDISTRIBUTION, help='Apply Estimator redistribution.')

    # input and output files
    parser.add_argument('--user_table', dest='user_table', type=str, default=USER_TABLE,
        help='User data csv. Default: {d}'.format(d=USER_TABLE))
    parser.add_argument('--population_data', dest='population_data', type=str, default=POPULATION_TABLE,
        help='Population data csv. Default: {d}'.format(d=POPULATION_TABLE))
    parser.add_argument('--output', dest='output', type=str, default=OUTPUT,
        help='Output file. Default: {d}'.format(d=OUTPUT))

    args = parser.parse_args()
    return args


def main(args):
    print('CREATING WEIGHTS FOR: {d}'.format(d=args.demographics))
    print('    WITH SMOOTHING K = {d}'.format(d=args.smoothing_k))
    print('    WITH MIN BIN NUM = {d}'.format(d=args.minimum_bin_threshold))
    if args.uninformed_smoothing:
        print('    WITH UNINFORMED SMOOTHING')
    if args.smooth_before_binning:
        print('    WITH SMOOTH BEFORE BINNING')
    if len(args.demographics) > 1:
        if args.naive_poststrat:
            print('    WITH NAIVE POST-STRATIFICATION')
        else:
            print('    WITH RAKING')
    if args.redistribution:
        print('    WITH REDISTRIBUTION')

    # ensure existing file is not overwritten
    if os.path.exists(args.output):
        sys.exit('ERROR: output file already exists')

    # initialize output csv with header
    with open(args.output, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['user_id', 'cnty', 'weight'])

    print('Loading Data')
    user_df, population_df = utils.load_data(args.user_table, args.population_data)
    if args.redistribution:
        print('Performing Redistribution')
        user_df = utils.apply_redistribution(user_df, args.demographics, REDIST_BINS, REDIST_PERCENTS)
    print('Performing Binning')
    user_df = utils.bin_demographics(user_df, args.demographics, USER_BINS, BINS)

    cnty_list = user_df.index.unique().tolist()
    cnty_list.sort()
    size = len(cnty_list)
    print('Number of counties: {d}'.format(d=size))

    for count, cnty in enumerate(cnty_list, start=1):
        print('== Processing county: {cnty} [{count} / {size}] =='.format(cnty=cnty, count=count, size=size))

        try:
            population_data = population_df.loc[cnty]
        except:
            print('    SKIPPING: population table does not contain county {d}'.format(d=cnty))
            continue
        user_data = user_df.loc[cnty]

        if len(args.demographics) == 1:
            # single correction factor
            dem = args.demographics[0]
            user_weights = weights.create_weights_single(
                user_data=user_data,
                population_data=population_data,
                dem=dem,
                smoothing=args.smoothing_k,
                min_bin_num=args.minimum_bin_threshold,
                smooth_before_binning=args.smooth_before_binning,
                uninformed_smoothing=args.uninformed_smoothing,
                user_bins=BINS[dem],
                population_cols=POPULATION_TABLE_COLS[dem],
            )
        elif not args.naive_poststrat:
            # raking
            user_weights = weights.create_weights_rake(
                user_data=user_data,
                population_data=population_data,
                demographics=args.demographics,
                smoothing=args.smoothing_k,
                min_bin_num=args.minimum_bin_threshold,
                smooth_before_binning=args.smooth_before_binning,
                uninformed_smoothing=args.uninformed_smoothing,
                user_dem_bins=BINS,
                population_dem_cols=POPULATION_TABLE_COLS,
            )
        else:
            # naive post-stratification
            user_weights = weights.create_weights_naive(
                user_data=user_data,
                population_data=population_data,
                demographics=args.demographics,
                smoothing=args.smoothing_k,
                min_bin_num=args.minimum_bin_threshold,
                smooth_before_binning=args.smooth_before_binning,
                uninformed_smoothing=args.uninformed_smoothing,
                user_dem_bins=BINS,
                population_dem_cols=POPULATION_TABLE_COLS,
            )

        # write weights to output file
        with open(args.output, 'a') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            for index, row in user_weights.iterrows():
                writer.writerow([int(row['user_id']), cnty, row['weight']])



if __name__ == '__main__':
    args = get_args()
    main(args)
