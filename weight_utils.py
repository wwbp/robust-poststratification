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

    # fill missing entries for naive post-stratification
    user_data['weight'].fillna(user_data['weight'].mean(), inplace=True)
    user_data['weight'] = user_data['weight'] / user_data['weight'].sum() * len(user_data)

    return user_data


def create_weights_single(user_data, population_subset, dem, smoothing, min_bin_num, smooth_before_binning, uninformed_smoothing, user_bins, population_cols):
    bins, user_percents, population_percents = bin_utils.get_bins(
        user_data=user_data,
        population_data=population_subset,
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

    all_targets = []
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

        demographic_targets = {}
        demographic_targets[dem + "_banded"] = {i+1:perc for i, perc in enumerate(population_percents)}
        all_sample_dems[dem] = {i+1:perc for i, perc in enumerate(user_percents)}
        all_population_dems[dem] = {i+1:perc for i, perc in enumerate(population_percents)}
        all_targets.append(demographic_targets)
    
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
    
    scheme = qp.Rim("_".join(demographics)+'_scheme')
    scheme.set_targets(targets=all_targets, group_name="_".join(demographics) + ' weights')

    if len(demographics) == 2:
        user_weights = rake_utils.create_raked_df_from_two(
            df=dataset.data(),
            dem=demographics,
            pop_percentages=all_population_dems,
            smoothed_k=smoothing,
            uninformed_smoothing=uninformed_smoothing,
        )
    else:
        user_weights = rake_utils.create_raked_df_from_four(
            df=dataset.data(),
            dem=demographics,
            pop_percentages=all_population_dems,
            smoothed_k=smoothing,
            uninformed_smoothing=uninformed_smoothing,
        )

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

    if set(demographics) == set(['age', 'gender']):
        combined_targets = {'age_gender_banded': {}}
        dataset.data()['age_gender_banded'] = 0
        age_list = sorted(data.age_banded.unique())
        gen_list = sorted(data.gender_banded.unique())
        group_dict = dataset.data().groupby(['age_banded', 'gender_banded']).agg(['count'])['user_id'].to_dict()

        i = 0
        skipped = False
        for ia, a in enumerate(age_list):
            for ig, g in enumerate(gen_list):
                try:
                    if group_dict['count'][(a, g)] < 1: 
                        skipped = True
                        continue
                except:
                    skipped = True
                    continue
                combined_targets['age_gender_banded'][i+1] = all_targets[0]['age_banded'][a] * all_targets[1]['gender_banded'][g] / 100
                query = '(age_banded == {a}) and (gender_banded == {g})'.format(a=str(a), g=str(g))
                dataset.data().loc[data.eval(query), 'age_gender_banded'] = i + 1
                i += 1
        
        if skipped:
            # renormalize combined_targets
            s = sum([v for kkey,v in combined_targets['age_gender_banded'].items()])
            combined_targets['age_gender_banded'] = {kkey: v*100/s for kkey, v in combined_targets['age_gender_banded'].items()}
        elif round(sum([v for kkey,v in combined_targets['age_gender_banded'].items()])) != 100:
            s = sum([v for kkey,v in combined_targets['age_gender_banded'].items()])
            combined_targets['age_gender_banded'] = {kkey: v*100/s for kkey, v in combined_targets['age_gender_banded'].items()}

        dataset.data().drop(['age_banded', 'gender_banded'], axis=1, inplace=True)

        sorted_bins = sorted(list(combined_targets['age_gender_banded'].keys()))
        sorted_targets = [combined_targets['age_gender_banded'][kkey] for kkey in sorted_bins]
        sorted_sample_targets = dataset.data().groupby(['age_gender_banded']).agg(['count'])['user_id'].to_dict()['count']
        t = float(sum(sorted_sample_targets.values()))
        sorted_sample_targets = {kkey:v/t for kkey,v in sorted_sample_targets.items()}
        sorted_sample_targets = [sorted_sample_targets[kkey] for kkey in sorted_bins]

        print(np.round(sorted_bins, 3))
        print(np.round(sorted_sample_targets, 3))
        print(np.round(sorted_targets, 3))

        user_weights = create_weights(
            user_data=dataset.data(),
            bins=sorted_bins,
            user_percents=sorted_sample_targets,
            population_percents=sorted_targets,
            dem='age_gender_banded',
            smoothing=smoothing,
            smooth_before_binning=smooth_before_binning,
            uninformed_smoothing=uninformed_smoothing
        )

    elif set(demographics) == set(['income', 'education']):
        combined_targets = {'income_education_banded': {}}
        dataset.data()['income_education_banded'] = 0
        inc_list = sorted(data.income_banded.unique())
        edu_list = sorted(data.education_banded.unique())
        group_dict = dataset.data().groupby(['income_banded', 'education_banded']).agg(['count'])['user_id'].to_dict()

        i = 0
        skipped = False
        for ia, a in enumerate(inc_list):
            for ig, g in enumerate(edu_list):
                try:
                    if group_dict['count'][(a, g)] < 1: 
                        skipped = True
                        continue
                except:
                    skipped = True
                    continue
                combined_targets['income_education_banded'][i+1] = all_targets[0]['income_banded'][a] * all_targets[1]['education_banded'][g] / 100
                query = '(income_banded == {a}) and (education_banded == {g})'.format(a=str(a), g=str(g))
                dataset.data().loc[data.eval(query), 'income_education_banded'] = i+1
                i += 1
        
        if skipped:
            #renormalize combined_targets
            s = sum([v for kkey,v in combined_targets['income_education_banded'].items()])
            combined_targets['income_education_banded'] = {kkey: v*100/s for kkey, v in combined_targets['income_education_banded'].items()}
        elif round(sum([v for kkey,v in combined_targets['income_education_banded'].items()])) != 100:
            s = sum([v for kkey,v in combined_targets['income_education_banded'].items()])
            combined_targets['income_education_banded'] = {kkey: v*100/s for kkey, v in combined_targets['income_education_banded'].items()}
        
        dataset.data().drop(['income_banded', 'education_banded'], axis=1, inplace=True)

        sorted_bins = sorted(list(combined_targets['income_education_banded'].keys()))
        sorted_targets = [combined_targets['income_education_banded'][kkey] for kkey in sorted_bins]
        sorted_sample_targets = dataset.data().groupby(['income_education_banded']).agg(['count'])['user_id'].to_dict()['count']
        t = float(sum(sorted_sample_targets.values()))
        sorted_sample_targets = {kkey:v/t for kkey,v in sorted_sample_targets.items()}
        sorted_sample_targets = [sorted_sample_targets[kkey] for kkey in sorted_bins]

        user_weights = create_weights(
            user_data=dataset.data(),
            bins=sorted_bins,
            user_percents=sorted_sample_targets,
            population_percents=sorted_targets,
            dem='income_education_banded',
            smoothing=smoothing,
            smooth_before_binning=smooth_before_binning,
            uninformed_smoothing=uninformed_smoothing
        )

    elif set(demographics) == set(['age', 'gender', 'income', 'education']):
        combined_targets = {'age_gender_income_education_banded': {}}
        dataset.data()['age_gender_income_education_banded'] = 0
        age_list = sorted(data.age_banded.unique())
        gen_list = sorted(data.gender_banded.unique())
        inc_list = sorted(data.income_banded.unique())
        edu_list = sorted(data.education_banded.unique())
        group_dict = dataset.data().groupby(['age_banded', 'gender_banded', 'income_banded', 'education_banded']).agg(['count'])['user_id'].to_dict()

        i = 0
        skipped = False
        for a in age_list:
            for g in gen_list:
                for ii in inc_list:
                    for e in edu_list:
                        try:
                            if group_dict['count'][(a, g, ii, e)] < 1: 
                                skipped = True
                                continue
                        except:
                            skipped = True
                            continue
                        combined_targets['age_gender_income_education_banded'][i+1] = all_targets[0]['age_banded'][a] * all_targets[2]['gender_banded'][g] * all_targets[1]['income_banded'][ii] * all_targets[3]['education_banded'][e] / (100*100*100)
                        query = '(age_banded == {a}) and (gender_banded == {g}) and (income_banded == {ii}) and (education_banded == {e})'.format(a=str(a), g=str(g), ii=str(ii), e=str(e))
                        dataset.data().loc[data.eval(query), 'age_gender_income_education_banded'] = i+1
                        i += 1

        if skipped:
            #renormalize combined_targets
            s = sum([v for kkey,v in combined_targets['age_gender_income_education_banded'].items()])
            combined_targets['age_gender_income_education_banded'] = {kkey: v*100/s for kkey, v in combined_targets['age_gender_income_education_banded'].items()}
        elif round(sum([v for kkey,v in combined_targets['age_gender_income_education_banded'].items()])) != 100:
            s = sum([v for kkey,v in combined_targets['age_gender_income_education_banded'].items()])
            combined_targets['age_gender_income_education_banded'] = {kkey: v*100/s for kkey, v in combined_targets['age_gender_income_education_banded'].items()}

        dataset.data().drop(['age_banded', 'gender_banded', 'income_banded', 'education_banded'], axis=1, inplace=True)

        sorted_bins = sorted(list(combined_targets['age_gender_income_education_banded'].keys()))
        sorted_targets = [combined_targets['age_gender_income_education_banded'][kkey] for kkey in sorted_bins]
        sorted_sample_targets = dataset.data().groupby(['age_gender_income_education_banded']).agg(['count'])['user_id'].to_dict()['count']
        t = float(sum(sorted_sample_targets.values()))
        sorted_sample_targets = {kkey:v/t for kkey,v in sorted_sample_targets.items()}
        sorted_sample_targets = [sorted_sample_targets[kkey] for kkey in sorted_bins]

        user_weights = create_weights(
            user_data=dataset.data(),
            bins=sorted_bins,
            user_percents=sorted_sample_targets,
            population_percents=sorted_targets,
            dem='age_gender_income_education_banded',
            smoothing=smoothing,
            smooth_before_binning=smooth_before_binning,
            uninformed_smoothing=uninformed_smoothing
        )
    
    return user_weights
