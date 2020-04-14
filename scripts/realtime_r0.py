#!/usr/bin/env python
# coding: utf-8

# Estimating COVID-19's $R_t$ in Real-Time
# Kevin Systrom - April 12
# https://github.com/k-sys/covid-19

import argparse
import json
import logging
import os
import time
import traceback
from typing import Dict, List

import numpy
import pandas
import tweepy
from scipy import stats as sps
from scipy.interpolate import interp1d

from common import (FILTERED_REGIONS, LATEST_PROCESSED, R_T_MAX, load_data,
                    load_meta, write_meta)

# We create an array for every possible value of Rt
r_t_range = numpy.linspace(0, R_T_MAX, R_T_MAX*100+1)

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
GAMMA = 1/4


THE_FAILING_NYT_URL = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'


logger = logging.getLogger(__name__)


def get_posteriors(sr, window=7, min_periods=1) -> numpy.array:
    lam = sr[:-1].values * numpy.exp(GAMMA * (r_t_range[:, None] - 1))

    # Note: if you want to have a Uniform prior you can use the following line instead.
    # I chose the gamma distribution because of our prior knowledge of the likely value
    # of R_t.

    # prior0 = numpy.full(len(r_t_range), numpy.log(1/len(r_t_range)))
    prior0 = numpy.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)

    likelihoods = pandas.DataFrame(
        # Short-hand way of concatenating the prior and likelihoods
        data=numpy.c_[prior0, sps.poisson.logpmf(sr[1:].values, lam)],
        index=r_t_range,
        columns=sr.index)

    # Perform a rolling sum of log likelihoods. This is the equivalent
    # of multiplying the original distributions. Exponentiate to move
    # out of log.
    posteriors = likelihoods.rolling(window,
                                     axis=1,
                                     min_periods=min_periods).sum()
    posteriors = numpy.exp(posteriors)

    # Normalize to 1.0
    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)

    return posteriors


def prepare_cases(cases) -> (pandas.Series, pandas.Series):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(7,
                                 win_type='gaussian',
                                 min_periods=1,
                                 center=True).mean(std=2).round()

    zeros = smoothed.index[smoothed.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed.index.get_loc(last_zero) + 1
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


def highest_density_interval(pmf, p=.95) -> pandas.Series:

    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pandas.DataFrame)):
        return pandas.DataFrame([highest_density_interval(pmf[col]) for col in pmf],
                                index=pmf.columns)

    cumsum = numpy.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i+1:]):
            if (high_value-value > p) and (not best or j < best[1]-best[0]):
                best = (i, i+j+1)
                break

    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pandas.Series([low, high], index=['Low', 'High'])


def states_to_realtime_r0s(states) -> Dict[str, pandas.DataFrame]:
    results = {}

    states_to_process = (
        states
        .loc[~states.index.get_level_values('state').isin(FILTERED_REGIONS)]
    )

    for state_name, cases in states_to_process.groupby(level='state'):
        print(f'Processing {state_name}')
        new, smoothed = prepare_cases(cases)
        try:
            print('\tGetting Posteriors')
            posteriors = get_posteriors(smoothed)
            print('\tGetting HDIs')
            hdis = highest_density_interval(posteriors)
            print('\tGetting most likely values')
            most_likely = posteriors.idxmax().rename('ML')
            result = pandas.concat([most_likely, hdis], axis=1)
            results[state_name] = result.droplevel(0)
        except Exception as e:
            print(f'{e}\n{cases}')
            print(traceback.format_exc())

    return results


def realtime_r0s_to_df(results) -> pandas.DataFrame:

    overall = None
    for state_name, result in results.items():
        r = result.copy()
        r.index = pandas.MultiIndex.from_product([[state_name], result.index])
        if overall is None:
            overall = r
        else:
            overall = pandas.concat([overall, r])

    overall.sort_index(inplace=True)
    return overall


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log', default='INFO',
                            type=str, help='Log level')
    args = arg_parser.parse_args()
    logging.basicConfig(level=args.log)

    start_time = time.time()

    try:
        results_df = load_data()
        latest_calc = results_df['date'].max()
        logger.info(
            f'loaded previous data, '
            f'including calculating through {latest_calc}'
        )
    except FileNotFoundError:
        latest_calc = '1970-01-01'
        logger.warning('file not found, recalcuting model over all data')

    states = pandas.read_csv(THE_FAILING_NYT_URL,
                             usecols=[0, 1, 3],
                             index_col=['state', 'date'],
                             parse_dates=['date'],
                             squeeze=True).sort_index()
    states.to_csv(f'data/raw_backups/nyt/{start_time:.0f}.csv')
    latest_data = states.index.get_level_values(1).max()

    if latest_calc < f'{latest_data:%Y-%m-%d}':
        logger.info(
            'crunching numbers from the fake news new york times. '
            'who knows if they are even be accurate? garbage in, garbage out!'
        )
        realtime_r0s = states_to_realtime_r0s(states)
        results_df = realtime_r0s_to_df(realtime_r0s)
        results_df.index.names = ['state', 'date']
        results_df.to_csv(LATEST_PROCESSED)

        latest_processed_backup = f'data/processed_backups/{start_time:.0f}.csv'
        results_df.to_csv(latest_processed_backup)
        meta = load_meta()
        meta['latest_processed_backup'] = latest_processed_backup
        write_meta(meta)
