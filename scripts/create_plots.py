import argparse
import datetime
import json
import logging
import os
from typing import Dict, List

import pandas
from dateutil.parser import parse
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num, num2date
from matplotlib.patches import Patch

from common import get_realtime_r0_filename, load_data, load_meta, write_meta

R_T_PLOTTING_MAX = 8.0
DEFAULT_START_DATE = '2020-03-01'
NOT_YET_STARTED_DATE = '2100-01-01'
LATEST_REOPENING_DATE = '2100-01-01'

FULL_COLOR = [.7, .7, .7]
ERROR_BAR_COLOR = [.3, .3, .3]

NONE_COLOR = [179/255, 35/255, 14/255]
NO_SHELTER_COLOR = [255/255, 204/255, 203/205]
NO_EMERGENCY_COLOR = [178/255, 70/255, 55/255]

logger = logging.getLogger(__name__)


def plot_standings(mr, lockdowns: Dict[str, List[str]], title: str, figsize=None):
    if not figsize:
        figsize = ((15.9/50)*len(mr)+.1, 10)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title)
    err = mr[['Low', 'High']].sub(mr['ML'], axis=0).abs()
    bars = ax.bar(mr['state'],
                  mr['ML'],
                  width=.825,
                  color=FULL_COLOR,
                  ecolor=ERROR_BAR_COLOR,
                  capsize=2,
                  error_kw={'alpha': .5, 'lw': 1},
                  yerr=err.values.T)

    for bar, state_name in zip(bars, mr['state']):
        no_emergency = state_name in lockdowns['no_emergency']
        no_shelter = state_name in lockdowns['no_shelter']
        if no_emergency and no_shelter:
            bar.set_color(NONE_COLOR)
        elif no_shelter:
            bar.set_color(NO_SHELTER_COLOR)
        elif no_emergency:
            bar.set_color(NO_EMERGENCY_COLOR)

    labels = mr['state'].replace({'District of Columbia': 'DC'})
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0, R_T_PLOTTING_MAX)
    ax.axhline(1.0, linestyle=':', color='k', lw=1)

    leg = ax.legend(handles=[
        Patch(label='Stay at Home Ordered', color=FULL_COLOR),
        Patch(label='State of Emergency Only', color=NO_SHELTER_COLOR),
        Patch(label='None', color=NONE_COLOR),
    ],
        title='Lockdown',
        ncol=1,
        loc='upper left',
        columnspacing=.75,
        handletextpad=.5,
        handlelength=1)

    leg._legend_box.align = "left"
    fig.set_facecolor('w')
    plt.tight_layout()
    return fig, ax


def save_states_plot(state_results_df: pandas.DataFrame, lockdowns: Dict, png_filename: str, plot_title):
    mr = state_results_df[['state', 'ML', 'High', 'Low']].sort_values('High')
    fig, ax = plot_standings(mr, lockdowns, title=plot_title)
    plt.savefig(png_filename)


def _is_locked_down(lockdown_data: Dict[str, str], date_string: str) -> bool:
    lockdown_start = lockdown_data['start'] or NOT_YET_STARTED_DATE
    lockdown_end = lockdown_data['end'] or LATEST_REOPENING_DATE
    return lockdown_start < date_string < lockdown_end


def get_lockdowns_by_date(lockdown_time_series, date_string):
    lockdowns = {'no_emergency': set(), 'no_shelter': set()}
    for state, state_lockdowns in lockdown_time_series.items():
        if not _is_locked_down(state_lockdowns['state_of_emergency'], date_string):
            lockdowns['no_emergency'].add(state)
        if not _is_locked_down(state_lockdowns['shelter_in_place'], date_string):
            lockdowns['no_shelter'].add(state)
    return lockdowns


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--log',
        default='INFO',
        type=str,
        help='Log level'
    )
    arg_parser.add_argument(
        '--start_date',
        default=DEFAULT_START_DATE,
        type=str
    )
    args = arg_parser.parse_args()
    logging.basicConfig(level=args.log)

    meta = load_meta()
    data = load_data()

    lockdown_time_series = json.load(open('data/lockdown_time_series.json'))

    plot_date = parse(args.start_date).date()
    max_data_date = parse(data['date'].max()).date()

    if not os.path.isdir('data/plots/realtime_r0'):
        os.mkdir('data/plots/realtime_r0')

    while plot_date < max_data_date:
        plot_date_str = f'{plot_date:%Y-%m-%d}'
        png_filename = get_realtime_r0_filename(plot_date)
        if not os.path.isfile(png_filename):
            save_states_plot(
                state_results_df=data[data['date'] == plot_date_str],
                lockdowns=get_lockdowns_by_date(
                    lockdown_time_series, plot_date_str),
                png_filename=png_filename,
                plot_title=f"State-level $R(t)$ as of {plot_date_str}"
            )
            logger.info(f'created R(t) for {plot_date} @ {png_filename}...')
        plot_date += datetime.timedelta(days=1)
