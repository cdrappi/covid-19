import datetime
import json
import os
from typing import Dict, List

import pandas
import tweepy
from dateutil.parser import parse
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num, num2date
from matplotlib.patches import Patch

from common import get_realtime_r0_filename, load_data, load_meta, write_meta

START_DATE = '2019-02-29'
TWITTER_HANDLE = 'dfsalbatross'

FULL_COLOR = [.7, .7, .7]
NONE_COLOR = [179/255, 35/255, 14/255]
PARTIAL_COLOR = [.5, .5, .5]
ERROR_BAR_COLOR = [.3, .3, .3]


def plot_standings(mr, lockdowns: Dict[str, List[str]], figsize=None, title='Most Recent $R_t$ by State'):
    if not figsize:
        figsize = ((15.9/50)*len(mr)+.1, 10)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title)
    err = mr[['Low', 'High']].sub(mr['ML'], axis=0).abs()
    bars = ax.bar(mr.index,
                  mr['ML'],
                  width=.825,
                  color=FULL_COLOR,
                  ecolor=ERROR_BAR_COLOR,
                  capsize=2,
                  error_kw={'alpha': .5, 'lw': 1},
                  yerr=err.values.T)

    for bar, state_name in zip(bars, mr.index):
        if state_name in lockdowns['none']:
            bar.set_color(NONE_COLOR)
        if state_name in lockdowns['partial']:
            bar.set_color(PARTIAL_COLOR)

    labels = mr.index.to_series().replace({'District of Columbia': 'DC'})
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0, 2.)
    ax.axhline(1.0, linestyle=':', color='k', lw=1)

    leg = ax.legend(handles=[
        Patch(label='Full', color=FULL_COLOR),
        Patch(label='Partial', color=PARTIAL_COLOR),
        Patch(label='None', color=NONE_COLOR)
    ],
        title='Lockdown',
        ncol=3,
        loc='upper left',
        columnspacing=.75,
        handletextpad=.5,
        handlelength=1)

    leg._legend_box.align = "left"
    fig.set_facecolor('w')
    plt.tight_layout()
    return fig, ax


def save_states_plot(state_results_df: pandas.DataFrame, lockdowns: Dict, png_filename: str):
    mr = state_results_df[['ML', 'High', 'Low']].sort_values('High')
    fig, ax = plot_standings(mr, lockdowns)
    plt.savefig(png_filename)


def get_twitter_api() -> tweepy.API:
    auth = tweepy.OAuthHandler(os.environ['TWITTER_CONSUMER_KEY'],
                               os.environ['TWITTER_CONSUMER_SECRET'])
    auth.set_access_token(os.environ['TWITTER_ACCESS_TOKEN'],
                          os.environ['TWITTER_ACCESS_SECRET'])

    return tweepy.API(auth, wait_on_rate_limit=True,
                      wait_on_rate_limit_notify=True,
                      compression=True)


if __name__ == '__main__':
    meta = load_meta()
    data = load_data()
    lockdowns = json.load(open('lockdowns.json'))

    try:
        last_tweet_date = max(meta['tweets'])
        last_tweet_id = meta['tweets'][last_tweet_date]['id']
    except ValueError:
        last_tweet_date = START_DATE
        last_tweet_id = None

    if last_tweet_date < data['date'].max():
        next_tweet_date = parse(last_tweet_date).date() + \
            datetime.timedelta(days=1)
        next_tweet_str = f'{next_tweet_date:%Y-%m-%d}'
        png_filename = get_realtime_r0_filename(next_tweet_date)

        api = get_twitter_api()
        media_id = api.media_upload(png_filename)
        tweet = api.update_status(
            status=f'R(t) @ {next_tweet_str})',
            media_ids=[media_id],
            in_reply_to_status_id=last_tweet_id
        )
        meta['tweets'][next_tweet_str] = {
            'id': tweet.id,
            'media': [media_id],
            'link': f'https://twitter.com/{TWITTER_HANDLE}/status/{tweet.id}'
        }
        write_meta(meta)
