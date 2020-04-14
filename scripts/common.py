import datetime
import json
from typing import Any, Dict

import pandas

R_T_MAX = 12

LATEST_PROCESSED = 'data/latest_processed.csv'

FILTERED_REGIONS = [
    'Virgin Islands',
    'American Samoa',
    'Northern Mariana Islands',
    'Guam',
    'Puerto Rico'
]


def get_realtime_r0_filename(date_obj: datetime.date) -> str:
    return f'data/plots/realtime_r0/{date_obj:%Y-%m-%d}.png'


def load_data() -> pandas.DataFrame:
    results_df = pandas.read_csv(LATEST_PROCESSED)
    results_df = results_df.rename(columns={'Unnamed: 0': 'state'})
    results_df = results_df[~results_df['state'].isin(FILTERED_REGIONS)]
    return results_df


def load_meta() -> Dict[str, Any]:
    try:
        return json.load(open('data/meta.json'))
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        return {"tweets": {}}


def write_meta(meta_json):
    with open('data/meta.json', 'w') as f:
        json.dump(meta_json, f)
