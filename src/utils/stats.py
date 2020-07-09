import pandas as pd
import numpy as np
import scipy.stats


def freq(pd_series):
    '''
        Returns a pandas.DataFrame with absolute and relative frequencies
    '''
    proportional = pd_series.astype(str).value_counts(normalize=True).rename("freq_relative")
    proportional = proportional * 100
    proportional = proportional.apply(lambda value : "{0:.2f}".format(round(value, 2)) + "%")

    absolute = pd_series.astype(str).value_counts().rename("freq_absolute")

    return pd.concat([absolute, proportional], axis=1)

def relative_frequencies(series):
    unity_value_percent = 100.0 / series.sum()
    return unity_value_percent * series

def abs_diff(a, b):
    return abs(a - b)

def mean_confidence_interval(data, confidence=0.75):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def normalize(series):
    return {"norm": (series-series.min()) / (series.max() - series.min()),
            "min": series.min(),
            "max": series.max()}

def revert_normalize(norm, s_min, s_max):
    return norm * (s_max-s_min) + 1
