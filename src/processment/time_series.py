import pandas as pd
from src.utils import stats


def time_series_similarity(ts1, ts2, colname):
    ts = ts1[[colname]].merge(ts2[[colname]],
                              how="inner", left_index=True, right_index=True, suffixes=["_ts1","_ts2"])
    return abs((ts[colname+"_ts1"] - ts[colname+"_ts2"]).mean())

def time_series_similarity_normalized(ts1, ts2, colname):
    ts = ts1[[colname]].merge(ts2[[colname]],
                              how="inner", left_index=True, right_index=True, suffixes=["_ts1","_ts2"])
    return abs((stats.normalize(ts[colname+"_ts1"]) - stats.normalize(ts[colname+"_ts2"])).mean())
