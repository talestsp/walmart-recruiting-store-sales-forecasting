import pandas as pd
from src.utils import time_utils
import numpy as np


def features_semantic_enrichment(data):
    data["datetime"] = data["Date"].apply(lambda d: time_utils.str_to_datetime(d, "%Y-%m-%d"))
    data["day_n"] = data["datetime"].apply(lambda d: d.day).astype(int)
    data["month_n"] = data["datetime"].apply(lambda d: d.month).astype(str).apply(lambda mn : "0" + mn if int(mn) <= 9 else mn)
    data = week_n(data)
    data["celsius"] = (data["Temperature"] - 32) * 5 / 9
    data["wm_date"] = data["month_n"] + "/" + data["week_n"].astype(str)
    data = data.groupby("Store").apply(temperature_diff)
    data = data.groupby("Store").apply(holiday_pre_pos).reset_index(drop=True)

    return data

def week_n(data):
    data["date_ym"] = data["Date"].apply(lambda dt_str: dt_str[0:7])

    week_ns = []

    for date_ym in data["date_ym"].drop_duplicates():
        days_n = data[data["date_ym"] == date_ym]["day_n"].drop_duplicates().sort_values().tolist()

        for day_n in days_n:
            week_n = days_n.index(day_n) + 1
            week_ns.append({"date_ym": date_ym, "day_n": day_n, "week_n": week_n})

    week_ns = pd.DataFrame(week_ns)

    return data.merge(pd.DataFrame(week_ns), how="left", left_on=["date_ym", "day_n"], right_on=["date_ym", "day_n"])


def week_n_straight(day_n):
    return round( 1 + (day_n/7.6))

def sales_diff(group):
    group_sales = group.sort_values("timestamp")["Weekly_Sales"].iloc[1: len(group)].reset_index(drop=True)
    prev_group_sales = group.sort_values("timestamp")["Weekly_Sales"].iloc[0: len(group) - 1].reset_index(drop=True)
    return pd.Series([np.NaN] + (group_sales - prev_group_sales).tolist()).astype(float).tolist()

def sales_diff_percent(group):
    group_sales = group.sort_values("timestamp")["Weekly_Sales"].iloc[1: len(group)].reset_index(drop=True)
    prev_group_sales = group.sort_values("timestamp")["Weekly_Sales"].iloc[0: len(group) - 1].reset_index(drop=True)
    diff_p = (group_sales / prev_group_sales) - 1
    return pd.Series([np.NaN] + diff_p.tolist()).astype(float).tolist()

def temperature_diff(group):
    group_sales = group.sort_values("timestamp")["celsius"].iloc[1: len(group)].reset_index(drop=True)
    prev_group_sales = group.sort_values("timestamp")["celsius"].iloc[0: len(group) - 1].reset_index(drop=True)
    group["celsius_diff"] = pd.Series([np.NaN] + (group_sales - prev_group_sales).tolist()).astype(float).tolist()
    return group
# def temperature_diff(group):
#     group_sales = group.sort_values("timestamp")["celsius"].iloc[1: len(group)].reset_index(drop=True)
#     prev_group_sales = group.sort_values("timestamp")["celsius"].iloc[0: len(group) - 1].reset_index(drop=True)
#     return pd.Series([np.NaN] + (group_sales - prev_group_sales).tolist()).astype(float).tolist()

def holiday_pre_pos(store_data):
    #TODO - maybe fill NULL pre and pos holiday with seasonality holidays
    use_store_data = store_data[["Store", "Date", "IsHoliday", "timestamp"]].drop_duplicates().sort_values("timestamp")
    pre_holiday = use_store_data["IsHoliday"].iloc[1: len(use_store_data)].tolist()
    pos_holiday = use_store_data["IsHoliday"].iloc[0: len(use_store_data) - 1].tolist()

    use_store_data["pre_holiday"] = pre_holiday + [np.NaN]
    use_store_data["pos_holiday"] = [np.NaN] + pos_holiday

    store_data = store_data.merge(use_store_data, how="inner", left_on=["Store", "Date"],
                                  right_on=["Store", "Date"], suffixes=["", "_y"])

    del store_data["IsHoliday_y"]
    del store_data["timestamp_y"]

    return store_data

def train_sales_semantic_enrichment(data):
    data["sales_diff"] = sales_diff(data)
    data["sales_diff_p"] = sales_diff_percent(data)
    data["up_diff"] = data["sales_diff"].apply(lambda diff : False if diff <= 0 else True)

    return data

