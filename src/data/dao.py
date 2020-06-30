import pandas as pd
from src.utils import time_utils

DTYPE = {"Store": str, "Dept": str, "Date": str, "Weekly_Sales": float, "IsHoliday": bool, "Temperature": float,
         "Fuel_Price": float, "MarkDown1": float, "MarkDown2": float,"MarkDown3": float,"MarkDown4": float,
         "MarkDown5": float, "CPI": float, "Unemployment": float, "Type": str,
         "Size": int}

def load_dataset(dataset):
    data = pd.read_csv("data/raw/{}.csv".format(dataset), dtype=DTYPE)
    data["timestamp"] = data["Date"].apply(lambda str_dt: time_utils.str_datetime_to_timestamp(str_dt, "%Y-%m-%d"))
    data["store_dept"] = data["Store"] + "_" + data["Dept"]
    return data.sort_values("timestamp")

def load_features():
    feat = pd.read_csv("data/raw/features.csv", dtype=DTYPE)
    feat["timestamp"] = feat["Date"].apply(lambda str_dt: time_utils.str_datetime_to_timestamp(str_dt, "%Y-%m-%d"))
    return feat.sort_values("timestamp")

def load_stores():
    return pd.read_csv("data/raw/stores.csv", dtype=DTYPE)

