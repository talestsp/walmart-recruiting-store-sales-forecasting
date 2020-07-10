import pandas as pd
from src.utils import time_utils
import numpy as np


def features_semantic_enrichment(data):
    data["datetime"] = data["Date"].apply(lambda d: time_utils.str_to_datetime(d, "%Y-%m-%d"))
    data["day_n"] = data["datetime"].apply(lambda d: d.day).astype(int)
    data["month_n"] = data["datetime"].apply(lambda d: d.month).astype(str).apply(lambda mn : "0" + mn if int(mn) <= 9 else mn)
    data["year"] = data["datetime"].apply(lambda d: d.year).astype(int)
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


def wm_data_BKP(data):
    transformed_data = []

    for (store_dept, wm_date), g in data.groupby(["store_dept", "wm_date"]):
        sorted_group = g[["year", "Weekly_Sales", "Size", "Store", "Dept", "Date", "IsHoliday"]].sort_values("year")
        store = sorted_group["Store"].iloc[0]
        dept = sorted_group["Dept"].iloc[0]
        date = sorted_group["Date"].iloc[0]
        is_holiday = sorted_group["IsHoliday"].iloc[0]

        transformed_data_row = {"store_dept": store_dept, "wm_date": wm_date,
                              "Store": store, "Dept": dept, "Date": date, "IsHoliday": is_holiday}

        for year_i in range(len(sorted_group)):
            year_n_label = "year" + str(year_i)
            year_sales_label = year_n_label + "_sales"
            size_n_label = year_n_label + "_size"

            year_value = sorted_group.iloc[year_i]["year"]
            x_value = sorted_group.iloc[year_i]["Weekly_Sales"]
            size_value = sorted_group.iloc[year_i]["Size"]

            transformed_data_row[year_n_label] = year_value
            transformed_data_row[year_sales_label] = x_value
            transformed_data_row[size_n_label] = size_value

        transformed_data.append(transformed_data_row)

    return pd.DataFrame(transformed_data)


def wm_data(data):
    transformed_data = []

    for (store_dept, wm_date), g in data.groupby(["store_dept", "wm_date"]):
        sorted_group = g[["year", "Weekly_Sales", "Size", "Store", "Dept", "Date", "IsHoliday",
                          "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
                          "Fuel_Price", "CPI", "Unemployment", "week_n"]].sort_values("year")

        store = sorted_group["Store"].iloc[0]
        dept = sorted_group["Dept"].iloc[0]
        date = sorted_group["Date"].iloc[0]
        #         is_holiday = sorted_group["IsHoliday"].iloc[0]

        transformed_data_row = {"store_dept": store_dept, "wm_date": wm_date,
                              "Store": store, "Dept": dept, "Date": date}

        for year_i in range(len(sorted_group)):
            year_n_label = "year" + str(year_i)
            year_n_sales_label = year_n_label + "_sales"
            size_n_label = year_n_label + "_size"
            isholiday_n_label = year_n_label + "_isholiday"
            fuel_price_n_label = year_n_label + "_fuel_price"
            cpi_n_label = year_n_label + "_cpi"
            unempl_n_label = year_n_label + "_unempl"
            week_n_n_label = year_n_label + "_week_n"
            md1_n_label = year_n_label + "_md1"
            md2_n_label = year_n_label + "_md2"
            md3_n_label = year_n_label + "_md3"
            md4_n_label = year_n_label + "_md4"
            md5_n_label = year_n_label + "_md5"

            year_value = sorted_group.iloc[year_i]["year"] - 2010
            x_value = sorted_group.iloc[year_i]["Weekly_Sales"]
            size_value = sorted_group.iloc[year_i]["Size"]
            isholiday_value = sorted_group.iloc[year_i]["IsHoliday"]
            fuel_price_value = sorted_group.iloc[year_i]["Fuel_Price"]
            cpi_value = sorted_group.iloc[year_i]["CPI"]
            unempl_value = sorted_group.iloc[year_i]["Unemployment"]
            week_n_value = sorted_group.iloc[year_i]["week_n"]
            md1_value = sorted_group.iloc[year_i]["MarkDown1"]
            md2_value = sorted_group.iloc[year_i]["MarkDown2"]
            md3_value = sorted_group.iloc[year_i]["MarkDown3"]
            md4_value = sorted_group.iloc[year_i]["MarkDown4"]
            md5_value = sorted_group.iloc[year_i]["MarkDown5"]

            transformed_data_row[year_n_label] = year_value
            transformed_data_row[year_n_sales_label] = x_value
            transformed_data_row[size_n_label] = size_value
            transformed_data_row[isholiday_n_label] = isholiday_value
            transformed_data_row[fuel_price_n_label] = fuel_price_value
            transformed_data_row[cpi_n_label] = cpi_value
            transformed_data_row[unempl_n_label] = unempl_value
            transformed_data_row[week_n_n_label] = week_n_value
            transformed_data_row[md1_n_label] = md1_value
            transformed_data_row[md2_n_label] = md2_value
            transformed_data_row[md3_n_label] = md3_value
            transformed_data_row[md4_n_label] = md4_value
            transformed_data_row[md5_n_label] = md5_value

        transformed_data.append(transformed_data_row)

    return pd.DataFrame(transformed_data)

def format_wm_data_colnames(data, dataset_name):
    print("Total groups: ", len(data.drop_duplicates(["wm_date", "store_dept"])))
    print(len(data))
    data = data.rename({"Store": "Store" + "_" + dataset_name,
                        "Dept": "Dept" + "_" + dataset_name,
                        "Date": "Date" + "_" + dataset_name,

                        "year0": "year0" + "_" + dataset_name,
                        "year0_sales": "year0_sales" + "_" + dataset_name,
                        "year0_size": "year0_size" + "_" + dataset_name,
                        "year0_isholiday": "year0_isholiday" + "_" + dataset_name,
                        "year0_fuel_price": "year0_fuel_price" + "_" + dataset_name,
                        "year0_cpi": "year0_cpi" + "_" + dataset_name,
                        "year0_unempl": "year0_unempl" + "_" + dataset_name,
                        "year0_week_n": "year0_week_n" + "_" + dataset_name,
                        "year0_md1": "year0_md1" + "_" + dataset_name,
                        "year0_md2": "year0_md2" + "_" + dataset_name,
                        "year0_md3": "year0_md3" + "_" + dataset_name,
                        "year0_md4": "year0_md4" + "_" + dataset_name,
                        "year0_md5": "year0_md5" + "_" + dataset_name,

                        "year1": "year1" + "_" + dataset_name,
                        "year1_sales": "year1_sales" + "_" + dataset_name,
                        "year1_size": "year1_size" + "_" + dataset_name,
                        "year1_isholiday": "year1_isholiday" + "_" + dataset_name,
                        "year1_fuel_price": "year1_fuel_price" + "_" + dataset_name,
                        "year1_cpi": "year1_cpi" + "_" + dataset_name,
                        "year1_unempl": "year1_unempl" + "_" + dataset_name,
                        "year1_week_n": "year1_week_n" + "_" + dataset_name,
                        "year1_md1": "year1_md1" + "_" + dataset_name,
                        "year1_md2": "year1_md2" + "_" + dataset_name,
                        "year1_md3": "year1_md3" + "_" + dataset_name,
                        "year1_md4": "year1_md4" + "_" + dataset_name,
                        "year1_md5": "year1_md5" + "_" + dataset_name}, axis=1)

    return data

def dummy_fill_store_dept_median(row, refference_data, fill_colnames):
    store_dept_data = refference_data[refference_data["store_dept"] == row["store_dept"]]
    for fill_colname in fill_colnames:
        row[fill_colname] = store_dept_data[fill_colname].median()
    return row

def dummy_fill_store_median(row, refference_data, fill_colnames):
    store_data = refference_data[refference_data["Store_train"] == row["Store_test"]]
    for fill_colname in fill_colnames:
        row[fill_colname] = store_data[fill_colname].median()
    return row