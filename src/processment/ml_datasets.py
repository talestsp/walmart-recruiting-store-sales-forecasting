import pandas as pd


def dummy_baseline_data(data):
    dummy_baseline = []

    for (store_dept, wm_date), g in data.groupby(["store_dept", "wm_date"]):
        sorted_group = g[["year", "Weekly_Sales", "Size", "Store", "Dept", "Date", "IsHoliday"]].sort_values("year")
        store = sorted_group["Store"].iloc[0]
        dept = sorted_group["Dept"].iloc[0]
        date = sorted_group["Date"].iloc[0]
        is_holiday = sorted_group["IsHoliday"].iloc[0]

        dummy_baseline_row = {"store_dept": store_dept, "wm_date": wm_date,
                              "Store": store, "Dept": dept, "Date": date, "IsHoliday": is_holiday}

        for year_i in range(len(sorted_group)):
            year_n_label = "year" + str(year_i)
            year_sales_label = year_n_label + "_sales"
            size_n_label = year_n_label + "_size"

            year_value = sorted_group.iloc[year_i]["year"]
            x_value = sorted_group.iloc[year_i]["Weekly_Sales"]
            size_value = sorted_group.iloc[year_i]["Size"]

            dummy_baseline_row[year_n_label] = year_value
            dummy_baseline_row[year_sales_label] = x_value
            dummy_baseline_row[size_n_label] = size_value

        dummy_baseline.append(dummy_baseline_row)

    return pd.DataFrame(dummy_baseline)

def fix_low_sales(predicted_data, historical_data, threshold=0):
    return predicted_data.copy().apply(lambda row : set_week_mont_median(row, historical_data, threshold))

def set_week_mont_median(row, historical_data, threshold):
    if row["Weekly_sales"] <= threshold:
        refference = historical_data[(historical_data["Store"] == row["Store"]) &
                                     (historical_data["Dept"] == row["Dept"]) &
                                     (historical_data["Date"] == row["Date"])]["Weekly_sales"].median()
        row["Weekly_sales"] = refference
        return row
    else:
        return row
