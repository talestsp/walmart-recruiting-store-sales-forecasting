import pandas as pd


def build_submission_df(test_df, target_predicted, store_colname="Store", dept_colname="Dept", date_colname="Date"):
    if len(test_df) != len(target_predicted):
        raise Exception(
            "Teste dataset and target columns must have the same lengths: {}, {}".format(len(test_df),
                                                                                         len(target_predicted)))
    submission_df = pd.DataFrame()
    submission_df["Id"] = test_df[store_colname].astype(str) + "_" + \
                          test_df[dept_colname].astype(str) + "_" +\
                          test_df[date_colname].astype(str)
    submission_df["Weekly_Sales"] = target_predicted
    return submission_df.set_index("Id")

def evaluate(submission_df, validation, valid_store_colname="Store", valid_dept_colname="Dept",
             valid_date_colname="Date", valid_weekly_sales="Weekly_Sales"):

    validation = validation[[valid_store_colname, valid_dept_colname, valid_date_colname,
                             valid_weekly_sales, "IsHoliday"]].copy()
    validation["Id"] = validation[valid_store_colname].astype(str) + "_" + \
                       validation[valid_dept_colname].astype(str) + "_" + \
                       validation[valid_date_colname].astype(str)
    validation = validation.set_index("Id")
    abs_diff = (validation[valid_weekly_sales] - submission_df[valid_weekly_sales]).apply(abs)
    w = validation["IsHoliday"].replace({True: 5, False: 1})
    w = w.reindex(validation.index)

    return (w * abs_diff).sum() / w.sum()