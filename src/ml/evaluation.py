import pandas as pd


def build_submission_df(test_df, target):
    if len(test_df) != len(target):
        raise Exception(
            "Teste dataset and target columns must have the same lengths: {}, {}".format(len(test_df), len(target)))
    submission_df = pd.DataFrame()
    submission_df["Id"] = test_df["Store"] + "_" + test_df["Dept"] + "_" + test_df["Date"]
    submission_df["Weekly_Sales"] = target
    return submission_df.set_index("Id")

def evaluate(submission_df, validation):
    validation = validation[["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]].copy()
    validation["Id"] = validation["Store"] + "_" + validation["Dept"] + "_" + validation["Date"]
    validation = validation.set_index("Id")
    abs_diff = (validation["Weekly_Sales"] - submission_df["Weekly_Sales"]).apply(abs)
    w = validation["IsHoliday"].replace({True: 5, False: 1})
    w = w.reindex(validation.index)

    return (w * abs_diff).sum() / w.sum()