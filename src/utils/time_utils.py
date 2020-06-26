import datetime
import pandas as pd
import numbers

HOUR_SECONDS = 3600.0

def str_to_datetime(str_dt, str_dt_format='%d/%m/%Y %H:%M:%S'):
    '''
    Returns a datetime.datetime object
    :param str_dt: string representing date and/or time
    :param str_dt_format:
    :return:
    '''
    use_str_dt = str_dt.split(".")[0]
    return datetime.datetime.strptime(use_str_dt, str_dt_format)

def str_to_datetime_try_formats(str_dt):
    try:
        return str_to_datetime(str_dt, str_dt_format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        return str_to_datetime(str_dt, str_dt_format='%Y/%m/%d %H:%M:%S')

def delta_time_seconds(str_dt_initial, str_dt_final, str_dt_format='%Y-%m-%d %H:%M:%S'):
    '''
    Returns the time difference (seconds) between two strings representing date and/time
    :param str_dt_initial: string representing date and/or time
    :param str_dt_final: string representing date and/or time
    :return:
    '''

    if type(str_dt_initial) is float or type(str_dt_initial) is int or isinstance(str_dt_initial, numbers.Number):
        str_dt_initial = int(round(str_dt_initial))
        if len(str(str_dt_initial)) == 13:
            str_dt_initial = timestamp_to_datetime(str_dt_initial / 1000)
        elif len(str(str_dt_initial)) == 10:
            str_dt_initial = timestamp_to_datetime(str_dt_initial)

    elif (not type(str_dt_initial) is datetime.datetime) and (not type(str_dt_initial) is pd.Timestamp):
        str_dt_initial = str_to_datetime(str_dt_initial, str_dt_format)


    if type(str_dt_final) is float or type(str_dt_final) is int or isinstance(str_dt_final, numbers.Number):
        str_dt_final = int(round(str_dt_final))
        if len(str(str_dt_final)) == 13:
            str_dt_final = timestamp_to_datetime(str_dt_final / 1000)
        elif len(str(str_dt_final)) == 10:
            str_dt_final = timestamp_to_datetime(str_dt_final)

    elif (not type(str_dt_final) is datetime.datetime) and (not type(str_dt_final) is pd.Timestamp):
        str_dt_final = str_to_datetime(str_dt_final, str_dt_format)

    return abs((str_dt_final - str_dt_initial).total_seconds())

def str_datetime_to_timestamp(str_datetime, str_dt_format='%Y-%m-%d %H:%M:%S'):
    use_str_dt = str_datetime.split(".")[0]
    dt = str_to_datetime(use_str_dt, str_dt_format)
    return datetime.datetime.timestamp(dt)

def timestamp_to_datetime(ts):
    return datetime.datetime.fromtimestamp(ts)
    
def delta_time_hour(str_dt_initial, str_dt_final, str_dt_format='%Y-%m-%d %H:%M:%S'):
    return delta_time_seconds(str_dt_initial, str_dt_final, str_dt_format=str_dt_format) / HOUR_SECONDS

def dt_to_str(dt, str_format):
    return dt.strftime(str_format)

def str_datetime_n_days_before(str_dt, n_delta_days, str_dt_format="%d/%m/%Y"):
    if n_delta_days == 0:
        return str_dt
    else:
        dt_n_days_before = str_to_datetime(str_dt, str_dt_format) + datetime.timedelta(days=n_delta_days)
        return dt_n_days_before.strftime(str_dt_format)

def is_commerce_time(h, m):
    if h >= 8 and h <= 17:
        result = True
    elif h == 18 and m == 0:
        result = True
    else:
        result = False

    return result

def calc_n_days_delta_date(date_str, n_days_list, str_dt_format="%d-%m-%Y"):
    dates_delta = []
    for n_day_delta in n_days_list:
        date_delta = str_datetime_n_days_before(date_str, n_day_delta, str_dt_format=str_dt_format)
        dates_delta.append(date_delta)
    return dates_delta
