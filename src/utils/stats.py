import pandas as pd
import numpy as np
from sklearn import metrics
import scipy.stats


from src.utils import pretties


INVALID_COLUMN_ALL_VALUES_NULL = "Todos os valores são NaN."
INVALID_COLUMN_ALL_VALUES_EQUAL = "Todos os valores são iguais."
VALID_COLUMN_OBS_ALL_NULL = "Valores não são iguais. Mas todos são nulos e/ou zeros."


def confusion_matrix(y_true, y_pred):
    cm_array = metrics.confusion_matrix(y_true=y_true,
                                        y_pred=y_pred)

    cm = pd.DataFrame(cm_array, index=["fut_rech REAL", "solv_ech REAL"]).rename(columns={0: "PRED\nfut_rech", 1: "PRED\nsolv_ech"})

    y_true_freq = pd.Series(y_true).value_counts()

    if y_true_freq.loc["fut_rech"] != cm.loc["fut_rech REAL"].sum():
        raise Exception("Provavelmente as colunas estão trocadas")

    pink = "#ffbaba"

    return cm

def confusion_matrix2(y_true, y_pred):
    cm_array = metrics.confusion_matrix(y_true=y_true,
                                        y_pred=y_pred)

    cm = pd.DataFrame(cm_array, index=["False REAL", "True REAL"]).rename(columns={0: "PRED False", 1: "PRED True"})

    y_true_freq = pd.Series(y_true).value_counts()

    if y_true_freq.loc[True] != cm.loc["True REAL"].sum():
        raise Exception("Provavelmente as colunas estão trocadas")

    return cm[["PRED True", "PRED False"]].loc[["True REAL", "False REAL"]]

def accuracy(y_pred, y_true):
    return metrics.accuracy_score(y_pred, y_true)

def freq(pd_series):
    '''
        Returns a pandas.DataFrame with absolute and relative frequencies
    '''
    proportional = pd_series.astype(str).value_counts(normalize=True).rename("freq_relative")
    proportional = proportional * 100
    proportional = proportional.apply(lambda value : "{0:.2f}".format(round(value, 2)) + "%")

    absolute = pd_series.astype(str).value_counts().rename("freq_absolute")

    return pd.concat([absolute, proportional], axis=1)

def grouped_freq(df, groupby_col, freq_col):

    freqs = None

    for value in df[groupby_col].drop_duplicates():
        group = df[df[groupby_col] == value]
        freq = group[freq_col].value_counts(normalize=True).to_frame().rename({freq_col: value}, axis=1)

        if freqs is None:
            freqs = freq
        else:
            freqs = freqs.join(freq, how="outer")

    freqs.index.name = freq_col
    freqs = freqs.sort_values(value, ascending=False)
    return freqs

def nullity_freq(pd_series):
    '''
    Return frequency of null and not null values.
    :param pd_series:
    :return:
    '''
    abs_freq = freq(series_nullity(pd_series))
    abs_freq.rename(index={True: "Nulos", False: "Não nulos"}, inplace=True)
    abs_freq.rename(index={"True": "Nulos", "False": "Não nulos"}, inplace=True)
    return abs_freq

def clean_describe(series):
    return series.describe()[["min", "25%", "50%", "75%", "max"]]

def series_nullity(series):
    '''
    Returns True for each null value. False for not null.
    Null value are: np.NaN, None and 0.
    :param series:
    :return:
    '''
    return series.isnull() | series.isna()

def relative_frequencies(series):
    unity_value_percent = 100.0 / series.sum()
    return unity_value_percent * series

def under_percentile_value(series, percentile=0.95):
    '''
    Retorna um pandas.Series removendo os valores que estejam abaixo do percentil informado.
    Atente para o fato de que distribuições com pouca variância de valores podem retornar uma lista vazia.
    :param series:
    :param percentile:
    :return:
    '''
    percentile_value = series.quantile(percentile)
    return series[series < percentile_value]

def abs_diff(a, b):
    return abs(a - b)


def correlate_categ_freq(feature, target, balance_levels=None):
    '''
    Computa a correlação entre variáveis categóricas através das frequências dos
    níveis do target, agrupadas pelos níveis de feature.
    Features pode ser multinível enquanto que target deve ser binária.

    O teste estatístico de Wilcoxon é aplicado às diferenças das frequências computadas para cada nível target.
    Ver https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

    :param feature:
    :param target:
    :return:
    '''



    if len(set(target)) != 2:
        raise Exception("Total target levels must be 2")

    rel_freq = target_freq_by_categ(feature=feature, target=target, plot=False)

    return scipy.stats.wilcoxon(x=rel_freq[rel_freq.columns[0]],
                                y=rel_freq[rel_freq.columns[1]])

def dispersion_categ_freq(feature, target, plot=True):
    '''
    Computa a variância das frequências dos níveis do target, agrupadas pelos níveis de feature.
    Features pode ser multinível enquanto que target deve ser binária.

    Importante níveis pouco frequentes tem o mesmo impacto dos níveis mais frequentes.

    :param feature:
    :param target:
    :param plot:
    :return:
    '''
    if len(set(target)) != 2:
        raise Exception("Total target levels must be 2")

    rel_freq = target_freq_by_categ(feature=feature, target=target, plot=plot)
    return rel_freq[rel_freq.columns[0]].mad() * 2


def build_percent_milestones(data):
    '''
    Retorna o percentual de completude de um valor, dado um valor absoluto.
    Este dicionário serve para processamentos em loop for par acompnhar o andamento da completude.

    :param data:
    :return:
    '''
    return {int(len(data) * 1 / 1000): '  0.1% done...',
            int(len(data) * 1 / 500):  '  0.2% done...',
            int(len(data) * 1 / 100):  '  1.0% done...',
            int(len(data) * 1 / 10) :  ' 10.0% done...',
            int(len(data) * 2 / 10) :  ' 20.0% done...',
            int(len(data) * 3 / 10) :  ' 30.0% done...',
            int(len(data) * 4 / 10) :  ' 40.0% done...',
            int(len(data) * 5 / 10) :  ' 50.0% done...',
            int(len(data) * 6 / 10) :  ' 60.0% done...',
            int(len(data) * 7 / 10) :  ' 70.0% done...',
            int(len(data) * 8 / 10) :  ' 80.0% done...',
            int(len(data) * 9 / 10) :  ' 90.0% done...',
            int(len(data))          :  '100.0% done!  '}

def is_useful_column(series):
    unique_values_len = len(series.astype(str).drop_duplicates())
    result = {"useful": True, "reason": "{} diffferent values".format(unique_values_len)}

    if len(series.value_counts()) == 0:
        result["reason"] = INVALID_COLUMN_ALL_VALUES_NULL
        result["useful"] = False

    elif unique_values_len == 1:
        result["reason"] = INVALID_COLUMN_ALL_VALUES_EQUAL
        result["useful"] = False

    elif unique_values_len > 1 and len(series_nullity(series.replace({0: np.NaN})).drop_duplicates()) == 1:
        result["reason"] = VALID_COLUMN_OBS_ALL_NULL
        result["useful"] = True

    return result


def useful_columns(ech_tb, basic_columns=[], report=False):

    for colname in ech_tb.columns.tolist():
        if colname in basic_columns:
            continue

        if is_useful_column(ech_tb[colname])["useful"]:
            basic_columns.append(colname)

    if 'ASAIUUI' in basic_columns:
        basic_columns.remove('ASAIUUI')
        basic_columns.append('ASAIUUI')

    if report:
        pretties.display_md("Original columns n: {}".format(len(ech_tb.columns)))
        pretties.display_md("Useful columns n: {}".format(len(basic_columns)))

    return basic_columns


def target_freq(group, target_colname="target"):
    '''
    Retorna a frequencia da=os níveis de target do grupo
    :param series:
    :return:
    '''
    f = (group[target_colname].value_counts() / len(group)).to_dict()
    f["index"] = group.name
    return f

def abs_diff(a, b):
    return abs(a - b)

def resampling(series, n=None):
    if n is None:
        n = series.value_counts().min()
        
    print("n:", n)
    
    if len(series) < n:
        resample = pd.Series()
        for i in range(n):
            resample = resample.append(series.sample())
    else:
        resample = series.sample(n)
    
    return resample

def categ_col_comparison(col1, col2):
    dividend = len((col1.astype(str) + "---" + col2.astype(str)).drop_duplicates())
    divisor = len(col1.drop_duplicates()) + len(col2.drop_duplicates())
    return 2 * dividend / divisor

def roc_curve_data(y_true, y_score, on_label="fut_rech", round_values=2, as_frame=True):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true,
                                             y_score=y_score,
                                             pos_label=on_label)

    fpr = [round(value, round_values) for value in fpr]
    tpr = [round(value, round_values) for value in tpr]

    if as_frame:
        return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds, "label": on_label})

    return {"fpr": list(fpr), "tpr": list(tpr), "threshold": list(thresholds), "label": on_label}

def score_curve_data(y_true, y_proba, on_label="fut_rech", round_values=2, as_frame=True):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true=y_true,
                                                                   probas_pred=y_proba,
                                                                   pos_label=on_label)

    precision = [round(value, round_values) for value in precision]
    recall = [round(value, round_values) for value in recall]
    thresholds = [round(value, round_values) for value in thresholds]

    if as_frame:
        return pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds, "label": on_label})

    return {"precision": list(precision[:-1]), "recall": list(recall[:-1]), "threshold": list(thresholds), "label": on_label}


def mean_confidence_interval(data, confidence=0.75):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
