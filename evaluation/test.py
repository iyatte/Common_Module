# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/18 21:24
@Author  : yany
@File    : test.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, precision_score, roc_auc_score, roc_curve


def test_model(y_test_, pre_y_, pre_y_pro_):
    # 混淆矩阵
    # 核心评估指标
    accuracy_s = accuracy_score(y_test_, pre_y_)  # 准确率
    precision_s = precision_score(y_test_, pre_y_)  # 精确度
    recall_s = recall_score(y_test_, pre_y_)  # 召回率
    auc_s = roc_auc_score(y_test_, pre_y_pro_)  # auc
    print('AUC:', auc_s, '准确率:', accuracy_s, '精确度:', precision_s, '召回率:', recall_s)
    return [accuracy_s, precision_s, recall_s, auc_s]


def ks_statistic(y_, y_pre):
    fpr, tpr, thresholds = roc_curve(y_, y_pre)
    ks = max(abs(tpr - fpr))
    return ks


def calc_woe(dataset, col, targe):
    # bad 1
    # good 0
    subdata = pd.DataFrame(dataset.groupby(col)[col].count())
    suby = pd.DataFrame(dataset.groupby(col)[targe].sum())
    data = pd.DataFrame(pd.merge(subdata, suby, how="left", left_index=True, right_index=True))
    b_total = data[targe].sum()
    total = data[col].sum()
    g_total = total - b_total
    data["bad"] = data.apply(lambda x: round(x[targe] / b_total, 3), axis=1)
    data["good"] = data.apply(lambda x: round((x[col] - x[targe]) / g_total, 3), axis=1)
    data["WOE"] = data.apply(lambda x: np.log(x.bad / x.good), axis=1)
    return data.loc[:, ["bad", "good", "WOE"]]


def calc_iv(dataset):
    dataset["IV"] = dataset.apply(lambda x: (x.bad - x.good) * x.WOE, axis=1)
    IV = sum(dataset["IV"])
    return IV