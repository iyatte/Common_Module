# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/18 21:26
@Author  : yany
@File    : ChiMerge.py
"""
import pandas as pd
import numpy as np


# ##函数########
# 计算变量分箱之后各分箱的坏样本率
def BinBadRate(df, col, target, grantRateIndicator=0):
    """
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    """
    # print(df.groupby([col])[target])
    total = df.groupby([col])[target].count()
    # print(total)
    total = pd.DataFrame({'total': total})
    # print(total)
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    # 合并
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    # print(regroup)
    regroup.reset_index(level=0, inplace=True)
    # print(regroup)
    # 计算坏样本率
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    # print(regroup)
    # 生成字典，（变量名取值：坏样本率）
    dicts = dict(zip(regroup[col], regroup['bad_rate']))
    if grantRateIndicator == 0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    # 总体样本率
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)


# 计算WOE值
def CalcWOE(df, col, target):
    """
    :param df: 包含需要计算WOE的变量和目标变量
    :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
    :param target: 目标变量，0、1表示好、坏
    :return: 返回WOE和IV
    """
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV': IV}


# # 判断某变量的坏样本率是否单调
def BadRateMonotone(df, sortByVar, target, special_attribute=[]):
    """
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    """
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1] * 1.0 / x[0] for x in combined]
    badRateNotMonotone = [
        badRate[i] < badRate[i + 1] and badRate[i] < badRate[i - 1] or badRate[i] > badRate[i + 1] and badRate[i] >
        badRate[i - 1]
        for i in range(1, len(badRate) - 1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


# ##函数########
# 计算变量分箱之后各分箱的坏样本率
def BinBadRate(df, col, target, grantRateIndicator=0):
    """
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    """
    # print(df.groupby([col])[target])
    total = df.groupby([col])[target].count()
    # print(total)
    total = pd.DataFrame({'total': total})
    # print(total)
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    # 合并
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    # print(regroup)
    regroup.reset_index(level=0, inplace=True)
    # print(regroup)
    # 计算坏样本率
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    # print(regroup)
    # 生成字典，（变量名取值：坏样本率）
    dicts = dict(zip(regroup[col], regroup['bad_rate']))
    if grantRateIndicator == 0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    # 总体样本率
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)


# # 判断某变量的坏样本率是否单调
def BadRateMonotone(df, sortByVar, target, special_attribute=[]):
    """
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    """
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1] * 1.0 / x[0] for x in combined]
    badRateNotMonotone = [
        badRate[i] < badRate[i + 1] and badRate[i] < badRate[i - 1] or badRate[i] > badRate[i + 1] and badRate[i] >
        badRate[i - 1]
        for i in range(1, len(badRate) - 1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


def AssignBin(x, cutOffPoints, special_attribute=[]):
    """
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    """
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x) + 1
        return 'Bin {}'.format(0 - i)
    if x <= cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin - 1)
    else:
        for i in range(0, numBin - 1):
            if cutOffPoints[i] < x <= cutOffPoints[i + 1]:
                return 'Bin {}'.format(i + 1)


def AssignGroup(x, bin):
    """
    :param x: 某个变量的某个取值
    :param bin: 上述变量的分箱结果
    :return: x在分箱结果下的映射
    """
    N = len(bin)
    if x <= min(bin):
        return min(bin)
    elif x > max(bin):
        return 10e10
    else:
        for i in range(N - 1):
            if bin[i] < x <= bin[i + 1]:
                return bin[i + 1]


def SplitData(df, col, numOfSplit, special_attribute=[]):
    """
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外
    :return: 在原数据集上增加一列，把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理
    """
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]  # 行数
    # " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法
    n = N // numOfSplit  # 每组样本数
    splitPointIndex = [i * n for i in range(1, numOfSplit)]  # 分割点的下标
    """
    [i*2 for i in range(1,100)]
    [2, 4, 6, 8, 10,......,198]
    """
    rawValues = sorted(list(df2[col]))  # 对取值进行排序
    # 取到粗糙卡方划分节点
    splitPoint = [rawValues[i] for i in splitPointIndex]  # 分割点的取值
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint


# 计算卡方值的函数
def Chi2(df, total_col, bad_col, overallRate):
    """
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param total_col: 全部样本的个数
    :param bad_col: 坏样本的个数
    :param overallRate: 全体样本的坏样本占比
    :return: 卡方值
    """
    df2 = df.copy()
    # 期望坏样本个数＝全部样本个数*平均坏样本占比
    df2['expected'] = df[total_col].apply(lambda x: x * overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0] - i[1]) ** 2 / i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


# #ChiMerge_MaxInterval：通过指定最大间隔数，使用卡方值分割连续变量
def ChiMerge(df, col, target, max_interval=5, special_attribute=[], minBinPcnt=0):
    """
    :param df: 包含目标变量与分箱属性的数据框
    :param col: 需要分箱的属性
    :param target: 目标变量，取值0或1
    :param max_interval: 最大分箱数。如果原始属性的取值个数低于该参数，不执行这段函数
    :param special_attribute: 不参与分箱的属性取值，缺失值的情况
    :param minBinPcnt：最小箱的占比，默认为0
    :return: 分箱结果
    """
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)  # 不同的取值个数
    if N_distinct <= max_interval:  # 如果原始属性的取值个数低于max_interval，不执行这段函数
        print("原始属性{}的取值个数低于max_interval".format(col))
        # 分箱分数间隔段，少一个值也可以
        # 返回值colLevels会少一个最大值
        return colLevels[:-1]
    else:
        if len(special_attribute) >= 1:
            # df1数据框取trainData中col那一列为特殊值的数据集
            # df1 = df.loc[df[col].isin(special_attribute)]
            print('{} 有缺失值的情况'.format(col))
            # 用逆函数对筛选后的结果取余，起删除指定行作用
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))  # 该特征不同的取值

        # 步骤一: 通过col对数据集进行分组，求出每组的总样本数与坏样本数
        if N_distinct > 100:
            """
            split_x样例
            [2, 8, 9.3 , 1 0, 30 ,......,1800]
            """
            split_x = SplitData(df2, col, 100)
            # 把值变为划分点的值
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        else:
            # 假如数值取值小于100就不发生变化了
            df2['temp'] = df2[col]
        # 总体bad rate将被用来计算expected bad count
        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)

        # 首先，每个单独的属性值将被分为单独的一组
        # 对属性值进行排序，然后两两组别进行合并
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]

        # 步骤二：建立循环，不断合并最优的相邻两个组别，直到：
        # 1，最终分裂出来的分箱数<＝预设的最大分箱数
        # 2，每箱的占比不低于预设值（可选）
        # 3，每箱同时包含好坏样本
        # 如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
        split_intervals = max_interval - len(special_attribute)
        while (len(groupIntervals) > split_intervals):  # 终止条件: 当前分箱数＝预设的分箱数
            # 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案
            # 存储卡方值
            chisqList = []
            for k in range(len(groupIntervals) - 1):
                temp_group = groupIntervals[k] + groupIntervals[k + 1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = Chi2(df2b, 'total', 'bad', overallRate)
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined + 1]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[best_comnbined + 1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        # 检查是否有箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
        groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
        # 已成完成卡方分箱，但是没有考虑其单调性
        df2['temp_Bin'] = groupedvalues
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
        while minBadRate == 0 or maxBadRate == 1:
            # 找出全部为好／坏样本的箱
            indexForBad01 = regroup[regroup['bad_rate'].isin([0, 1])].temp_Bin.tolist()
            bin = indexForBad01[0]
            # 如果是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
            if bin == max(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[:-1]
            # 如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
            elif bin == min(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[1:]
            # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
            else:
                # 和前一箱进行合并，并且计算卡方值
                currentIndex = list(regroup.temp_Bin).index(bin)
                prevIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
                # 和后一箱进行合并，并且计算卡方值
                laterIndex = list(regroup.temp_Bin)[currentIndex + 1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])
            # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
            [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]

        # 需要检查分箱后的最小占比
        if minBinPcnt > 0:
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            # value_counts每个数值出现了多少次
            valueCounts = groupedvalues.value_counts().to_frame()
            N = sum(valueCounts['temp'])
            valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / N)
            valueCounts = valueCounts.sort_index()
            minPcnt = min(valueCounts['pcnt'])
            # 一定要箱数大于2才可以，要不就不能再合并了
            while minPcnt < minBinPcnt and len(cutOffPoints) > 2:
                # 找出占比最小的箱
                indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0]
                # 如果占比最小的箱是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
                if indexForMinPcnt == max(valueCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                # 如果占比最小的箱是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
                elif indexForMinPcnt == min(valueCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                # 如果占比最小的箱是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
                else:
                    # 和前一箱进行合并，并且计算卡方值
                    currentIndex = list(valueCounts.index).index(indexForMinPcnt)
                    prevIndex = list(valueCounts.index)[currentIndex - 1]
                    df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                    chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
                    # 和后一箱进行合并，并且计算卡方值
                    laterIndex = list(valueCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                    chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])
        cutOffPoints = special_attribute + cutOffPoints
        return cutOffPoints

if __name__ == "__main__":
    # demo
    numericalFeatures = []
    trainData = pd.DataFrame()
    WOE_IV_dict = dict()
    bin_dict = []
    for var in numericalFeatures:
        binNum = 5
        newBin = var + '_Bin'
        bin = ChiMerge(trainData, var, 'label', max_interval=binNum, minBinPcnt=0.05)
        trainData[newBin] = trainData[var].apply(lambda x: AssignBin(x, bin))
        # 如果不满足单调性，就降低分箱个数
        while not BadRateMonotone(trainData, newBin, 'label'):
            binNum -= 1
            bin = ChiMerge(trainData, var, 'label', max_interval=binNum, minBinPcnt=0.05)
            trainData[newBin] = trainData[var].apply(lambda x: AssignBin(x, bin))
        WOE_IV_dict[newBin] = CalcWOE(trainData, newBin, 'label')
        bin_dict.append({var: bin})