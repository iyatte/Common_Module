from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVC


MODULE_FEATURE_COLUMN_PATH_KEY = 'Feature_Selection_Column_Path'
MODULE_FEATURE_RESULT_PATH_KEY = 'Feature_Selection_Result_Path'


#随机森林
def forest_feature(x, label_col_name, feat_labels):
    forest = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(x[feat_labels], x[label_col_name].tolist())
    #打印特征重要性评分
    importances = pd.DataFrame(forest.feature_importances_,columns = ['rd_score'])
    importances['rd_rank'] = importances.rank(ascending = False)
    importances['NAME'] = feat_labels
    return importances
#皮尔逊
def data_pro_pear(x,i):
    x_bak = []
    for f in range(len(x)):
        x_bak.append(x[f][i])
    return x_bak
def pear_feature(x1, label_col_name, feat_labels):
    y1 = x1[label_col_name].values
    x = x1[feat_labels].values
    pear_result = []
    P_result = []
    for i in range(x.shape[1]):
        x_bak = data_pro_pear(x,i)
        pear_result.append(abs(pearsonr(x_bak,y1)[0]))
        P_result.append(pearsonr(x_bak,y1)[1])
    pear_result_1 = pd.DataFrame(pear_result,columns = ['pearson_score'])
    P_result_1 = pd.DataFrame(P_result,columns = ['SCORE'])
    pear_result_1['pearson_rank'] = pear_result_1.rank(ascending = False)
    pear_result_1['NAME'] = feat_labels
    P_result_1['RANK'] = P_result_1.rank(ascending = True)
    P_result_1['NAME'] = feat_labels
    return pear_result_1
#卡方
def chi2_feature(x1, label_col_name, feat_labels):
    y1 = x1[label_col_name].values
    x = x1[feat_labels].values
    model1 = SelectKBest(chi2, k=2)#选择k个最佳特征
    model1.fit_transform(x, y1.astype('int'))
    chi2_result = pd.DataFrame(model1.scores_,columns = ['chi2_score'])
    chi2_result['chi2_rank'] = chi2_result.rank(ascending = False)
    chi2_result['NAME'] = feat_labels
    return chi2_result
#GBDT
def gbdt_feature(x, label_col_name, feat_labels):
    gbdt=GradientBoostingRegressor(
      loss='ls'
    , learning_rate=0.1
    , n_estimators=100
    , subsample=1
    , min_samples_split=2
    , min_samples_leaf=1
    , max_depth=3
    , init=None
    , random_state=None
    , max_features=None
    , alpha=0.9
    , verbose=0
    , max_leaf_nodes=None
    , warm_start=False
    )
    y1 = x[label_col_name].values
    x1 = x[feat_labels].values
    gbdt.fit(x1,y1)
    score = gbdt.feature_importances_
    gbdt_result = pd.DataFrame(score,columns = ['gbdt_score'])
    gbdt_result['gbdt_rank'] = gbdt_result.rank(ascending = False)
    gbdt_result['NAME'] = feat_labels
    return gbdt_result
#L1范式
def l1_feature(x, label_col_name, feat_labels):
    y1 = x[label_col_name].values
    x1 = x[feat_labels].values
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x1, y1.astype('int'))
    cofe_lsvc = lsvc.coef_.T.sum(axis = 1)
    l1_result = pd.DataFrame(abs(cofe_lsvc),columns = ['l1_score'])
    l1_result['l1_rank'] = l1_result.rank(ascending = False)
    l1_result['NAME'] = feat_labels
    return l1_result
#计算排名
def feature_importance(df_deal_all_minmax,label_col_name,feature_col_name,selector_type):
    all_result = pd.DataFrame(feature_col_name,columns=['NAME'])
    feature_selector_type = []
    if '皮尔逊相关系数' in selector_type:
        pear_feature_result = pear_feature(df_deal_all_minmax, label_col_name, feature_col_name)
        all_result = pd.merge(all_result,pear_feature_result,how='inner')
        del all_result['pearson_score']
        feature_selector_type.append('pearson_rank')
    if '卡方检验' in selector_type:
        chi2_feature_result = chi2_feature(df_deal_all_minmax, label_col_name, feature_col_name)
        all_result = pd.merge(all_result,chi2_feature_result,how='inner')
        del all_result['chi2_score']
        feature_selector_type.append('chi2_rank')
    if '随机森林' in selector_type:
        forest_result = forest_feature(df_deal_all_minmax, label_col_name, feature_col_name)
        all_result = pd.merge(all_result,forest_result,how='inner')
        del all_result['rd_score']
        feature_selector_type.append('rd_rank')
    if 'GBDT树' in selector_type:
        gbdt_result = gbdt_feature(df_deal_all_minmax, label_col_name, feature_col_name)
        all_result = pd.merge(all_result,gbdt_result,how='inner')
        del all_result['gbdt_score']
        feature_selector_type.append('gbdt_rank')
    if 'L1范式' in selector_type:
        l1_result = l1_feature(df_deal_all_minmax, label_col_name, feature_col_name)
        all_result = pd.merge(all_result,l1_result,how='inner')
        del all_result['l1_score']
        feature_selector_type.append('l1_rank')
    #行求和
    all_result['Col_sum'] = all_result[feature_selector_type].apply(lambda x: x.sum(), axis=1)
    #排序生成排名
    all_result.sort_values("Col_sum",inplace=True)
    all_result = all_result.reset_index(drop=True)
    del all_result['Col_sum']
    all_result.insert(1,'rank_all',all_result.index+1)
    new_name = all_result[all_result.rank_all<=10]['NAME'].values.tolist()
    for i in range(len(feature_col_name)):
        if feature_col_name[i] not in new_name:
            del df_deal_all_minmax[feature_col_name[i]]
    return df_deal_all_minmax,all_result




