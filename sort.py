#Todo：
#1.多重排序法计算多空组合√
#2.类交易模块，计算因子组合的收益率
import pandas as pd
import numpy as np
import init
#%%

def single_sort(df,keys,quantile_dict):
    if len(keys) != len(quantile_dict):
        raise ValueError("keys 和 quantile_dict 的长度必须相等")
    for key in keys:
        df[key + '_label'] = pd.qcut(df[key], quantile_dict[key], labels=False, duplicates='drop')
    return df

def single_sort_result(factor, factor_key):
    def f(group):
        group[factor_key + '_weight'] = 0
        group.loc[group[factor_key + '_label'] == group.loc[:, factor_key + '_label'].max(), factor_key + '_weight'] = 1
        group.loc[group[factor_key + '_label'] == group.loc[:, factor_key + '_label'].min(), factor_key + '_weight'] = -1
        return group
    factor = factor.groupby(factor.index.get_level_values(0),as_index=False,).apply(f)
    return factor.filter(like='weight')

def multi_sort(df,keys,quantile_dict):
    key = keys[0]
    df[key + '_label'] = pd.qcut(df[key], quantile_dict[key], labels=False, duplicates='drop')
    if len(keys) == 1:
        return df
    else:
        keys = keys[1:]
        return df.groupby(key + '_label',as_index=False,group_keys=False,).apply(lambda group: multi_sort(group,keys,quantile_dict),include_groups=True)

def multi_sort_result(factor, factor_keys):
    factor['type'] = factor[[x + "_label" for x in factor_keys]].apply(lambda row: ', '.join(row.values.astype(str)),axis=1)
    ptf_names = factor['type'].reset_index(drop=True).drop_duplicates().reset_index(drop=True)
    collect = {}
    for ptf in ptf_names:
        factor['weight'] = 0
        factor.loc[factor['type'] == ptf, 'weight'] = 1
        collect.update({ptf: factor})
    return collect

#%%










#%%
df = data.loc[:,['close_timestamp','windcode','open','high','low','close','volume']]
df.sort_values('close_timestamp',ascending=True,inplace=True)
df = df.iloc[:1000]
#%%
keys = ['volume','high']
quantile_dict = {
    'volume':3,
    'high':2,
}
def f(group,keys,quantile_dict):
    return multi_sort(group,keys,quantile_dict)
df = df.groupby('close_timestamp',as_index=False,).apply(f,keys,quantile_dict)
df.set_index(['close_timestamp','windcode'],inplace=True)
#%%临时的
def calc_ret(df, idl, price):
    #idl is PERMNO or windcode etc., price to calc return rate
    def f(group):
        group['ret'] = np.log(group[price]/group[price].shift(1))
        return group
    df = df.apply(lambda x: x.sort_index(ascending=True))
    df = df.groupby(df.index.get_level_values(1),as_index=False,group_keys=False).apply(f)
    return df
ret = pd.DataFrame(df['close'])

ret = calc_ret(ret, 'windcode','close')
ret = ret.reset_index().pivot_table(index='close_timestamp', columns='windcode', values='ret')
#%%
factor = df
factor_key = 'volume'
def f(group):
    group[factor_key + '_weight'] = 0
    group.loc[group[factor_key + '_label'] == group.loc[:, factor_key + '_label'].max(),factor_key + '_weight'] = 1
    group.loc[group[factor_key + '_label'] == group.loc[:, factor_key + '_label'].min(), factor_key + '_weight'] = -1
    return group

factor = factor.groupby(factor.index.get_level_values(0),as_index=False,).apply(f)
factor.filter(like='weight')
#%%
factor = df
factor_keys = ['high','volume']
factor['type'] = factor[[x + "_label" for x in factor_keys]].apply(lambda row: ', '.join(row.values.astype(str)),axis=1)
ptf_names = factor['type'].reset_index(drop=True).drop_duplicates().reset_index(drop=True)
collect = {}
for ptf in ptf_names:
    factor['weight'] = 0
    factor.loc[factor['type'] == ptf, 'weight'] = 1
    collect.update({ptf: factor['weight']})