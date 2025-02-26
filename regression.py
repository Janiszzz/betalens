#ToDo：
#实现石川的时间序列回归、截面回归和 Fama-Macbeth 回归
#时间序列回归适用于因子收益率已知的情况。有资产收益率和因子收益率，计算资产i的因子暴露beta，并求资产alpha
#截面回归可以处理因子收益率未知的情况。有资产收益率和因子值（往往是宏观因子），计算资产i的因子暴露beta，排序后获得因子收益率。
#Fama-Macbeth 回归是两步回归。对N个资产做N次时间序列回归，在T个时间点做T次截面回归（并取平均）。

#%%c-s reg
import pandas as pd
import numpy as np
import concurrent.futures

def create_sample_dataframes():
    # 创建一个包含时间戳的日期范围
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')

    # 假设我们有两个DataFrame，一个包含资产收益率，一个包含因子值
    asset_returns = pd.DataFrame({
        'date': dates,
        'asset_1': np.random.randn(1000),
        'asset_2': np.random.randn(1000),
        'asset_3': np.random.randn(1000)
    }).set_index('date')

    factor_values = pd.DataFrame({
        'date': dates,
        'factor_1': np.random.randn(1000),
        'factor_2': np.random.randn(1000),
        'factor_3': np.random.randn(1000)
    }).set_index('date')
    return asset_returns, factor_values

def gen_rolling_list(asset_returns, factor_values, window_length):
    asset_returns = asset_returns.join(factor_values)
    data_list = [[asset_returns.index[i + window_length - 1], asset_returns.iloc[i:i + window_length]] for i in
                 range(len(asset_returns) - window_length + 1)]
    return data_list

def c_s_regression(aligned_data, asset_names, factor_names, date):
    # 对每个资产进行回归分析并提取参数和t值
    import statsmodels.api as sm

    results = {}
    for asset in asset_names:
        asset_data = aligned_data[[asset] + factor_names]
        asset_data = sm.add_constant(asset_data, has_constant='add')
        model = sm.OLS(asset_data[asset], asset_data[['const'] + factor_names])
        fit = model.fit()
        results[asset] = fit.params

    # 计算资产收益率的时序平均
    mean_returns = aligned_data.mean()

    # 提取回归得到的参数
    params = pd.DataFrame({asset: results[asset] for asset in results}).T.drop('const', axis=1)
    params = sm.add_constant(params, has_constant='add')

    # 进行第二步回归
    second_step_model = sm.OLS(mean_returns[asset_names], params)
    second_step_results = second_step_model.fit()
    return pd.DataFrame(second_step_results.params, columns=[date]).T

#%%
asset_returns, factor_values = create_sample_dataframes()
window_length = 30
aligned_data_list = gen_rolling_list(asset_returns, factor_values, window_length)

asset_names = ['asset_1', 'asset_2', 'asset_3']
factor_names = ['factor_1', 'factor_2', 'factor_3']
#%%t-s

results = {}
for asset in asset_names:
    asset_data = aligned_data[[asset] + factor_names]
    asset_data = sm.add_constant(asset_data, has_constant='add')
    model = sm.OLS(asset_data[asset], asset_data[['const'] + factor_names])
    fit = model.fit()
    results[asset] = fit.params

#%%
futures = []
ans = pd.DataFrame()
with concurrent.futures.ThreadPoolExecutor() as executor:
    for date, aligned_data in aligned_data_list:
        # Pass process function and any additional
        # positional arguments and keyword arguments to executor.submit
        future = executor.submit(c_s_regression, aligned_data, asset_names, factor_names, date)
        futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        # do something with the result
        ans = pd.concat([ans, future.result()])

ans.sort_index(inplace=True)