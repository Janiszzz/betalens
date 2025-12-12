"""
Lucky Factors 检验模块 (Harvey and Liu, 2021)
对应旧版 robust.py，用于因子增量检验与 Alpha 显著性检验
"""
import pandas as pd
import numpy as np
import concurrent.futures
import statsmodels.api as sm


class RobustTest:
    def __init__(self, target: pd.Series, factors: pd.DataFrame):
        data = pd.concat([target, factors], axis=1).dropna()
        self.y = data.iloc[:, 0]
        self.X = data.iloc[:, 1:]
        self.target_name = target.name if target.name else 'target'
        self._OX = pd.DataFrame()
        self._T = pd.DataFrame()

    def _orthogonalize(self):
        self._T = {}
        self._OX = pd.DataFrame()
        for i in range(self.X.shape[1]):
            Xi = self.X.iloc[:, i]
            model = sm.OLS(Xi, sm.add_constant(self.y)).fit()
            self._OX = pd.concat([self._OX, model.resid], axis=1)
            model = sm.OLS(self.y, sm.add_constant(Xi)).fit()
            self._T[self.X.columns[i]] = model.tvalues.iloc[1]
        self._OX.columns = self.X.columns
        self._T = pd.DataFrame(self._T, index=[self.target_name]).T.abs()
        return self._OX, self._T

    def _bootstrap_resample(self, data):
        # 对应旧版 bootstrap_resample: 有放回抽样
        indices = np.sort(np.random.choice(range(data.shape[0]), data.shape[0], replace=True))
        return data.iloc[indices, :]

    def _max_statistic(self, data):
        # 对应旧版 max_statistic: 计算所有因子t统计量的最大值
        bs_y = data.iloc[:, 0]
        bs_OX = data.iloc[:, 1:]
        t_stats = []
        for i in range(bs_OX.shape[1]):
            model = sm.OLS(bs_y, sm.add_constant(bs_OX.iloc[:, i])).fit()
            t_stats.append(model.tvalues.iloc[1])
        return np.max(t_stats)

    def _panel_regression(self, X, y):
        # 对应旧版 panel: 多因子回归，返回系数/残差/t值
        model = sm.OLS(y, sm.add_constant(X)).fit()
        return model.params, model.resid, model.tvalues, model.params

    def _fake_fund(self, X, B, OX):
        # 对应旧版 fake_fund: 用系数重构+残差Bootstrap生成虚拟基金
        indices = np.random.choice(range(OX.shape[0]), OX.shape[0], replace=True)
        bootstrapped_OX = OX.iloc[indices].reset_index(drop=True)
        fake_y = X.mul(B.iloc[0, :], axis=1).drop('const', axis=1).sum(axis=1) + bootstrapped_OX
        return fake_y

    def _bootstrap_once(self, n_bootstraps=1000):
        # 对应旧版 bootstrap_once: 并行Bootstrap构建max-t分布，计算修正p值
        max_stat_pdf = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = pd.concat([self._y, self._OX], axis=1)
            for _ in range(n_bootstraps):
                future = executor.submit(self._max_statistic, self._bootstrap_resample(data))
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                max_stat_pdf.append(future.result())
        max_stat_pdf = np.sort(np.abs(np.array(max_stat_pdf)))
        modified_p = self._T.abs().apply(lambda x: np.searchsorted(max_stat_pdf, x) / len(max_stat_pdf))
        modified_p = 1 - modified_p
        eff_factors = modified_p.loc[modified_p[self._y.name] < 0.1].index.tolist()
        return eff_factors, modified_p

    def incremental_test(self, n_bootstraps=1000) -> pd.DataFrame:
        """
        因子增量检验，对应旧版 work() 迭代流程:
        1. neu(): 正交化因子，计算单因子t统计量
        2. bootstrap_once(): Bootstrap得到修正p值
        3. 识别显著因子(p<0.1)与非显著因子
        4. 若收敛则返回；否则用显著因子回归y取残差，剔除显著因子后继续
        """
        self._X = self.X.copy()
        self._y = self.y.copy()
        
        while True:
            if self._X.shape[0] < 100 or self._X.shape[1] == 0:
                return pd.DataFrame()
            
            # 正交化: 对应旧版 neu()
            self._T = {}
            self._OX = pd.DataFrame()
            for i in range(self._X.shape[1]):
                Xi = self._X.iloc[:, i]
                model = sm.OLS(Xi, sm.add_constant(self._y)).fit()
                self._OX = pd.concat([self._OX, model.resid], axis=1)
                model = sm.OLS(self._y, sm.add_constant(Xi)).fit()
                self._T[self._X.columns[i]] = model.tvalues.iloc[1]
            self._OX.columns = self._X.columns
            self._T = pd.DataFrame(self._T, index=[self._y.name]).T.abs()
            
            # Bootstrap检验
            eff_factors, modified_p = self._bootstrap_once(n_bootstraps)
            not_eff_factors = list(set(self._X.columns) - set(eff_factors))
            
            # 收敛条件: 全显著或全不显著
            if len(not_eff_factors) == 0 or len(eff_factors) == 0:
                return pd.DataFrame({
                    'factor': self._T.index,
                    't_stat': self._T[self._y.name].values,
                    'modified_p': modified_p[self._y.name].values,
                    'significant': modified_p[self._y.name].values < 0.1
                })
            
            # 用显著因子回归y取残差
            oy = self._y
            for fct in eff_factors:
                EXi = self._X.loc[:, fct]
                model = sm.OLS(oy, sm.add_constant(EXi)).fit()
                oy = model.resid
            
            # 剔除显著因子，继续迭代
            self._X = self._X[not_eff_factors]
            self._y = oy

    def alpha_test(self, n_bootstraps=1000) -> pd.DataFrame:
        """
        Alpha显著性检验，对应旧版 bootstrap_fake_fund 流程:
        1. panel(): 多因子回归得到系数B、残差OX、t值T
        2. fake_fund(): 用B重构+残差Bootstrap生成虚拟基金
        3. 对虚拟基金重复回归，构建alpha的t分布
        4. 计算真实alpha的修正p值
        """
        X = sm.add_constant(self.X)
        B, OX, T, _ = self._panel_regression(X, self.y)
        B = pd.DataFrame(B).T
        T = pd.DataFrame(T).T
        
        max_stat_pdf = []
        futures = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in range(n_bootstraps):
                future = executor.submit(self._panel_regression, X, self._fake_fund(X, B, OX))
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                max_stat_pdf.append(future.result()[2].iloc[0])

        max_stat_pdf = np.sort(np.abs(np.array(max_stat_pdf)))
        modified_p = T.abs().apply(lambda x: np.searchsorted(max_stat_pdf, x) / len(max_stat_pdf))
        modified_p = 1 - modified_p
        
        return pd.DataFrame({
            'alpha_t': T['const'].iloc[0],
            'alpha_p': modified_p['const'].iloc[0]
        }, index=[self.target_name])

    def rolling_test(self, interval='1Y', n_bootstraps=1000) -> pd.DataFrame:
        # 滚动时间段Alpha检验，对应旧版 gen_date_pairs + work 循环
        start_points = pd.date_range(start=self.y.index[0], end=self.y.index[-1], freq=interval)
        results = []
        
        for start in start_points:
            end = start + pd.Timedelta(interval)
            if end > self.y.index[-1]:
                end = self.y.index[-1]
            
            mask = (self.y.index >= start) & (self.y.index <= end)
            if mask.sum() < 30:
                continue
            
            sub_test = RobustTest(self.y[mask], self.X[mask])
            alpha_result = sub_test.alpha_test(n_bootstraps)
            
            results.append({
                'start': start,
                'end': end,
                'alpha_t': alpha_result['alpha_t'].iloc[0],
                'alpha_p': alpha_result['alpha_p'].iloc[0]
            })
        
        return pd.DataFrame(results)

    def segment_test(self, segments: list, n_bootstraps=1000) -> pd.DataFrame:
        # 指定时间段Alpha检验，对应旧版 parse_name_dates + get_interval + work
        results = []
        
        for seg in segments:
            start, end = seg['start'], seg['end']
            label = seg.get('label', f"{start}_{end}")
            
            if isinstance(start, str):
                start = pd.to_datetime(start)
            if isinstance(end, str):
                end = pd.to_datetime(end)
            
            mask = (self.y.index >= start) & (self.y.index <= end)
            if mask.sum() < 30:
                continue
            
            sub_test = RobustTest(self.y[mask], self.X[mask])
            alpha_result = sub_test.alpha_test(n_bootstraps)
            
            results.append({
                'segment': label,
                'start': start,
                'end': end,
                'alpha_t': alpha_result['alpha_t'].iloc[0],
                'alpha_p': alpha_result['alpha_p'].iloc[0]
            })
        
        return pd.DataFrame(results)

