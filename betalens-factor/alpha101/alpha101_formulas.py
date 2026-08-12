"""Canonical WorldQuant Alpha101 formulas on daily wide DataFrames."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from factor_template_alpha101 import (
    clean_inf,
    correlation,
    covariance,
    decay_linear,
    delay,
    delta,
    elementwise_max,
    elementwise_min,
    indneutralize,
    product,
    rank,
    scale,
    sign,
    signed_power,
    stddev,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_sum,
    where,
)


MARKET_INPUTS = {
    "open": ("open_wide", "开盘价(元)"),
    "close": ("close_wide", "收盘价(元)"),
    "high": ("high_wide", "最高价(元)"),
    "low": ("low_wide", "最低价(元)"),
    "volume": ("volume_wide", "成交量(股)"),
    "vwap": ("vwap_wide", "均价(元)"),
    "returns": ("returns_wide", "涨跌幅(%)"),
    "cap": ("cap_wide", "总市值(元)"),
    "amount": ("amount_wide", "成交金额(元)"),
}
INDUSTRY_INPUTS = {
    "sector": ("sector_wide", "申万一级行业"),
    "industry": ("industry_wide", "申万二级行业"),
    "subindustry": ("subindustry_wide", "申万三级行业"),
}
INPUT_ORDER = tuple(MARKET_INPUTS) + tuple(INDUSTRY_INPUTS)


FORMULAS = {
    1: "rank(ts_argmax(signedpower((returns < 0 ? stddev(returns,20) : close),2),5))-0.5",
    2: "-1*correlation(rank(delta(log(volume),2)),rank((close-open)/open),6)",
    3: "-1*correlation(rank(open),rank(volume),10)",
    4: "-1*ts_rank(rank(low),9)",
    5: "rank(open-sum(vwap,10)/10)*(-1*abs(rank(close-vwap)))",
    6: "-1*correlation(open,volume,10)",
    7: "adv20<volume ? -1*ts_rank(abs(delta(close,7)),60)*sign(delta(close,7)) : -1",
    8: "-1*rank(sum(open,5)*sum(returns,5)-delay(sum(open,5)*sum(returns,5),10))",
    9: "0<ts_min(delta(close,1),5) ? delta(close,1) : (ts_max(delta(close,1),5)<0 ? delta(close,1) : -delta(close,1))",
    10: "rank(0<ts_min(delta(close,1),4) ? delta(close,1) : (ts_max(delta(close,1),4)<0 ? delta(close,1) : -delta(close,1)))",
    11: "(rank(ts_max(vwap-close,3))+rank(ts_min(vwap-close,3)))*rank(delta(volume,3))",
    12: "sign(delta(volume,1))*(-1*delta(close,1))",
    13: "-1*rank(covariance(rank(close),rank(volume),5))",
    14: "-1*rank(delta(returns,3))*correlation(open,volume,10)",
    15: "-1*sum(rank(correlation(rank(high),rank(volume),3)),3)",
    16: "-1*rank(covariance(rank(high),rank(volume),5))",
    17: "-1*rank(ts_rank(close,10))*rank(delta(delta(close,1),1))*rank(ts_rank(volume/adv20,5))",
    18: "-1*rank(stddev(abs(close-open),5)+(close-open)+correlation(close,open,10))",
    19: "-1*sign((close-delay(close,7))+delta(close,7))*(1+rank(1+sum(returns,250)))",
    20: "-1*rank(open-delay(high,1))*rank(open-delay(close,1))*rank(open-delay(low,1))",
    21: "sum(close,8)/8+stddev(close,8)<sum(close,2)/2 ? -1 : (sum(close,2)/2<sum(close,8)/8-stddev(close,8) ? 1 : (1<=volume/adv20 ? 1 : -1))",
    22: "-1*delta(correlation(high,volume,5),5)*rank(stddev(close,20))",
    23: "sum(high,20)/20<high ? -1*delta(high,2) : 0",
    24: "delta(sum(close,100)/100,100)/delay(close,100)<=0.05 ? -1*(close-ts_min(close,100)) : -1*delta(close,3)",
    25: "rank(-1*returns*adv20*vwap*(high-close))",
    26: "-1*ts_max(correlation(ts_rank(volume,5),ts_rank(high,5),5),3)",
    27: "0.5<rank(sum(correlation(rank(volume),rank(vwap),6),2)/2) ? -1 : 1",
    28: "scale(correlation(adv20,low,5)+(high+low)/2-close)",
    29: "min(product(rank(rank(scale(log(sum(ts_min(rank(rank(-1*rank(delta(close-1,5)))),2),1))))),1),5)+ts_rank(delay(-returns,6),5)",
    30: "(1-rank(sign(close-delay(close,1))+sign(delay(close,1)-delay(close,2))+sign(delay(close,2)-delay(close,3))))*sum(volume,5)/sum(volume,20)",
    31: "rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10))))+rank(-1*delta(close,3))+sign(scale(correlation(adv20,low,12)))",
    32: "scale(sum(close,7)/7-close)+20*scale(correlation(vwap,delay(close,5),230))",
    33: "rank(-1*(1-open/close)^1)",
    34: "rank((1-rank(stddev(returns,2)/stddev(returns,5)))+(1-rank(delta(close,1))))",
    35: "ts_rank(volume,32)*(1-ts_rank(close+high-low,16))*(1-ts_rank(returns,32))",
    36: "2.21*rank(correlation(close-open,delay(volume,1),15))+0.7*rank(open-close)+0.73*rank(ts_rank(delay(-returns,6),5))+rank(abs(correlation(vwap,adv20,6)))+0.6*rank((sum(close,200)/200-open)*(close-open))",
    37: "rank(correlation(delay(open-close,1),close,200))+rank(open-close)",
    38: "-1*rank(ts_rank(close,10))*rank(close/open)",
    39: "-1*rank(delta(close,7)*(1-rank(decay_linear(volume/adv20,9))))*(1+rank(sum(returns,250)))",
    40: "-1*rank(stddev(high,10))*correlation(high,volume,10)",
    41: "(high*low)^0.5-vwap",
    42: "rank(vwap-close)/rank(vwap+close)",
    43: "ts_rank(volume/adv20,20)*ts_rank(-1*delta(close,7),8)",
    44: "-1*correlation(high,rank(volume),5)",
    45: "-1*rank(sum(delay(close,5),20)/20)*correlation(close,volume,2)*rank(correlation(sum(close,5),sum(close,20),2))",
    46: "0.25<((delay(close,20)-delay(close,10))/10-(delay(close,10)-close)/10) ? -1 : (((delay(close,20)-delay(close,10))/10-(delay(close,10)-close)/10)<0 ? 1 : -1*(close-delay(close,1)))",
    47: "rank(1/close)*volume/adv20*high*rank(high-close)/(sum(high,5)/5)-rank(vwap-delay(vwap,5))",
    48: "indneutralize(correlation(delta(close,1),delta(delay(close,1),1),250)*delta(close,1)/close,subindustry)/sum((delta(close,1)/delay(close,1))^2,250)",
    49: "((delay(close,20)-delay(close,10))/10-(delay(close,10)-close)/10)<-0.1 ? 1 : -1*(close-delay(close,1))",
    50: "-1*ts_max(rank(correlation(rank(volume),rank(vwap),5)),5)",
    51: "((delay(close,20)-delay(close,10))/10-(delay(close,10)-close)/10)<-0.05 ? 1 : -1*(close-delay(close,1))",
    52: "(-1*ts_min(low,5)+delay(ts_min(low,5),5))*rank((sum(returns,240)-sum(returns,20))/220)*ts_rank(volume,5)",
    53: "-1*delta(((close-low)-(high-close))/(close-low),9)",
    54: "-1*(low-close)*(open^5)/((low-high)*(close^5))",
    55: "-1*correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))),rank(volume),6)",
    56: "-1*rank(sum(returns,10)/sum(sum(returns,2),3))*rank(returns*cap)",
    57: "-1*(close-vwap)/decay_linear(rank(ts_argmax(close,30)),2)",
    58: "-1*ts_rank(decay_linear(correlation(indneutralize(vwap,sector),volume,3.92795),7.89291),5.50322)",
    59: "-1*ts_rank(decay_linear(correlation(indneutralize(vwap*0.728317+vwap*(1-0.728317),industry),volume,4.25197),16.2289),8.19648)",
    60: "-1*(2*scale(rank((((close-low)-(high-close))/(high-low))*volume))-scale(rank(ts_argmax(close,10))))",
    61: "rank(vwap-ts_min(vwap,16.1219))<rank(correlation(vwap,adv180,17.9282))",
    62: "-1*(rank(correlation(vwap,sum(adv20,22.4101),9.91009))<rank((rank(open)+rank(open))<(rank((high+low)/2)+rank(high))))",
    63: "-1*(rank(decay_linear(delta(indneutralize(close,industry),2.25164),8.22237))-rank(decay_linear(correlation(vwap*0.318108+open*(1-0.318108),sum(adv180,37.2467),13.557),12.2883)))",
    64: "-1*(rank(correlation(sum(open*0.178404+low*(1-0.178404),12.7054),sum(adv120,12.7054),16.6208))<rank(delta(((high+low)/2)*0.178404+vwap*(1-0.178404),3.69741)))",
    65: "-1*(rank(correlation(open*0.00817205+vwap*(1-0.00817205),sum(adv60,8.6911),6.40374))<rank(open-ts_min(open,13.635)))",
    66: "-1*(rank(decay_linear(delta(vwap,3.51013),7.23052))+ts_rank(decay_linear((low-vwap)/(open-(high+low)/2),11.4157),6.72611))",
    67: "-1*(rank(high-ts_min(high,2.14593))^rank(correlation(indneutralize(vwap,sector),indneutralize(adv20,subindustry),6.02936)))",
    68: "-1*(ts_rank(correlation(rank(high),rank(adv15),8.91644),13.9333)<rank(delta(close*0.518371+low*(1-0.518371),1.06157)))",
    69: "-1*(rank(ts_max(delta(indneutralize(vwap,industry),2.72412),4.79344))^ts_rank(correlation(close*0.490655+vwap*(1-0.490655),adv20,4.92416),9.0615))",
    70: "-1*(rank(delta(vwap,1.29456))^ts_rank(correlation(indneutralize(close,industry),adv50,17.8256),17.9171))",
    71: "max(ts_rank(decay_linear(correlation(ts_rank(close,3.43976),ts_rank(adv180,12.0647),18.0175),4.20501),15.6948),ts_rank(decay_linear(rank(low+open-vwap-vwap)^2,16.4662),4.4388))",
    72: "rank(decay_linear(correlation((high+low)/2,adv40,8.93345),10.1519))/rank(decay_linear(correlation(ts_rank(vwap,3.72469),ts_rank(volume,18.5188),6.86671),2.95011))",
    73: "-1*max(rank(decay_linear(delta(vwap,4.72775),2.91864)),ts_rank(decay_linear(-1*delta(open*0.147155+low*(1-0.147155),2.03608)/(open*0.147155+low*(1-0.147155)),3.33829),16.7411))",
    74: "-1*(rank(correlation(close,sum(adv30,37.4843),15.1365))<rank(correlation(rank(high*0.0261661+vwap*(1-0.0261661)),rank(volume),11.4791)))",
    75: "rank(correlation(vwap,volume,4.24304))<rank(correlation(rank(low),rank(adv50),12.4413))",
    76: "-1*max(rank(decay_linear(delta(vwap,1.24383),11.8259)),ts_rank(decay_linear(ts_rank(correlation(indneutralize(low,sector),adv81,8.14941),19.569),17.1543),19.383))",
    77: "min(rank(decay_linear((high+low)/2-vwap,20.0451)),rank(decay_linear(correlation((high+low)/2,adv40,3.1614),5.64125)))",
    78: "rank(correlation(sum(low*0.352233+vwap*(1-0.352233),19.7428),sum(adv40,19.7428),6.83313))^rank(correlation(rank(vwap),rank(volume),5.77492))",
    79: "rank(delta(indneutralize(close*0.60733+open*(1-0.60733),sector),1.23438))<rank(correlation(ts_rank(vwap,3.60973),ts_rank(adv150,9.18637),14.6644))",
    80: "-1*(rank(sign(delta(indneutralize(open*0.868128+high*(1-0.868128),industry),4.04545)))^ts_rank(correlation(high,adv10,5.11456),5.53756))",
    81: "-1*(rank(log(product(rank(rank(correlation(vwap,sum(adv10,49.6054),8.47743))^4),14.9655)))<rank(correlation(rank(vwap),rank(volume),5.07914)))",
    82: "-1*min(rank(decay_linear(delta(open,1.46063),14.8717)),ts_rank(decay_linear(correlation(indneutralize(volume,sector),open,17.4842),6.92131),13.4283))",
    83: "rank(delay((high-low)/(sum(close,5)/5),2))*rank(rank(volume))/(((high-low)/(sum(close,5)/5))/(vwap-close))",
    84: "signedpower(ts_rank(vwap-ts_max(vwap,15.3217),20.7127),delta(close,4.96796))",
    85: "rank(correlation(high*0.876703+close*(1-0.876703),adv30,9.61331))^rank(correlation(ts_rank((high+low)/2,3.70596),ts_rank(volume,10.1595),7.11408))",
    86: "-1*(ts_rank(correlation(close,sum(adv20,14.7444),6.00049),20.4195)<rank(close-vwap))",
    87: "-1*max(rank(decay_linear(delta(close*0.369701+vwap*(1-0.369701),1.91233),2.65461)),ts_rank(decay_linear(abs(correlation(indneutralize(adv81,industry),close,13.4132)),4.89768),14.4535))",
    88: "min(rank(decay_linear(rank(open)+rank(low)-rank(high)-rank(close),8.06882)),ts_rank(decay_linear(correlation(ts_rank(close,8.44728),ts_rank(adv60,20.6966),8.01266),6.65053),2.61957))",
    89: "ts_rank(decay_linear(correlation(low,adv10,6.94279),5.51607),3.79744)-ts_rank(decay_linear(delta(indneutralize(vwap,industry),3.48158),10.1466),15.3012)",
    90: "-1*(rank(close-ts_max(close,4.66719))^ts_rank(correlation(indneutralize(adv40,subindustry),low,5.38375),3.21856))",
    91: "-1*(ts_rank(decay_linear(decay_linear(correlation(indneutralize(close,industry),volume,9.74928),16.398),3.83219),4.8667)-rank(decay_linear(correlation(vwap,adv30,4.01303),2.6809)))",
    92: "min(ts_rank(decay_linear(((high+low)/2+close)<(low+open),14.7221),18.8683),ts_rank(decay_linear(correlation(rank(low),rank(adv30),7.58555),6.94024),6.80584))",
    93: "ts_rank(decay_linear(correlation(indneutralize(vwap,industry),adv81,17.4193),19.848),7.54455)/rank(decay_linear(delta(close*0.524434+vwap*(1-0.524434),2.77377),16.2664))",
    94: "-1*(rank(vwap-ts_min(vwap,11.5783))^ts_rank(correlation(ts_rank(vwap,19.6462),ts_rank(adv60,4.02992),18.0926),2.70756))",
    95: "rank(open-ts_min(open,12.4105))<ts_rank(rank(correlation(sum((high+low)/2,19.1351),sum(adv40,19.1351),12.8742))^5,11.7584)",
    96: "-1*max(ts_rank(decay_linear(correlation(rank(vwap),rank(volume),3.83878),4.16783),8.38151),ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close,7.45404),ts_rank(adv60,4.13242),3.65459),12.6556),14.0365),13.4143))",
    97: "-1*(rank(decay_linear(delta(indneutralize(low*0.721001+vwap*(1-0.721001),industry),3.3705),20.4523))-ts_rank(decay_linear(ts_rank(correlation(ts_rank(low,7.87871),ts_rank(adv60,17.255),4.97547),18.5925),15.7152),6.71659))",
    98: "rank(decay_linear(correlation(vwap,sum(adv5,26.4719),4.58418),7.18088))-rank(decay_linear(ts_rank(ts_argmin(correlation(rank(open),rank(adv15),20.8187),8.62571),6.95668),8.07206))",
    99: "-1*(rank(correlation(sum((high+low)/2,19.8975),sum(adv60,19.8975),8.8136))<rank(correlation(low,volume,6.28259)))",
    100: "-1*((1.5*scale(indneutralize(indneutralize(rank((((close-low)-(high-close))/(high-low))*volume),subindustry),subindustry))-scale(indneutralize(correlation(close,rank(adv20),5)-rank(ts_argmin(close,30)),subindustry)))*(volume/adv20))",
    101: "(close-open)/((high-low)+0.001)",
}


# Minimum input bars needed for each expression, including the current bar.
# Values are derived compositionally: delay/delta add their offset, rolling
# operators add window - 1, and branches use the longest dependency path.
REQUIRED_HISTORY_BARS = (
    24, 8, 10, 9, 10, 10, 67, 15, 6, 5, 4, 2, 5, 10, 5, 5, 24,
    10, 250, 2, 20, 20, 20, 200, 20, 11, 7, 24, 11, 20, 31, 235, 1,
    5, 32, 200, 201, 10, 250, 10, 1, 1, 39, 5, 25, 21, 20, 252, 21,
    9, 21, 240, 10, 1, 17, 10, 31, 16, 26, 10, 197, 50, 240, 148,
    73, 17, 25, 36, 32, 84, 226, 57, 21, 80, 61, 141, 47, 65, 172,
    19, 80, 35, 7, 35, 39, 58, 110, 95, 27, 46, 35, 49, 123, 82, 81,
    103, 119, 56, 87, 30, 1,
)


@dataclass(frozen=True)
class AlphaDefinition:
    number: int
    name: str
    formula: str
    inputs: Mapping[str, str]
    industry_inputs: Mapping[str, str]
    required_history_bars: int
    compute: Callable[[dict[str, pd.DataFrame], Mapping[str, Any] | None], pd.DataFrame]
    parameters: Mapping[str, "AlphaParameter"]


@dataclass(frozen=True)
class AlphaParameter:
    """One numeric formula literal exposed for controlled parameter mining."""

    name: str
    default: int | float
    kind: str
    searchable: bool
    source_line: int


def _adv(data: dict[str, pd.DataFrame], n: float) -> pd.DataFrame:
    return ts_mean(data["amount"], n)


def _bool(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(float)


def alpha1(d, *, returns_threshold=0, returns_stddev_window=20, base_power_exponent=2.0, signed_power_base_argmax_window=5, rank_ts_argmax_signed_power_center=0.5):
    base = where(d['returns'] < returns_threshold, stddev(d['returns'], returns_stddev_window), d['close'])
    return rank(ts_argmax(signed_power(base, base_power_exponent), signed_power_base_argmax_window)) - rank_ts_argmax_signed_power_center


def alpha2(d, *, volume_delta_lag=2, rank_delta_volume_rank_open_close_correlation_window=6):
    return -correlation(rank(delta(np.log(d['volume']), volume_delta_lag)), rank((d['close'] - d['open']) / d['open']), rank_delta_volume_rank_open_close_correlation_window)


def alpha3(d, *, rank_open_rank_volume_correlation_window=10):
    return -correlation(rank(d['open']), rank(d['volume']), rank_open_rank_volume_correlation_window)


def alpha4(d, *, rank_low_rank_window=9):
    return -ts_rank(rank(d['low']), rank_low_rank_window)


def alpha5(d, *, vwap_sum_window=10, ts_sum_vwap_divisor=10):
    return rank(d['open'] - ts_sum(d['vwap'], vwap_sum_window) / ts_sum_vwap_divisor) * -rank(d['close'] - d['vwap']).abs()


def alpha6(d, *, open_volume_correlation_window=10):
    return -correlation(d['open'], d['volume'], open_volume_correlation_window)


def alpha7(d, *, close_delta_lag=7, change_rank_window=60, amount_average_window=20, volume_adv_false_value=-1.0):
    change = delta(d['close'], close_delta_lag)
    signal = -ts_rank(change.abs(), change_rank_window) * sign(change)
    return where(_adv(d, amount_average_window) < d['volume'], signal, volume_adv_false_value)


def alpha8(d, *, open_sum_window=5, returns_sum_window=5, base_delay_lag=10):
    base = ts_sum(d['open'], open_sum_window) * ts_sum(d['returns'], returns_sum_window)
    return -rank(base - delay(base, base_delay_lag))


def alpha9(d, *, close_delta_lag=1, change_minimum_window=5, ts_min_change_threshold=0, change_maximum_window=5, ts_max_change_threshold=0):
    change = delta(d['close'], close_delta_lag)
    trend = (ts_min(change, change_minimum_window) > ts_min_change_threshold) | (ts_max(change, change_maximum_window) < ts_max_change_threshold)
    return where(trend, change, -change)


def alpha10(d, *, close_delta_lag=1, change_minimum_window=4, ts_min_change_threshold=0, change_maximum_window=4, ts_max_change_threshold=0):
    change = delta(d['close'], close_delta_lag)
    trend = (ts_min(change, change_minimum_window) > ts_min_change_threshold) | (ts_max(change, change_maximum_window) < ts_max_change_threshold)
    return rank(where(trend, change, -change))


def alpha11(d, *, spread_maximum_window=3, spread_minimum_window=3, volume_delta_lag=3):
    spread = d['vwap'] - d['close']
    return (rank(ts_max(spread, spread_maximum_window)) + rank(ts_min(spread, spread_minimum_window))) * rank(delta(d['volume'], volume_delta_lag))


def alpha12(d, *, volume_delta_lag=1, close_delta_lag=1):
    return sign(delta(d['volume'], volume_delta_lag)) * -delta(d['close'], close_delta_lag)


def alpha13(d, *, rank_close_rank_volume_covariance_window=5):
    return -rank(covariance(rank(d['close']), rank(d['volume']), rank_close_rank_volume_covariance_window))


def alpha14(d, *, returns_delta_lag=3, open_volume_correlation_window=10):
    return -rank(delta(d['returns'], returns_delta_lag)) * correlation(d['open'], d['volume'], open_volume_correlation_window)


def alpha15(d, *, rank_high_rank_volume_correlation_window=3, rank_correlation_high_sum_window=3):
    return -ts_sum(rank(correlation(rank(d['high']), rank(d['volume']), rank_high_rank_volume_correlation_window)), rank_correlation_high_sum_window)


def alpha16(d, *, rank_high_rank_volume_covariance_window=5):
    return -rank(covariance(rank(d['high']), rank(d['volume']), rank_high_rank_volume_covariance_window))


def alpha17(d, *, close_rank_window=10, close_delta_lag=1, delta_close_delta_lag=1, amount_average_window=20, volume_adv_rank_window=5):
    return -rank(ts_rank(d['close'], close_rank_window)) * rank(delta(delta(d['close'], close_delta_lag), delta_close_delta_lag)) * rank(ts_rank(d['volume'] / _adv(d, amount_average_window), volume_adv_rank_window))


def alpha18(d, *, close_open_stddev_window=5, close_open_correlation_window=10):
    return -rank(stddev((d['close'] - d['open']).abs(), close_open_stddev_window) + d['close'] - d['open'] + correlation(d['close'], d['open'], close_open_correlation_window))


def alpha19(d, *, close_delay_lag=7, close_delta_lag=7, rank_ts_sum_returns_offset=1, ts_sum_returns_offset=1, returns_sum_window=250):
    direction = -sign(d['close'] - delay(d['close'], close_delay_lag) + delta(d['close'], close_delta_lag))
    return direction * (rank_ts_sum_returns_offset + rank(ts_sum_returns_offset + ts_sum(d['returns'], returns_sum_window)))


def alpha20(d, *, high_delay_lag=1, close_delay_lag=1, low_delay_lag=1):
    return -rank(d['open'] - delay(d['high'], high_delay_lag)) * rank(d['open'] - delay(d['close'], close_delay_lag)) * rank(d['open'] - delay(d['low'], low_delay_lag))


def alpha21(d, *, close_mean_window=8, close_mean_window_2=2, close_stddev_window=8, close_stddev_window_2=8, amount_average_window=20, volume_adv_threshold=1, condition1_true_value=-1.0, condition2_true_value=1.0, volume_condition_true_value=1.0, volume_condition_false_value=-1.0):
    mean8 = ts_mean(d['close'], close_mean_window)
    mean2 = ts_mean(d['close'], close_mean_window_2)
    condition1 = mean8 + stddev(d['close'], close_stddev_window) < mean2
    condition2 = mean2 < mean8 - stddev(d['close'], close_stddev_window_2)
    volume_condition = d['volume'] / _adv(d, amount_average_window) >= volume_adv_threshold
    return where(condition1, condition1_true_value, where(condition2, condition2_true_value, where(volume_condition, volume_condition_true_value, volume_condition_false_value)))


def alpha22(d, *, high_volume_correlation_window=5, correlation_high_volume_delta_lag=5, close_stddev_window=20):
    return -delta(correlation(d['high'], d['volume'], high_volume_correlation_window), correlation_high_volume_delta_lag) * rank(stddev(d['close'], close_stddev_window))


def alpha23(d, *, high_mean_window=20, high_delta_lag=2, high_ts_mean_false_value=0.0):
    return where(ts_mean(d['high'], high_mean_window) < d['high'], -delta(d['high'], high_delta_lag), high_ts_mean_false_value)


def alpha24(d, *, close_mean_window=100, ts_mean_close_delta_lag=100, close_delay_lag=100, trend_threshold=0.05, close_minimum_window=100, close_delta_lag=3):
    trend = delta(ts_mean(d['close'], close_mean_window), ts_mean_close_delta_lag) / delay(d['close'], close_delay_lag)
    return where(trend <= trend_threshold, -(d['close'] - ts_min(d['close'], close_minimum_window)), -delta(d['close'], close_delta_lag))


def alpha25(d, *, amount_average_window=20):
    return rank(-d['returns'] * _adv(d, amount_average_window) * d['vwap'] * (d['high'] - d['close']))


def alpha26(d, *, volume_rank_window=5, high_rank_window=5, ts_rank_volume_ts_rank_high_correlation_window=5, correlation_ts_rank_volume_maximum_window=3):
    return -ts_max(correlation(ts_rank(d['volume'], volume_rank_window), ts_rank(d['high'], high_rank_window), ts_rank_volume_ts_rank_high_correlation_window), correlation_ts_rank_volume_maximum_window)


def alpha27(d, *, rank_volume_rank_vwap_correlation_window=6, correlation_rank_volume_sum_window=2, ts_sum_correlation_rank_divisor=2.0, value_threshold=0.5, value_true_value=-1.0, value_false_value=1.0):
    value = rank(ts_sum(correlation(rank(d['volume']), rank(d['vwap']), rank_volume_rank_vwap_correlation_window), correlation_rank_volume_sum_window) / ts_sum_correlation_rank_divisor)
    return where(value > value_threshold, value_true_value, value_false_value)


def alpha28(d, *, amount_average_window=20, adv_low_correlation_window=5, high_low_divisor=2):
    return scale(correlation(_adv(d, amount_average_window), d['low'], adv_low_correlation_window) + (d['high'] + d['low']) / high_low_divisor - d['close'])


def alpha29(d, *, close_center=1, close_delta_lag=5, rank_inner_minimum_window=2, ts_min_rank_inner_sum_window=1, rank_scale_ts_sum_product_window=1, product_rank_scale_minimum_window=5, returns_delay_lag=6, delay_returns_rank_window=5):
    inner = -rank(delta(d['close'] - close_center, close_delta_lag))
    left = ts_min(product(rank(rank(scale(np.log(ts_sum(ts_min(rank(rank(inner)), rank_inner_minimum_window), ts_min_rank_inner_sum_window))))), rank_scale_ts_sum_product_window), product_rank_scale_minimum_window)
    return left + ts_rank(delay(-d['returns'], returns_delay_lag), delay_returns_rank_window)


def alpha30(d, *, close_delay_lag=1, close_delay_lag_2=1, close_delay_lag_3=2, close_delay_lag_4=2, close_delay_lag_5=3, rank_direction_complement_base=1, volume_sum_window=5, volume_sum_window_2=20):
    direction = sign(d['close'] - delay(d['close'], close_delay_lag)) + sign(delay(d['close'], close_delay_lag_2) - delay(d['close'], close_delay_lag_3)) + sign(delay(d['close'], close_delay_lag_4) - delay(d['close'], close_delay_lag_5))
    return (rank_direction_complement_base - rank(direction)) * ts_sum(d['volume'], volume_sum_window) / ts_sum(d['volume'], volume_sum_window_2)


def alpha31(d, *, close_delta_lag=10, rank_delta_close_decay_window=10, close_delta_lag_2=3, amount_average_window=20, adv_low_correlation_window=12):
    return rank(rank(rank(decay_linear(-rank(rank(delta(d['close'], close_delta_lag))), rank_delta_close_decay_window)))) + rank(-delta(d['close'], close_delta_lag_2)) + sign(scale(correlation(_adv(d, amount_average_window), d['low'], adv_low_correlation_window)))


def alpha32(d, *, close_mean_window=7, scale_correlation_vwap_coefficient=20, close_delay_lag=5, vwap_delay_close_correlation_window=230):
    return scale(ts_mean(d['close'], close_mean_window) - d['close']) + scale_correlation_vwap_coefficient * scale(correlation(d['vwap'], delay(d['close'], close_delay_lag), vwap_delay_close_correlation_window))


def alpha33(d, *, open_close_coefficient=-1, open_close_complement_base=1):
    return rank(open_close_coefficient * (open_close_complement_base - d['open'] / d['close']))


def alpha34(d, *, rank_stddev_returns_complement_base=1, returns_stddev_window=2, returns_stddev_window_2=5, rank_delta_close_complement_base=1, close_delta_lag=1):
    return rank(rank_stddev_returns_complement_base - rank(stddev(d['returns'], returns_stddev_window) / stddev(d['returns'], returns_stddev_window_2)) + (rank_delta_close_complement_base - rank(delta(d['close'], close_delta_lag))))


def alpha35(d, *, volume_rank_window=32, ts_rank_low_close_complement_base=1, low_close_high_rank_window=16, ts_rank_returns_complement_base=1, returns_rank_window=32):
    return ts_rank(d['volume'], volume_rank_window) * (ts_rank_low_close_complement_base - ts_rank(d['close'] + d['high'] - d['low'], low_close_high_rank_window)) * (ts_rank_returns_complement_base - ts_rank(d['returns'], returns_rank_window))


def alpha36(d, *, rank_correlation_close_coefficient=2.21, volume_delay_lag=1, close_open_delay_volume_correlation_window=15, rank_open_close_coefficient=0.7, rank_ts_rank_delay_coefficient=0.73, returns_delay_lag=6, delay_returns_rank_window=5, amount_average_window=20, vwap_adv_correlation_window=6, rank_open_close_coefficient_2=0.6, close_mean_window=200):
    return rank_correlation_close_coefficient * rank(correlation(d['close'] - d['open'], delay(d['volume'], volume_delay_lag), close_open_delay_volume_correlation_window)) + rank_open_close_coefficient * rank(d['open'] - d['close']) + rank_ts_rank_delay_coefficient * rank(ts_rank(delay(-d['returns'], returns_delay_lag), delay_returns_rank_window)) + rank(correlation(d['vwap'], _adv(d, amount_average_window), vwap_adv_correlation_window).abs()) + rank_open_close_coefficient_2 * rank((ts_mean(d['close'], close_mean_window) - d['open']) * (d['close'] - d['open']))


def alpha37(d, *, open_close_delay_lag=1, delay_open_close_close_correlation_window=200):
    return rank(correlation(delay(d['open'] - d['close'], open_close_delay_lag), d['close'], delay_open_close_close_correlation_window)) + rank(d['open'] - d['close'])


def alpha38(d, *, close_rank_window=10):
    return -rank(ts_rank(d['close'], close_rank_window)) * rank(d['close'] / d['open'])


def alpha39(d, *, close_delta_lag=7, rank_decay_linear_volume_complement_base=1, amount_average_window=20, volume_adv_decay_window=9, rank_ts_sum_returns_offset=1, returns_sum_window=250):
    return -rank(delta(d['close'], close_delta_lag) * (rank_decay_linear_volume_complement_base - rank(decay_linear(d['volume'] / _adv(d, amount_average_window), volume_adv_decay_window)))) * (rank_ts_sum_returns_offset + rank(ts_sum(d['returns'], returns_sum_window)))


def alpha40(d, *, high_stddev_window=10, high_volume_correlation_window=10):
    return -rank(stddev(d['high'], high_stddev_window)) * correlation(d['high'], d['volume'], high_volume_correlation_window)


def alpha41(d):
    return np.sqrt(d['high'] * d['low']) - d['vwap']


def alpha42(d):
    return rank(d['vwap'] - d['close']) / rank(d['vwap'] + d['close'])


def alpha43(d, *, amount_average_window=20, volume_adv_rank_window=20, close_delta_lag=7, delta_close_rank_window=8):
    return ts_rank(d['volume'] / _adv(d, amount_average_window), volume_adv_rank_window) * ts_rank(-delta(d['close'], close_delta_lag), delta_close_rank_window)


def alpha44(d, *, high_rank_volume_correlation_window=5):
    return -correlation(d['high'], rank(d['volume']), high_rank_volume_correlation_window)


def alpha45(d, *, close_delay_lag=5, delay_close_sum_window=20, ts_sum_delay_close_divisor=20, close_volume_correlation_window=2, close_sum_window=5, close_sum_window_2=20, ts_sum_close_ts_sum_close_correlation_window=2):
    return -rank(ts_sum(delay(d['close'], close_delay_lag), delay_close_sum_window) / ts_sum_delay_close_divisor) * correlation(d['close'], d['volume'], close_volume_correlation_window) * rank(correlation(ts_sum(d['close'], close_sum_window), ts_sum(d['close'], close_sum_window_2), ts_sum_close_ts_sum_close_correlation_window))


def _alpha_trend(
    d,
    *,
    long_delay_lag=20,
    first_short_delay_lag=10,
    first_slope_divisor=10,
    second_short_delay_lag=10,
    second_slope_divisor=10,
):
    return (
        (delay(d["close"], long_delay_lag) - delay(d["close"], first_short_delay_lag))
        / first_slope_divisor
        - (delay(d["close"], second_short_delay_lag) - d["close"])
        / second_slope_divisor
    )


def alpha46(d, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, trend_threshold=0.25, trend_true_value=-1.0, trend_threshold_2=0, trend_true_value_2=1.0, close_delay_lag=1):
    trend = _alpha_trend(d, long_delay_lag=trend_long_delay_lag, first_short_delay_lag=trend_first_short_delay_lag, first_slope_divisor=trend_first_slope_divisor, second_short_delay_lag=trend_second_short_delay_lag, second_slope_divisor=trend_second_slope_divisor)
    return where(trend > trend_threshold, trend_true_value, where(trend < trend_threshold_2, trend_true_value_2, -(d['close'] - delay(d['close'], close_delay_lag))))


def alpha47(d, *, close_divisor=1, amount_average_window=20, high_mean_window=5, vwap_delay_lag=5):
    return rank(close_divisor / d['close']) * d['volume'] / _adv(d, amount_average_window) * d['high'] * rank(d['high'] - d['close']) / ts_mean(d['high'], high_mean_window) - rank(d['vwap'] - delay(d['vwap'], vwap_delay_lag))


def alpha48(d, *, close_delta_lag=1, close_delay_lag=1, delay_close_delta_lag=1, delta_close_delta_delay_close_correlation_window=250, close_delta_lag_2=1, close_delta_lag_3=1, close_delay_lag_2=1, delta_close_delay_power_exponent=2, delta_close_delay_sum_window=250):
    numerator = correlation(delta(d['close'], close_delta_lag), delta(delay(d['close'], close_delay_lag), delay_close_delta_lag), delta_close_delta_delay_close_correlation_window)
    numerator = indneutralize(numerator * delta(d['close'], close_delta_lag_2) / d['close'], d['subindustry'])
    denominator = ts_sum((delta(d['close'], close_delta_lag_3) / delay(d['close'], close_delay_lag_2)) ** delta_close_delay_power_exponent, delta_close_delay_sum_window)
    return numerator / denominator


def alpha49(d, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, alpha_trend_threshold=-0.1, alpha_trend_true_value=1.0, close_delay_lag=1):
    return where(_alpha_trend(d, long_delay_lag=trend_long_delay_lag, first_short_delay_lag=trend_first_short_delay_lag, first_slope_divisor=trend_first_slope_divisor, second_short_delay_lag=trend_second_short_delay_lag, second_slope_divisor=trend_second_slope_divisor) < alpha_trend_threshold, alpha_trend_true_value, -(d['close'] - delay(d['close'], close_delay_lag)))


def alpha50(d, *, rank_volume_rank_vwap_correlation_window=5, rank_correlation_volume_maximum_window=5):
    return -ts_max(rank(correlation(rank(d['volume']), rank(d['vwap']), rank_volume_rank_vwap_correlation_window)), rank_correlation_volume_maximum_window)


def alpha51(d, *, trend_long_delay_lag=20, trend_first_short_delay_lag=10, trend_first_slope_divisor=10, trend_second_short_delay_lag=10, trend_second_slope_divisor=10, alpha_trend_threshold=-0.05, alpha_trend_true_value=1.0, close_delay_lag=1):
    return where(_alpha_trend(d, long_delay_lag=trend_long_delay_lag, first_short_delay_lag=trend_first_short_delay_lag, first_slope_divisor=trend_first_slope_divisor, second_short_delay_lag=trend_second_short_delay_lag, second_slope_divisor=trend_second_slope_divisor) < alpha_trend_threshold, alpha_trend_true_value, -(d['close'] - delay(d['close'], close_delay_lag)))


def alpha52(d, *, low_minimum_window=5, low_minimum_window_2=5, ts_min_low_delay_lag=5, returns_sum_window=240, returns_sum_window_2=20, ts_sum_returns_divisor=220, volume_rank_window=5):
    return (-ts_min(d['low'], low_minimum_window) + delay(ts_min(d['low'], low_minimum_window_2), ts_min_low_delay_lag)) * rank((ts_sum(d['returns'], returns_sum_window) - ts_sum(d['returns'], returns_sum_window_2)) / ts_sum_returns_divisor) * ts_rank(d['volume'], volume_rank_window)


def alpha53(d, *, oscillator_delta_lag=9):
    oscillator = (d['close'] - d['low'] - (d['high'] - d['close'])) / (d['close'] - d['low'])
    return -delta(oscillator, oscillator_delta_lag)


def alpha54(d, *, open_power_exponent=5, close_power_exponent=5):
    return -(d['low'] - d['close']) * d['open'] ** open_power_exponent / ((d['low'] - d['high']) * d['close'] ** close_power_exponent)


def alpha55(d, *, low_minimum_window=12, high_maximum_window=12, low_minimum_window_2=12, rank_stochastic_rank_volume_correlation_window=6):
    stochastic = (d['close'] - ts_min(d['low'], low_minimum_window)) / (ts_max(d['high'], high_maximum_window) - ts_min(d['low'], low_minimum_window_2))
    return -correlation(rank(stochastic), rank(d['volume']), rank_stochastic_rank_volume_correlation_window)


def alpha56(d, *, returns_sum_window=10, returns_sum_window_2=2, ts_sum_returns_sum_window=3):
    return -rank(ts_sum(d['returns'], returns_sum_window) / ts_sum(ts_sum(d['returns'], returns_sum_window_2), ts_sum_returns_sum_window)) * rank(d['returns'] * d['cap'])


def alpha57(d, *, close_argmax_window=30, rank_ts_argmax_close_decay_window=2):
    return -(d['close'] - d['vwap']) / decay_linear(rank(ts_argmax(d['close'], close_argmax_window)), rank_ts_argmax_close_decay_window)


def alpha58(d, *, indneutralize_vwap_sector_volume_correlation_window=3.92795, value_decay_window=7.89291, decay_linear_value_rank_window=5.50322):
    value = correlation(indneutralize(d['vwap'], d['sector']), d['volume'], indneutralize_vwap_sector_volume_correlation_window)
    return -ts_rank(decay_linear(value, value_decay_window), decay_linear_value_rank_window)


def alpha59(d, *, vwap_mix_weight=0.728317, mixed_complement_base=1, mixed_complement_weight=0.728317, indneutralize_mixed_industry_volume_correlation_window=4.25197, value_decay_window=16.2289, decay_linear_value_rank_window=8.19648):
    mixed = d['vwap'] * vwap_mix_weight + d['vwap'] * (mixed_complement_base - mixed_complement_weight)
    value = correlation(indneutralize(mixed, d['industry']), d['volume'], indneutralize_mixed_industry_volume_correlation_window)
    return -ts_rank(decay_linear(value, value_decay_window), decay_linear_value_rank_window)


def alpha60(d, *, scale_rank_oscillator_coefficient=2, close_argmax_window=10):
    oscillator = (d['close'] - d['low'] - (d['high'] - d['close'])) / (d['high'] - d['low']) * d['volume']
    return -(scale_rank_oscillator_coefficient * scale(rank(oscillator)) - scale(rank(ts_argmax(d['close'], close_argmax_window))))


def alpha61(d, *, vwap_minimum_window=16.1219, amount_average_window=180, vwap_adv_correlation_window=17.9282):
    return _bool(rank(d['vwap'] - ts_min(d['vwap'], vwap_minimum_window)) < rank(correlation(d['vwap'], _adv(d, amount_average_window), vwap_adv_correlation_window)))


def alpha62(d, *, amount_average_window=20, adv_sum_window=22.4101, vwap_ts_sum_adv_correlation_window=9.91009, high_low_divisor=2):
    left = rank(correlation(d['vwap'], ts_sum(_adv(d, amount_average_window), adv_sum_window), vwap_ts_sum_adv_correlation_window))
    inner = rank(d['open']) + rank(d['open']) < rank((d['high'] + d['low']) / high_low_divisor) + rank(d['high'])
    return -_bool(left < rank(_bool(inner)))


def alpha63(d, *, indneutralize_close_industry_delta_lag=2.25164, delta_indneutralize_close_decay_window=8.22237, vwap_mix_weight=0.318108, mixed_complement_base=1, mixed_complement_weight=0.318108, amount_average_window=180, adv_sum_window=37.2467, mixed_ts_sum_adv_correlation_window=13.557, correlation_mixed_ts_sum_decay_window=12.2883):
    left = rank(decay_linear(delta(indneutralize(d['close'], d['industry']), indneutralize_close_industry_delta_lag), delta_indneutralize_close_decay_window))
    mixed = d['vwap'] * vwap_mix_weight + d['open'] * (mixed_complement_base - mixed_complement_weight)
    right = rank(decay_linear(correlation(mixed, ts_sum(_adv(d, amount_average_window), adv_sum_window), mixed_ts_sum_adv_correlation_window), correlation_mixed_ts_sum_decay_window))
    return -(left - right)


def alpha64(d, *, open_mix_weight=0.178404, mixed1_complement_base=1, mixed1_complement_weight=0.178404, mixed1_sum_window=12.7054, amount_average_window=120, adv_sum_window=12.7054, ts_sum_mixed1_ts_sum_adv_correlation_window=16.6208, high_low_divisor=2, high_low_mix_weight=0.178404, mixed2_complement_base=1, mixed2_complement_weight=0.178404, mixed2_delta_lag=3.69741):
    mixed1 = d['open'] * open_mix_weight + d['low'] * (mixed1_complement_base - mixed1_complement_weight)
    left = rank(correlation(ts_sum(mixed1, mixed1_sum_window), ts_sum(_adv(d, amount_average_window), adv_sum_window), ts_sum_mixed1_ts_sum_adv_correlation_window))
    mixed2 = (d['high'] + d['low']) / high_low_divisor * high_low_mix_weight + d['vwap'] * (mixed2_complement_base - mixed2_complement_weight)
    return -_bool(left < rank(delta(mixed2, mixed2_delta_lag)))


def alpha65(d, *, open_mix_weight=0.00817205, mixed_complement_base=1, mixed_complement_weight=0.00817205, amount_average_window=60, adv_sum_window=8.6911, mixed_ts_sum_adv_correlation_window=6.40374, open_minimum_window=13.635):
    mixed = d['open'] * open_mix_weight + d['vwap'] * (mixed_complement_base - mixed_complement_weight)
    left = rank(correlation(mixed, ts_sum(_adv(d, amount_average_window), adv_sum_window), mixed_ts_sum_adv_correlation_window))
    return -_bool(left < rank(d['open'] - ts_min(d['open'], open_minimum_window)))


def alpha66(d, *, vwap_delta_lag=3.51013, delta_vwap_decay_window=7.23052, high_low_divisor=2, right_base_decay_window=11.4157, decay_linear_right_base_rank_window=6.72611):
    left = rank(decay_linear(delta(d['vwap'], vwap_delta_lag), delta_vwap_decay_window))
    right_base = (d['low'] - d['vwap']) / (d['open'] - (d['high'] + d['low']) / high_low_divisor)
    right = ts_rank(decay_linear(right_base, right_base_decay_window), decay_linear_right_base_rank_window)
    return -(left + right)


def alpha67(d, *, high_minimum_window=2.14593, amount_average_window=20, indneutralize_vwap_sector_indneutralize_subindustry_adv_correlation_wind=6.02936):
    base = rank(d['high'] - ts_min(d['high'], high_minimum_window))
    exponent = rank(correlation(indneutralize(d['vwap'], d['sector']), indneutralize(_adv(d, amount_average_window), d['subindustry']), indneutralize_vwap_sector_indneutralize_subindustry_adv_correlation_wind))
    return -np.power(base, exponent)


def alpha68(d, *, amount_average_window=15, rank_high_rank_adv_correlation_window=8.91644, correlation_rank_high_rank_window=13.9333, close_mix_weight=0.518371, mixed_complement_base=1, mixed_complement_weight=0.518371, mixed_delta_lag=1.06157):
    left = ts_rank(correlation(rank(d['high']), rank(_adv(d, amount_average_window)), rank_high_rank_adv_correlation_window), correlation_rank_high_rank_window)
    mixed = d['close'] * close_mix_weight + d['low'] * (mixed_complement_base - mixed_complement_weight)
    return -_bool(left < rank(delta(mixed, mixed_delta_lag)))


def alpha69(d, *, indneutralize_vwap_industry_delta_lag=2.72412, delta_indneutralize_vwap_maximum_window=4.79344, close_mix_weight=0.490655, mixed_complement_base=1, mixed_complement_weight=0.490655, amount_average_window=20, mixed_adv_correlation_window=4.92416, correlation_mixed_adv_rank_window=9.0615):
    base = rank(ts_max(delta(indneutralize(d['vwap'], d['industry']), indneutralize_vwap_industry_delta_lag), delta_indneutralize_vwap_maximum_window))
    mixed = d['close'] * close_mix_weight + d['vwap'] * (mixed_complement_base - mixed_complement_weight)
    exponent = ts_rank(correlation(mixed, _adv(d, amount_average_window), mixed_adv_correlation_window), correlation_mixed_adv_rank_window)
    return -np.power(base, exponent)


def alpha70(d, *, vwap_delta_lag=1.29456, amount_average_window=50, indneutralize_close_industry_adv_correlation_window=17.8256, correlation_indneutralize_close_rank_window=17.9171):
    base = rank(delta(d['vwap'], vwap_delta_lag))
    exponent = ts_rank(correlation(indneutralize(d['close'], d['industry']), _adv(d, amount_average_window), indneutralize_close_industry_adv_correlation_window), correlation_indneutralize_close_rank_window)
    return -np.power(base, exponent)


def alpha71(d, *, close_rank_window=3.43976, amount_average_window=180, adv_rank_window=12.0647, ts_rank_close_ts_rank_adv_correlation_window=18.0175, correlation_ts_rank_close_decay_window=4.20501, decay_linear_correlation_ts_rank_rank_window=15.6948, rank_vwap_low_power_exponent=2, rank_vwap_low_decay_window=16.4662, decay_linear_rank_vwap_rank_window=4.4388):
    left = ts_rank(decay_linear(correlation(ts_rank(d['close'], close_rank_window), ts_rank(_adv(d, amount_average_window), adv_rank_window), ts_rank_close_ts_rank_adv_correlation_window), correlation_ts_rank_close_decay_window), decay_linear_correlation_ts_rank_rank_window)
    right = ts_rank(decay_linear(rank(d['low'] + d['open'] - d['vwap'] - d['vwap']) ** rank_vwap_low_power_exponent, rank_vwap_low_decay_window), decay_linear_rank_vwap_rank_window)
    return elementwise_max(left, right)


def alpha72(d, *, high_low_divisor=2, amount_average_window=40, high_low_adv_correlation_window=8.93345, correlation_adv_high_decay_window=10.1519, vwap_rank_window=3.72469, volume_rank_window=18.5188, ts_rank_vwap_ts_rank_volume_correlation_window=6.86671, correlation_ts_rank_vwap_decay_window=2.95011):
    left = rank(decay_linear(correlation((d['high'] + d['low']) / high_low_divisor, _adv(d, amount_average_window), high_low_adv_correlation_window), correlation_adv_high_decay_window))
    right = rank(decay_linear(correlation(ts_rank(d['vwap'], vwap_rank_window), ts_rank(d['volume'], volume_rank_window), ts_rank_vwap_ts_rank_volume_correlation_window), correlation_ts_rank_vwap_decay_window))
    return left / right


def alpha73(d, *, vwap_delta_lag=4.72775, delta_vwap_decay_window=2.91864, open_mix_weight=0.147155, mixed_complement_base=1, mixed_complement_weight=0.147155, mixed_delta_lag=2.03608, mixed_delta_decay_window=3.33829, decay_linear_mixed_delta_rank_window=16.7411):
    left = rank(decay_linear(delta(d['vwap'], vwap_delta_lag), delta_vwap_decay_window))
    mixed = d['open'] * open_mix_weight + d['low'] * (mixed_complement_base - mixed_complement_weight)
    right = ts_rank(decay_linear(-delta(mixed, mixed_delta_lag) / mixed, mixed_delta_decay_window), decay_linear_mixed_delta_rank_window)
    return -elementwise_max(left, right)


def alpha74(d, *, amount_average_window=30, adv_sum_window=37.4843, close_ts_sum_adv_correlation_window=15.1365, high_mix_weight=0.0261661, mixed_complement_base=1, mixed_complement_weight=0.0261661, rank_mixed_rank_volume_correlation_window=11.4791):
    left = rank(correlation(d['close'], ts_sum(_adv(d, amount_average_window), adv_sum_window), close_ts_sum_adv_correlation_window))
    mixed = d['high'] * high_mix_weight + d['vwap'] * (mixed_complement_base - mixed_complement_weight)
    right = rank(correlation(rank(mixed), rank(d['volume']), rank_mixed_rank_volume_correlation_window))
    return -_bool(left < right)


def alpha75(d, *, vwap_volume_correlation_window=4.24304, amount_average_window=50, rank_low_rank_adv_correlation_window=12.4413):
    left = rank(correlation(d['vwap'], d['volume'], vwap_volume_correlation_window))
    right = rank(correlation(rank(d['low']), rank(_adv(d, amount_average_window)), rank_low_rank_adv_correlation_window))
    return _bool(left < right)


def alpha76(d, *, vwap_delta_lag=1.24383, delta_vwap_decay_window=11.8259, amount_average_window=81, indneutralize_low_sector_adv_correlation_window=8.14941, corr_rank_window=19.569, ts_rank_corr_decay_window=17.1543, decay_linear_ts_rank_corr_rank_window=19.383):
    left = rank(decay_linear(delta(d['vwap'], vwap_delta_lag), delta_vwap_decay_window))
    corr = correlation(indneutralize(d['low'], d['sector']), _adv(d, amount_average_window), indneutralize_low_sector_adv_correlation_window)
    right = ts_rank(decay_linear(ts_rank(corr, corr_rank_window), ts_rank_corr_decay_window), decay_linear_ts_rank_corr_rank_window)
    return -elementwise_max(left, right)


def alpha77(d, *, high_low_divisor=2, vwap_high_low_decay_window=20.0451, high_low_divisor_2=2, amount_average_window=40, high_low_adv_correlation_window=3.1614, correlation_adv_high_decay_window=5.64125):
    left = rank(decay_linear((d['high'] + d['low']) / high_low_divisor - d['vwap'], vwap_high_low_decay_window))
    right = rank(decay_linear(correlation((d['high'] + d['low']) / high_low_divisor_2, _adv(d, amount_average_window), high_low_adv_correlation_window), correlation_adv_high_decay_window))
    return elementwise_min(left, right)


def alpha78(d, *, low_mix_weight=0.352233, mixed_complement_base=1, mixed_complement_weight=0.352233, mixed_sum_window=19.7428, amount_average_window=40, adv_sum_window=19.7428, ts_sum_mixed_ts_sum_adv_correlation_window=6.83313, rank_vwap_rank_volume_correlation_window=5.77492):
    mixed = d['low'] * low_mix_weight + d['vwap'] * (mixed_complement_base - mixed_complement_weight)
    base = rank(correlation(ts_sum(mixed, mixed_sum_window), ts_sum(_adv(d, amount_average_window), adv_sum_window), ts_sum_mixed_ts_sum_adv_correlation_window))
    exponent = rank(correlation(rank(d['vwap']), rank(d['volume']), rank_vwap_rank_volume_correlation_window))
    return np.power(base, exponent)


def alpha79(d, *, close_mix_weight=0.60733, mixed_complement_base=1, mixed_complement_weight=0.60733, indneutralize_mixed_sector_delta_lag=1.23438, vwap_rank_window=3.60973, amount_average_window=150, adv_rank_window=9.18637, ts_rank_vwap_ts_rank_adv_correlation_window=14.6644):
    mixed = d['close'] * close_mix_weight + d['open'] * (mixed_complement_base - mixed_complement_weight)
    left = rank(delta(indneutralize(mixed, d['sector']), indneutralize_mixed_sector_delta_lag))
    right = rank(correlation(ts_rank(d['vwap'], vwap_rank_window), ts_rank(_adv(d, amount_average_window), adv_rank_window), ts_rank_vwap_ts_rank_adv_correlation_window))
    return _bool(left < right)


def alpha80(d, *, open_mix_weight=0.868128, mixed_complement_base=1, mixed_complement_weight=0.868128, indneutralize_mixed_industry_delta_lag=4.04545, amount_average_window=10, high_adv_correlation_window=5.11456, correlation_high_adv_rank_window=5.53756):
    mixed = d['open'] * open_mix_weight + d['high'] * (mixed_complement_base - mixed_complement_weight)
    base = rank(sign(delta(indneutralize(mixed, d['industry']), indneutralize_mixed_industry_delta_lag)))
    exponent = ts_rank(correlation(d['high'], _adv(d, amount_average_window), high_adv_correlation_window), correlation_high_adv_rank_window)
    return -np.power(base, exponent)


def alpha81(d, *, amount_average_window=10, adv_sum_window=49.6054, vwap_ts_sum_adv_correlation_window=8.47743, left_constant=4, rank_corr_product_window=14.9655, rank_vwap_rank_volume_correlation_window=5.07914):
    corr = correlation(d['vwap'], ts_sum(_adv(d, amount_average_window), adv_sum_window), vwap_ts_sum_adv_correlation_window)
    left = rank(np.log(product(rank(np.power(rank(corr), left_constant)), rank_corr_product_window)))
    right = rank(correlation(rank(d['vwap']), rank(d['volume']), rank_vwap_rank_volume_correlation_window))
    return -_bool(left < right)


def alpha82(d, *, open_delta_lag=1.46063, delta_open_decay_window=14.8717, indneutralize_volume_sector_open_correlation_window=17.4842, corr_decay_window=6.92131, decay_linear_corr_rank_window=13.4283):
    left = rank(decay_linear(delta(d['open'], open_delta_lag), delta_open_decay_window))
    corr = correlation(indneutralize(d['volume'], d['sector']), d['open'], indneutralize_volume_sector_open_correlation_window)
    right = ts_rank(decay_linear(corr, corr_decay_window), decay_linear_corr_rank_window)
    return -elementwise_min(left, right)


def alpha83(d, *, close_mean_window=5, ratio_delay_lag=2):
    ratio = (d['high'] - d['low']) / ts_mean(d['close'], close_mean_window)
    return rank(delay(ratio, ratio_delay_lag)) * rank(rank(d['volume'])) / (ratio / (d['vwap'] - d['close']))


def alpha84(d, *, vwap_maximum_window=15.3217, vwap_ts_max_rank_window=20.7127, close_delta_lag=4.96796):
    base = ts_rank(d['vwap'] - ts_max(d['vwap'], vwap_maximum_window), vwap_ts_max_rank_window)
    return signed_power(base, delta(d['close'], close_delta_lag))


def alpha85(d, *, high_mix_weight=0.876703, mixed_complement_base=1, mixed_complement_weight=0.876703, amount_average_window=30, mixed_adv_correlation_window=9.61331, high_low_divisor=2, high_low_rank_window=3.70596, volume_rank_window=10.1595, ts_rank_high_low_ts_rank_volume_correlation_window=7.11408):
    mixed = d['high'] * high_mix_weight + d['close'] * (mixed_complement_base - mixed_complement_weight)
    base = rank(correlation(mixed, _adv(d, amount_average_window), mixed_adv_correlation_window))
    exponent = rank(correlation(ts_rank((d['high'] + d['low']) / high_low_divisor, high_low_rank_window), ts_rank(d['volume'], volume_rank_window), ts_rank_high_low_ts_rank_volume_correlation_window))
    return np.power(base, exponent)


def alpha86(d, *, amount_average_window=20, adv_sum_window=14.7444, close_ts_sum_adv_correlation_window=6.00049, correlation_close_ts_sum_rank_window=20.4195):
    left = ts_rank(correlation(d['close'], ts_sum(_adv(d, amount_average_window), adv_sum_window), close_ts_sum_adv_correlation_window), correlation_close_ts_sum_rank_window)
    return -_bool(left < rank(d['close'] - d['vwap']))


def alpha87(d, *, close_mix_weight=0.369701, mixed_complement_base=1, mixed_complement_weight=0.369701, mixed_delta_lag=1.91233, delta_mixed_decay_window=2.65461, amount_average_window=81, indneutralize_industry_adv_close_correlation_window=13.4132, corr_decay_window=4.89768, decay_linear_corr_rank_window=14.4535):
    mixed = d['close'] * close_mix_weight + d['vwap'] * (mixed_complement_base - mixed_complement_weight)
    left = rank(decay_linear(delta(mixed, mixed_delta_lag), delta_mixed_decay_window))
    corr = correlation(indneutralize(_adv(d, amount_average_window), d['industry']), d['close'], indneutralize_industry_adv_close_correlation_window).abs()
    right = ts_rank(decay_linear(corr, corr_decay_window), decay_linear_corr_rank_window)
    return -elementwise_max(left, right)


def alpha88(d, *, rank_close_high_decay_window=8.06882, close_rank_window=8.44728, amount_average_window=60, adv_rank_window=20.6966, ts_rank_close_ts_rank_adv_correlation_window=8.01266, corr_decay_window=6.65053, decay_linear_corr_rank_window=2.61957):
    left = rank(decay_linear(rank(d['open']) + rank(d['low']) - rank(d['high']) - rank(d['close']), rank_close_high_decay_window))
    corr = correlation(ts_rank(d['close'], close_rank_window), ts_rank(_adv(d, amount_average_window), adv_rank_window), ts_rank_close_ts_rank_adv_correlation_window)
    right = ts_rank(decay_linear(corr, corr_decay_window), decay_linear_corr_rank_window)
    return elementwise_min(left, right)


def alpha89(d, *, amount_average_window=10, low_adv_correlation_window=6.94279, correlation_low_adv_decay_window=5.51607, decay_linear_correlation_low_rank_window=3.79744, indneutralize_vwap_industry_delta_lag=3.48158, delta_indneutralize_vwap_decay_window=10.1466, decay_linear_delta_indneutralize_rank_window=15.3012):
    left = ts_rank(decay_linear(correlation(d['low'], _adv(d, amount_average_window), low_adv_correlation_window), correlation_low_adv_decay_window), decay_linear_correlation_low_rank_window)
    right = ts_rank(decay_linear(delta(indneutralize(d['vwap'], d['industry']), indneutralize_vwap_industry_delta_lag), delta_indneutralize_vwap_decay_window), decay_linear_delta_indneutralize_rank_window)
    return left - right


def alpha90(d, *, close_maximum_window=4.66719, amount_average_window=40, indneutralize_subindustry_adv_low_correlation_window=5.38375, correlation_low_indneutralize_rank_window=3.21856):
    base = rank(d['close'] - ts_max(d['close'], close_maximum_window))
    exponent = ts_rank(correlation(indneutralize(_adv(d, amount_average_window), d['subindustry']), d['low'], indneutralize_subindustry_adv_low_correlation_window), correlation_low_indneutralize_rank_window)
    return -np.power(base, exponent)


def alpha91(d, *, indneutralize_close_industry_volume_correlation_window=9.74928, corr1_decay_window=16.398, decay_linear_corr1_decay_window=3.83219, decay_linear_corr1_rank_window=4.8667, amount_average_window=30, vwap_adv_correlation_window=4.01303, correlation_vwap_adv_decay_window=2.6809):
    corr1 = correlation(indneutralize(d['close'], d['industry']), d['volume'], indneutralize_close_industry_volume_correlation_window)
    left = ts_rank(decay_linear(decay_linear(corr1, corr1_decay_window), decay_linear_corr1_decay_window), decay_linear_corr1_rank_window)
    right = rank(decay_linear(correlation(d['vwap'], _adv(d, amount_average_window), vwap_adv_correlation_window), correlation_vwap_adv_decay_window))
    return -(left - right)


def alpha92(d, *, high_low_divisor=2, condition_decay_window=14.7221, decay_linear_condition_rank_window=18.8683, amount_average_window=30, rank_low_rank_adv_correlation_window=7.58555, correlation_rank_low_decay_window=6.94024, decay_linear_correlation_rank_rank_window=6.80584):
    condition = _bool((d['high'] + d['low']) / high_low_divisor + d['close'] < d['low'] + d['open'])
    left = ts_rank(decay_linear(condition, condition_decay_window), decay_linear_condition_rank_window)
    right = ts_rank(decay_linear(correlation(rank(d['low']), rank(_adv(d, amount_average_window)), rank_low_rank_adv_correlation_window), correlation_rank_low_decay_window), decay_linear_correlation_rank_rank_window)
    return elementwise_min(left, right)


def alpha93(d, *, amount_average_window=81, indneutralize_vwap_industry_adv_correlation_window=17.4193, correlation_indneutralize_vwap_decay_window=19.848, decay_linear_correlation_indneutralize_rank_window=7.54455, close_mix_weight=0.524434, mixed_complement_base=1, mixed_complement_weight=0.524434, mixed_delta_lag=2.77377, delta_mixed_decay_window=16.2664):
    left = ts_rank(decay_linear(correlation(indneutralize(d['vwap'], d['industry']), _adv(d, amount_average_window), indneutralize_vwap_industry_adv_correlation_window), correlation_indneutralize_vwap_decay_window), decay_linear_correlation_indneutralize_rank_window)
    mixed = d['close'] * close_mix_weight + d['vwap'] * (mixed_complement_base - mixed_complement_weight)
    right = rank(decay_linear(delta(mixed, mixed_delta_lag), delta_mixed_decay_window))
    return left / right


def alpha94(d, *, vwap_minimum_window=11.5783, vwap_rank_window=19.6462, amount_average_window=60, adv_rank_window=4.02992, ts_rank_vwap_ts_rank_adv_correlation_window=18.0926, correlation_ts_rank_vwap_rank_window=2.70756):
    base = rank(d['vwap'] - ts_min(d['vwap'], vwap_minimum_window))
    exponent = ts_rank(correlation(ts_rank(d['vwap'], vwap_rank_window), ts_rank(_adv(d, amount_average_window), adv_rank_window), ts_rank_vwap_ts_rank_adv_correlation_window), correlation_ts_rank_vwap_rank_window)
    return -np.power(base, exponent)


def alpha95(d, *, open_minimum_window=12.4105, high_low_divisor=2, high_low_sum_window=19.1351, amount_average_window=40, adv_sum_window=19.1351, ts_sum_high_low_ts_sum_adv_correlation_window=12.8742, right_constant=5, rank_corr_rank_window=11.7584):
    left = rank(d['open'] - ts_min(d['open'], open_minimum_window))
    corr = correlation(ts_sum((d['high'] + d['low']) / high_low_divisor, high_low_sum_window), ts_sum(_adv(d, amount_average_window), adv_sum_window), ts_sum_high_low_ts_sum_adv_correlation_window)
    right = ts_rank(np.power(rank(corr), right_constant), rank_corr_rank_window)
    return _bool(left < right)


def alpha96(d, *, rank_vwap_rank_volume_correlation_window=3.83878, correlation_rank_vwap_decay_window=4.16783, decay_linear_correlation_rank_rank_window=8.38151, close_rank_window=7.45404, amount_average_window=60, adv_rank_window=4.13242, ts_rank_close_ts_rank_adv_correlation_window=3.65459, corr_argmax_window=12.6556, ts_argmax_corr_decay_window=14.0365, decay_linear_ts_argmax_corr_rank_window=13.4143):
    left = ts_rank(decay_linear(correlation(rank(d['vwap']), rank(d['volume']), rank_vwap_rank_volume_correlation_window), correlation_rank_vwap_decay_window), decay_linear_correlation_rank_rank_window)
    corr = correlation(ts_rank(d['close'], close_rank_window), ts_rank(_adv(d, amount_average_window), adv_rank_window), ts_rank_close_ts_rank_adv_correlation_window)
    right = ts_rank(decay_linear(ts_argmax(corr, corr_argmax_window), ts_argmax_corr_decay_window), decay_linear_ts_argmax_corr_rank_window)
    return -elementwise_max(left, right)


def alpha97(d, *, low_mix_weight=0.721001, mixed_complement_base=1, mixed_complement_weight=0.721001, indneutralize_mixed_industry_delta_lag=3.3705, delta_indneutralize_mixed_decay_window=20.4523, low_rank_window=7.87871, amount_average_window=60, adv_rank_window=17.255, ts_rank_low_ts_rank_adv_correlation_window=4.97547, corr_rank_window=18.5925, ts_rank_corr_decay_window=15.7152, decay_linear_ts_rank_corr_rank_window=6.71659):
    mixed = d['low'] * low_mix_weight + d['vwap'] * (mixed_complement_base - mixed_complement_weight)
    left = rank(decay_linear(delta(indneutralize(mixed, d['industry']), indneutralize_mixed_industry_delta_lag), delta_indneutralize_mixed_decay_window))
    corr = correlation(ts_rank(d['low'], low_rank_window), ts_rank(_adv(d, amount_average_window), adv_rank_window), ts_rank_low_ts_rank_adv_correlation_window)
    right = ts_rank(decay_linear(ts_rank(corr, corr_rank_window), ts_rank_corr_decay_window), decay_linear_ts_rank_corr_rank_window)
    return -(left - right)


def alpha98(d, *, amount_average_window=5, adv_sum_window=26.4719, vwap_ts_sum_adv_correlation_window=4.58418, correlation_vwap_ts_sum_decay_window=7.18088, amount_average_window_2=15, rank_open_rank_adv_correlation_window=20.8187, corr_argmin_window=8.62571, ts_argmin_corr_rank_window=6.95668, ts_rank_ts_argmin_corr_decay_window=8.07206):
    left = rank(decay_linear(correlation(d['vwap'], ts_sum(_adv(d, amount_average_window), adv_sum_window), vwap_ts_sum_adv_correlation_window), correlation_vwap_ts_sum_decay_window))
    corr = correlation(rank(d['open']), rank(_adv(d, amount_average_window_2)), rank_open_rank_adv_correlation_window)
    right = rank(decay_linear(ts_rank(ts_argmin(corr, corr_argmin_window), ts_argmin_corr_rank_window), ts_rank_ts_argmin_corr_decay_window))
    return left - right


def alpha99(d, *, high_low_divisor=2, high_low_sum_window=19.8975, amount_average_window=60, adv_sum_window=19.8975, ts_sum_high_low_ts_sum_adv_correlation_window=8.8136, low_volume_correlation_window=6.28259):
    left = rank(correlation(ts_sum((d['high'] + d['low']) / high_low_divisor, high_low_sum_window), ts_sum(_adv(d, amount_average_window), adv_sum_window), ts_sum_high_low_ts_sum_adv_correlation_window))
    right = rank(correlation(d['low'], d['volume'], low_volume_correlation_window))
    return -_bool(left < right)


def alpha100(d, *, amount_average_window=20, close_rank_adv_correlation_window=5, close_argmin_window=30, scale_first_coefficient=1.5, amount_average_window_2=20):
    oscillator = (d['close'] - d['low'] - (d['high'] - d['close'])) / (d['high'] - d['low']) * d['volume']
    first = indneutralize(indneutralize(rank(oscillator), d['subindustry']), d['subindustry'])
    second = correlation(d['close'], rank(_adv(d, amount_average_window)), close_rank_adv_correlation_window) - rank(ts_argmin(d['close'], close_argmin_window))
    second = indneutralize(second, d['subindustry'])
    return -((scale_first_coefficient * scale(first) - scale(second)) * (d['volume'] / _adv(d, amount_average_window_2)))


def alpha101(d, *, high_low_epsilon=0.001):
    return (d['close'] - d['open']) / (d['high'] - d['low'] + high_low_epsilon)


def _parameter_kind(name: str) -> str:
    base = re.sub(r"_\d+$", "", name)
    if base.endswith("_window"):
        return "window"
    if base.endswith("_lag"):
        return "lag"
    if base.endswith("_threshold"):
        return "threshold"
    if base.endswith("_exponent"):
        return "exponent"
    if base.endswith("_epsilon"):
        return "epsilon"
    if base.endswith("_weight"):
        return "weight"
    if base.endswith("_divisor"):
        return "divisor"
    if base.endswith(("_constant", "_base", "_value", "_center")):
        return "constant"
    return "coefficient"


ALPHA_FUNCTIONS = {number: globals()[f"alpha{number}"] for number in range(1, 102)}
ALPHA_PARAMETER_SPECS = {
    number: {
        name: AlphaParameter(
            name=name,
            default=default,
            kind=_parameter_kind(name),
            searchable=_parameter_kind(name) in {"window", "lag", "threshold", "exponent", "weight"},
            source_line=function.__code__.co_firstlineno,
        )
        for name, default in (function.__kwdefaults__ or {}).items()
    }
    for number, function in ALPHA_FUNCTIONS.items()
}


def _input_specs(formula: str) -> tuple[dict[str, str], dict[str, str]]:
    market = {}
    industry = {}
    for variable, (argument, metric) in MARKET_INPUTS.items():
        if variable == "amount":
            present = bool(re.search(r"\badv\d+\b", formula, flags=re.IGNORECASE))
        else:
            present = bool(re.search(rf"\b{variable}\b", formula, flags=re.IGNORECASE))
        if present:
            market[argument] = metric
    for variable, (argument, scheme) in INDUSTRY_INPUTS.items():
        if re.search(rf"\b{variable}\b", formula, flags=re.IGNORECASE):
            industry[argument] = scheme
    return market, industry


def _definition(number: int) -> AlphaDefinition:
    formula = FORMULAS[number]
    inputs, industry_inputs = _input_specs(formula)
    return AlphaDefinition(
        number=number,
        name=f"ALPHA{number}",
        formula=formula,
        inputs=inputs,
        industry_inputs=industry_inputs,
        required_history_bars=REQUIRED_HISTORY_BARS[number - 1],
        compute=ALPHA_FUNCTIONS[number],
        parameters=ALPHA_PARAMETER_SPECS[number],
    )


ALPHA_DEFINITIONS = {number: _definition(number) for number in range(1, 102)}


def get_definition(name_or_number: str | int) -> AlphaDefinition:
    if isinstance(name_or_number, str):
        match = re.fullmatch(r"ALPHA(\d+)(?:_timing)?", name_or_number.upper())
        if not match:
            raise KeyError(f"unknown Alpha101 name: {name_or_number}")
        number = int(match.group(1))
    else:
        number = int(name_or_number)
    try:
        return ALPHA_DEFINITIONS[number]
    except KeyError as exc:
        raise KeyError(f"Alpha101 number must be 1..101: {number}") from exc


def resolve_compute_kwargs(
    name_or_number: str | int,
    compute_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate flat formula parameters while preserving paper defaults."""
    definition = get_definition(name_or_number)
    supplied = dict(compute_kwargs or {})
    unknown = sorted(set(supplied) - set(definition.parameters))
    if unknown:
        raise KeyError(f"unknown {definition.name} formula parameter(s): {', '.join(unknown)}")
    resolved = {}
    for name, spec in definition.parameters.items():
        value = supplied.get(name, spec.default)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{definition.name}.{name} must be numeric")
        if isinstance(spec.default, float) or isinstance(value, (float, np.floating)):
            value = float(value)
        else:
            value = int(value)
        if not np.isfinite(value):
            raise ValueError(f"{definition.name}.{name} must be finite")
        if spec.kind in {"window", "lag"} and value < 1:
            raise ValueError(f"{definition.name}.{name} must be >= 1")
        if spec.kind == "exponent" and value == 0:
            raise ValueError(f"{definition.name}.{name} must be non-zero")
        resolved[name] = value
    return resolved


def default_compute_kwargs(name_or_number: str | int) -> dict[str, Any]:
    return resolve_compute_kwargs(name_or_number)


def required_history_bars_for_alpha(
    name_or_number: str | int,
    compute_kwargs: Mapping[str, Any] | None = None,
) -> int:
    """Return a conservative history requirement for a parameterized formula."""
    definition = get_definition(name_or_number)
    resolved = resolve_compute_kwargs(name_or_number, compute_kwargs)
    required = int(definition.required_history_bars)
    for name, spec in definition.parameters.items():
        if spec.kind not in {"window", "lag"}:
            continue
        default = int(round(float(spec.default)))
        value = int(round(float(resolved[name])))
        required += max(0, value - default)
    return max(1, required)


def compute_alpha(name_or_number: str | int, **wides) -> pd.DataFrame:
    definition = get_definition(name_or_number)
    if "formula_params" in wides:
        raise TypeError("formula_params is no longer supported; pass flat compute_kwargs parameters")
    parameter_names = set(definition.parameters)
    supplied_params = {name: wides.pop(name) for name in list(wides) if name in parameter_names}
    known_inputs = {
        argument
        for argument, _metric in (*MARKET_INPUTS.values(), *INDUSTRY_INPUTS.values())
    }
    unknown = sorted(set(wides) - known_inputs)
    if unknown:
        raise TypeError(f"unknown {definition.name} compute argument(s): {', '.join(unknown)}")
    resolved_params = resolve_compute_kwargs(name_or_number, supplied_params)
    reference = next((value for value in wides.values() if isinstance(value, pd.DataFrame)), None)
    if reference is None:
        raise ValueError(f"{definition.name} requires at least one input DataFrame")
    data = {}
    for variable, (argument, _metric) in MARKET_INPUTS.items():
        if argument not in wides:
            continue
        values = wides[argument]
        data[variable] = values / 100.0 if variable == "returns" else values
    for variable, (argument, _scheme) in INDUSTRY_INPUTS.items():
        if argument in wides:
            data[variable] = wides[argument]
    result = definition.compute(data, **resolved_params)
    return clean_inf(result).reindex_like(reference)


__all__ = [
    "ALPHA_DEFINITIONS",
    "ALPHA_FUNCTIONS",
    "AlphaDefinition",
    "AlphaParameter",
    "ALPHA_PARAMETER_SPECS",
    "FORMULAS",
    "MARKET_INPUTS",
    "INDUSTRY_INPUTS",
    "REQUIRED_HISTORY_BARS",
    "compute_alpha",
    "default_compute_kwargs",
    "get_definition",
    "required_history_bars_for_alpha",
    "resolve_compute_kwargs",
]
