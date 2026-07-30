"""Canonical WorldQuant Alpha101 formulas on daily wide DataFrames."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Mapping

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
    compute: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


def _adv(data: dict[str, pd.DataFrame], n: float) -> pd.DataFrame:
    return ts_mean(data["amount"], n)


def _bool(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(float)


def alpha1(d):
    base = where(d["returns"] < 0, stddev(d["returns"], 20), d["close"])
    return rank(ts_argmax(signed_power(base, 2.0), 5)) - 0.5


def alpha2(d):
    return -correlation(rank(delta(np.log(d["volume"]), 2)), rank((d["close"] - d["open"]) / d["open"]), 6)


def alpha3(d):
    return -correlation(rank(d["open"]), rank(d["volume"]), 10)


def alpha4(d):
    return -ts_rank(rank(d["low"]), 9)


def alpha5(d):
    return rank(d["open"] - ts_sum(d["vwap"], 10) / 10) * -rank(d["close"] - d["vwap"]).abs()


def alpha6(d):
    return -correlation(d["open"], d["volume"], 10)


def alpha7(d):
    change = delta(d["close"], 7)
    signal = -ts_rank(change.abs(), 60) * sign(change)
    return where(_adv(d, 20) < d["volume"], signal, -1.0)


def alpha8(d):
    base = ts_sum(d["open"], 5) * ts_sum(d["returns"], 5)
    return -rank(base - delay(base, 10))


def alpha9(d):
    change = delta(d["close"], 1)
    trend = (ts_min(change, 5) > 0) | (ts_max(change, 5) < 0)
    return where(trend, change, -change)


def alpha10(d):
    change = delta(d["close"], 1)
    trend = (ts_min(change, 4) > 0) | (ts_max(change, 4) < 0)
    return rank(where(trend, change, -change))


def alpha11(d):
    spread = d["vwap"] - d["close"]
    return (rank(ts_max(spread, 3)) + rank(ts_min(spread, 3))) * rank(delta(d["volume"], 3))


def alpha12(d):
    return sign(delta(d["volume"], 1)) * -delta(d["close"], 1)


def alpha13(d):
    return -rank(covariance(rank(d["close"]), rank(d["volume"]), 5))


def alpha14(d):
    return -rank(delta(d["returns"], 3)) * correlation(d["open"], d["volume"], 10)


def alpha15(d):
    return -ts_sum(rank(correlation(rank(d["high"]), rank(d["volume"]), 3)), 3)


def alpha16(d):
    return -rank(covariance(rank(d["high"]), rank(d["volume"]), 5))


def alpha17(d):
    return (
        -rank(ts_rank(d["close"], 10))
        * rank(delta(delta(d["close"], 1), 1))
        * rank(ts_rank(d["volume"] / _adv(d, 20), 5))
    )


def alpha18(d):
    return -rank(
        stddev((d["close"] - d["open"]).abs(), 5)
        + d["close"] - d["open"]
        + correlation(d["close"], d["open"], 10)
    )


def alpha19(d):
    direction = -sign((d["close"] - delay(d["close"], 7)) + delta(d["close"], 7))
    return direction * (1 + rank(1 + ts_sum(d["returns"], 250)))


def alpha20(d):
    return (
        -rank(d["open"] - delay(d["high"], 1))
        * rank(d["open"] - delay(d["close"], 1))
        * rank(d["open"] - delay(d["low"], 1))
    )


def alpha21(d):
    mean8 = ts_mean(d["close"], 8)
    mean2 = ts_mean(d["close"], 2)
    condition1 = mean8 + stddev(d["close"], 8) < mean2
    condition2 = mean2 < mean8 - stddev(d["close"], 8)
    volume_condition = d["volume"] / _adv(d, 20) >= 1
    return where(condition1, -1.0, where(condition2, 1.0, where(volume_condition, 1.0, -1.0)))


def alpha22(d):
    return -delta(correlation(d["high"], d["volume"], 5), 5) * rank(stddev(d["close"], 20))


def alpha23(d):
    return where(ts_mean(d["high"], 20) < d["high"], -delta(d["high"], 2), 0.0)


def alpha24(d):
    trend = delta(ts_mean(d["close"], 100), 100) / delay(d["close"], 100)
    return where(trend <= 0.05, -(d["close"] - ts_min(d["close"], 100)), -delta(d["close"], 3))


def alpha25(d):
    return rank(-d["returns"] * _adv(d, 20) * d["vwap"] * (d["high"] - d["close"]))


def alpha26(d):
    return -ts_max(correlation(ts_rank(d["volume"], 5), ts_rank(d["high"], 5), 5), 3)


def alpha27(d):
    value = rank(ts_sum(correlation(rank(d["volume"]), rank(d["vwap"]), 6), 2) / 2.0)
    return where(value > 0.5, -1.0, 1.0)


def alpha28(d):
    return scale(correlation(_adv(d, 20), d["low"], 5) + (d["high"] + d["low"]) / 2 - d["close"])


def alpha29(d):
    inner = -rank(delta(d["close"] - 1, 5))
    left = ts_min(product(rank(rank(scale(np.log(ts_sum(ts_min(rank(rank(inner)), 2), 1))))), 1), 5)
    return left + ts_rank(delay(-d["returns"], 6), 5)


def alpha30(d):
    direction = (
        sign(d["close"] - delay(d["close"], 1))
        + sign(delay(d["close"], 1) - delay(d["close"], 2))
        + sign(delay(d["close"], 2) - delay(d["close"], 3))
    )
    return (1 - rank(direction)) * ts_sum(d["volume"], 5) / ts_sum(d["volume"], 20)


def alpha31(d):
    return (
        rank(rank(rank(decay_linear(-rank(rank(delta(d["close"], 10))), 10))))
        + rank(-delta(d["close"], 3))
        + sign(scale(correlation(_adv(d, 20), d["low"], 12)))
    )


def alpha32(d):
    return scale(ts_mean(d["close"], 7) - d["close"]) + 20 * scale(
        correlation(d["vwap"], delay(d["close"], 5), 230)
    )


def alpha33(d):
    return rank(-1 * (1 - d["open"] / d["close"]))


def alpha34(d):
    return rank(
        (1 - rank(stddev(d["returns"], 2) / stddev(d["returns"], 5)))
        + (1 - rank(delta(d["close"], 1)))
    )


def alpha35(d):
    return (
        ts_rank(d["volume"], 32)
        * (1 - ts_rank(d["close"] + d["high"] - d["low"], 16))
        * (1 - ts_rank(d["returns"], 32))
    )


def alpha36(d):
    return (
        2.21 * rank(correlation(d["close"] - d["open"], delay(d["volume"], 1), 15))
        + 0.7 * rank(d["open"] - d["close"])
        + 0.73 * rank(ts_rank(delay(-d["returns"], 6), 5))
        + rank(correlation(d["vwap"], _adv(d, 20), 6).abs())
        + 0.6 * rank((ts_mean(d["close"], 200) - d["open"]) * (d["close"] - d["open"]))
    )


def alpha37(d):
    return rank(correlation(delay(d["open"] - d["close"], 1), d["close"], 200)) + rank(
        d["open"] - d["close"]
    )


def alpha38(d):
    return -rank(ts_rank(d["close"], 10)) * rank(d["close"] / d["open"])


def alpha39(d):
    return (
        -rank(delta(d["close"], 7) * (1 - rank(decay_linear(d["volume"] / _adv(d, 20), 9))))
        * (1 + rank(ts_sum(d["returns"], 250)))
    )


def alpha40(d):
    return -rank(stddev(d["high"], 10)) * correlation(d["high"], d["volume"], 10)


def alpha41(d):
    return np.sqrt(d["high"] * d["low"]) - d["vwap"]


def alpha42(d):
    return rank(d["vwap"] - d["close"]) / rank(d["vwap"] + d["close"])


def alpha43(d):
    return ts_rank(d["volume"] / _adv(d, 20), 20) * ts_rank(-delta(d["close"], 7), 8)


def alpha44(d):
    return -correlation(d["high"], rank(d["volume"]), 5)


def alpha45(d):
    return (
        -rank(ts_sum(delay(d["close"], 5), 20) / 20)
        * correlation(d["close"], d["volume"], 2)
        * rank(correlation(ts_sum(d["close"], 5), ts_sum(d["close"], 20), 2))
    )


def _alpha_trend(d):
    return (delay(d["close"], 20) - delay(d["close"], 10)) / 10 - (
        delay(d["close"], 10) - d["close"]
    ) / 10


def alpha46(d):
    trend = _alpha_trend(d)
    return where(trend > 0.25, -1.0, where(trend < 0, 1.0, -(d["close"] - delay(d["close"], 1))))


def alpha47(d):
    return (
        rank(1 / d["close"])
        * d["volume"] / _adv(d, 20)
        * d["high"] * rank(d["high"] - d["close"]) / ts_mean(d["high"], 5)
        - rank(d["vwap"] - delay(d["vwap"], 5))
    )


def alpha48(d):
    numerator = correlation(delta(d["close"], 1), delta(delay(d["close"], 1), 1), 250)
    numerator = indneutralize(numerator * delta(d["close"], 1) / d["close"], d["subindustry"])
    denominator = ts_sum((delta(d["close"], 1) / delay(d["close"], 1)) ** 2, 250)
    return numerator / denominator


def alpha49(d):
    return where(_alpha_trend(d) < -0.1, 1.0, -(d["close"] - delay(d["close"], 1)))


def alpha50(d):
    return -ts_max(rank(correlation(rank(d["volume"]), rank(d["vwap"]), 5)), 5)


def alpha51(d):
    return where(_alpha_trend(d) < -0.05, 1.0, -(d["close"] - delay(d["close"], 1)))


def alpha52(d):
    return (
        (-ts_min(d["low"], 5) + delay(ts_min(d["low"], 5), 5))
        * rank((ts_sum(d["returns"], 240) - ts_sum(d["returns"], 20)) / 220)
        * ts_rank(d["volume"], 5)
    )


def alpha53(d):
    oscillator = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / (
        d["close"] - d["low"]
    )
    return -delta(oscillator, 9)


def alpha54(d):
    return -(d["low"] - d["close"]) * d["open"] ** 5 / (
        (d["low"] - d["high"]) * d["close"] ** 5
    )


def alpha55(d):
    stochastic = (d["close"] - ts_min(d["low"], 12)) / (
        ts_max(d["high"], 12) - ts_min(d["low"], 12)
    )
    return -correlation(rank(stochastic), rank(d["volume"]), 6)


def alpha56(d):
    return -rank(ts_sum(d["returns"], 10) / ts_sum(ts_sum(d["returns"], 2), 3)) * rank(
        d["returns"] * d["cap"]
    )


def alpha57(d):
    return -(d["close"] - d["vwap"]) / decay_linear(rank(ts_argmax(d["close"], 30)), 2)


def alpha58(d):
    value = correlation(indneutralize(d["vwap"], d["sector"]), d["volume"], 3.92795)
    return -ts_rank(decay_linear(value, 7.89291), 5.50322)


def alpha59(d):
    mixed = d["vwap"] * 0.728317 + d["vwap"] * (1 - 0.728317)
    value = correlation(indneutralize(mixed, d["industry"]), d["volume"], 4.25197)
    return -ts_rank(decay_linear(value, 16.2289), 8.19648)


def alpha60(d):
    oscillator = (((d["close"] - d["low"]) - (d["high"] - d["close"])) / (
        d["high"] - d["low"]
    )) * d["volume"]
    return -(2 * scale(rank(oscillator)) - scale(rank(ts_argmax(d["close"], 10))))


def alpha61(d):
    return _bool(rank(d["vwap"] - ts_min(d["vwap"], 16.1219)) < rank(correlation(d["vwap"], _adv(d, 180), 17.9282)))


def alpha62(d):
    left = rank(correlation(d["vwap"], ts_sum(_adv(d, 20), 22.4101), 9.91009))
    inner = (rank(d["open"]) + rank(d["open"])) < (rank((d["high"] + d["low"]) / 2) + rank(d["high"]))
    return -_bool(left < rank(_bool(inner)))


def alpha63(d):
    left = rank(decay_linear(delta(indneutralize(d["close"], d["industry"]), 2.25164), 8.22237))
    mixed = d["vwap"] * 0.318108 + d["open"] * (1 - 0.318108)
    right = rank(decay_linear(correlation(mixed, ts_sum(_adv(d, 180), 37.2467), 13.557), 12.2883))
    return -(left - right)


def alpha64(d):
    mixed1 = d["open"] * 0.178404 + d["low"] * (1 - 0.178404)
    left = rank(correlation(ts_sum(mixed1, 12.7054), ts_sum(_adv(d, 120), 12.7054), 16.6208))
    mixed2 = ((d["high"] + d["low"]) / 2) * 0.178404 + d["vwap"] * (1 - 0.178404)
    return -_bool(left < rank(delta(mixed2, 3.69741)))


def alpha65(d):
    mixed = d["open"] * 0.00817205 + d["vwap"] * (1 - 0.00817205)
    left = rank(correlation(mixed, ts_sum(_adv(d, 60), 8.6911), 6.40374))
    return -_bool(left < rank(d["open"] - ts_min(d["open"], 13.635)))


def alpha66(d):
    left = rank(decay_linear(delta(d["vwap"], 3.51013), 7.23052))
    right_base = (d["low"] - d["vwap"]) / (d["open"] - (d["high"] + d["low"]) / 2)
    right = ts_rank(decay_linear(right_base, 11.4157), 6.72611)
    return -(left + right)


def alpha67(d):
    base = rank(d["high"] - ts_min(d["high"], 2.14593))
    exponent = rank(
        correlation(
            indneutralize(d["vwap"], d["sector"]),
            indneutralize(_adv(d, 20), d["subindustry"]),
            6.02936,
        )
    )
    return -np.power(base, exponent)


def alpha68(d):
    left = ts_rank(correlation(rank(d["high"]), rank(_adv(d, 15)), 8.91644), 13.9333)
    mixed = d["close"] * 0.518371 + d["low"] * (1 - 0.518371)
    return -_bool(left < rank(delta(mixed, 1.06157)))


def alpha69(d):
    base = rank(ts_max(delta(indneutralize(d["vwap"], d["industry"]), 2.72412), 4.79344))
    mixed = d["close"] * 0.490655 + d["vwap"] * (1 - 0.490655)
    exponent = ts_rank(correlation(mixed, _adv(d, 20), 4.92416), 9.0615)
    return -np.power(base, exponent)


def alpha70(d):
    base = rank(delta(d["vwap"], 1.29456))
    exponent = ts_rank(correlation(indneutralize(d["close"], d["industry"]), _adv(d, 50), 17.8256), 17.9171)
    return -np.power(base, exponent)


def alpha71(d):
    left = ts_rank(
        decay_linear(correlation(ts_rank(d["close"], 3.43976), ts_rank(_adv(d, 180), 12.0647), 18.0175), 4.20501),
        15.6948,
    )
    right = ts_rank(decay_linear(rank(d["low"] + d["open"] - d["vwap"] - d["vwap"]) ** 2, 16.4662), 4.4388)
    return elementwise_max(left, right)


def alpha72(d):
    left = rank(decay_linear(correlation((d["high"] + d["low"]) / 2, _adv(d, 40), 8.93345), 10.1519))
    right = rank(
        decay_linear(correlation(ts_rank(d["vwap"], 3.72469), ts_rank(d["volume"], 18.5188), 6.86671), 2.95011)
    )
    return left / right


def alpha73(d):
    left = rank(decay_linear(delta(d["vwap"], 4.72775), 2.91864))
    mixed = d["open"] * 0.147155 + d["low"] * (1 - 0.147155)
    right = ts_rank(decay_linear(-delta(mixed, 2.03608) / mixed, 3.33829), 16.7411)
    return -elementwise_max(left, right)


def alpha74(d):
    left = rank(correlation(d["close"], ts_sum(_adv(d, 30), 37.4843), 15.1365))
    mixed = d["high"] * 0.0261661 + d["vwap"] * (1 - 0.0261661)
    right = rank(correlation(rank(mixed), rank(d["volume"]), 11.4791))
    return -_bool(left < right)


def alpha75(d):
    left = rank(correlation(d["vwap"], d["volume"], 4.24304))
    right = rank(correlation(rank(d["low"]), rank(_adv(d, 50)), 12.4413))
    return _bool(left < right)


def alpha76(d):
    left = rank(decay_linear(delta(d["vwap"], 1.24383), 11.8259))
    corr = correlation(indneutralize(d["low"], d["sector"]), _adv(d, 81), 8.14941)
    right = ts_rank(decay_linear(ts_rank(corr, 19.569), 17.1543), 19.383)
    return -elementwise_max(left, right)


def alpha77(d):
    left = rank(decay_linear((d["high"] + d["low"]) / 2 - d["vwap"], 20.0451))
    right = rank(decay_linear(correlation((d["high"] + d["low"]) / 2, _adv(d, 40), 3.1614), 5.64125))
    return elementwise_min(left, right)


def alpha78(d):
    mixed = d["low"] * 0.352233 + d["vwap"] * (1 - 0.352233)
    base = rank(correlation(ts_sum(mixed, 19.7428), ts_sum(_adv(d, 40), 19.7428), 6.83313))
    exponent = rank(correlation(rank(d["vwap"]), rank(d["volume"]), 5.77492))
    return np.power(base, exponent)


def alpha79(d):
    mixed = d["close"] * 0.60733 + d["open"] * (1 - 0.60733)
    left = rank(delta(indneutralize(mixed, d["sector"]), 1.23438))
    right = rank(correlation(ts_rank(d["vwap"], 3.60973), ts_rank(_adv(d, 150), 9.18637), 14.6644))
    return _bool(left < right)


def alpha80(d):
    mixed = d["open"] * 0.868128 + d["high"] * (1 - 0.868128)
    base = rank(sign(delta(indneutralize(mixed, d["industry"]), 4.04545)))
    exponent = ts_rank(correlation(d["high"], _adv(d, 10), 5.11456), 5.53756)
    return -np.power(base, exponent)


def alpha81(d):
    corr = correlation(d["vwap"], ts_sum(_adv(d, 10), 49.6054), 8.47743)
    left = rank(np.log(product(rank(np.power(rank(corr), 4)), 14.9655)))
    right = rank(correlation(rank(d["vwap"]), rank(d["volume"]), 5.07914))
    return -_bool(left < right)


def alpha82(d):
    left = rank(decay_linear(delta(d["open"], 1.46063), 14.8717))
    corr = correlation(indneutralize(d["volume"], d["sector"]), d["open"], 17.4842)
    right = ts_rank(decay_linear(corr, 6.92131), 13.4283)
    return -elementwise_min(left, right)


def alpha83(d):
    ratio = (d["high"] - d["low"]) / ts_mean(d["close"], 5)
    return rank(delay(ratio, 2)) * rank(rank(d["volume"])) / (ratio / (d["vwap"] - d["close"]))


def alpha84(d):
    base = ts_rank(d["vwap"] - ts_max(d["vwap"], 15.3217), 20.7127)
    return signed_power(base, delta(d["close"], 4.96796))


def alpha85(d):
    mixed = d["high"] * 0.876703 + d["close"] * (1 - 0.876703)
    base = rank(correlation(mixed, _adv(d, 30), 9.61331))
    exponent = rank(
        correlation(ts_rank((d["high"] + d["low"]) / 2, 3.70596), ts_rank(d["volume"], 10.1595), 7.11408)
    )
    return np.power(base, exponent)


def alpha86(d):
    left = ts_rank(correlation(d["close"], ts_sum(_adv(d, 20), 14.7444), 6.00049), 20.4195)
    return -_bool(left < rank(d["close"] - d["vwap"]))


def alpha87(d):
    mixed = d["close"] * 0.369701 + d["vwap"] * (1 - 0.369701)
    left = rank(decay_linear(delta(mixed, 1.91233), 2.65461))
    corr = correlation(indneutralize(_adv(d, 81), d["industry"]), d["close"], 13.4132).abs()
    right = ts_rank(decay_linear(corr, 4.89768), 14.4535)
    return -elementwise_max(left, right)


def alpha88(d):
    left = rank(decay_linear(rank(d["open"]) + rank(d["low"]) - rank(d["high"]) - rank(d["close"]), 8.06882))
    corr = correlation(ts_rank(d["close"], 8.44728), ts_rank(_adv(d, 60), 20.6966), 8.01266)
    right = ts_rank(decay_linear(corr, 6.65053), 2.61957)
    return elementwise_min(left, right)


def alpha89(d):
    left = ts_rank(decay_linear(correlation(d["low"], _adv(d, 10), 6.94279), 5.51607), 3.79744)
    right = ts_rank(decay_linear(delta(indneutralize(d["vwap"], d["industry"]), 3.48158), 10.1466), 15.3012)
    return left - right


def alpha90(d):
    base = rank(d["close"] - ts_max(d["close"], 4.66719))
    exponent = ts_rank(correlation(indneutralize(_adv(d, 40), d["subindustry"]), d["low"], 5.38375), 3.21856)
    return -np.power(base, exponent)


def alpha91(d):
    corr1 = correlation(indneutralize(d["close"], d["industry"]), d["volume"], 9.74928)
    left = ts_rank(decay_linear(decay_linear(corr1, 16.398), 3.83219), 4.8667)
    right = rank(decay_linear(correlation(d["vwap"], _adv(d, 30), 4.01303), 2.6809))
    return -(left - right)


def alpha92(d):
    condition = _bool((d["high"] + d["low"]) / 2 + d["close"] < d["low"] + d["open"])
    left = ts_rank(decay_linear(condition, 14.7221), 18.8683)
    right = ts_rank(decay_linear(correlation(rank(d["low"]), rank(_adv(d, 30)), 7.58555), 6.94024), 6.80584)
    return elementwise_min(left, right)


def alpha93(d):
    left = ts_rank(decay_linear(correlation(indneutralize(d["vwap"], d["industry"]), _adv(d, 81), 17.4193), 19.848), 7.54455)
    mixed = d["close"] * 0.524434 + d["vwap"] * (1 - 0.524434)
    right = rank(decay_linear(delta(mixed, 2.77377), 16.2664))
    return left / right


def alpha94(d):
    base = rank(d["vwap"] - ts_min(d["vwap"], 11.5783))
    exponent = ts_rank(correlation(ts_rank(d["vwap"], 19.6462), ts_rank(_adv(d, 60), 4.02992), 18.0926), 2.70756)
    return -np.power(base, exponent)


def alpha95(d):
    left = rank(d["open"] - ts_min(d["open"], 12.4105))
    corr = correlation(ts_sum((d["high"] + d["low"]) / 2, 19.1351), ts_sum(_adv(d, 40), 19.1351), 12.8742)
    right = ts_rank(np.power(rank(corr), 5), 11.7584)
    return _bool(left < right)


def alpha96(d):
    left = ts_rank(decay_linear(correlation(rank(d["vwap"]), rank(d["volume"]), 3.83878), 4.16783), 8.38151)
    corr = correlation(ts_rank(d["close"], 7.45404), ts_rank(_adv(d, 60), 4.13242), 3.65459)
    right = ts_rank(decay_linear(ts_argmax(corr, 12.6556), 14.0365), 13.4143)
    return -elementwise_max(left, right)


def alpha97(d):
    mixed = d["low"] * 0.721001 + d["vwap"] * (1 - 0.721001)
    left = rank(decay_linear(delta(indneutralize(mixed, d["industry"]), 3.3705), 20.4523))
    corr = correlation(ts_rank(d["low"], 7.87871), ts_rank(_adv(d, 60), 17.255), 4.97547)
    right = ts_rank(decay_linear(ts_rank(corr, 18.5925), 15.7152), 6.71659)
    return -(left - right)


def alpha98(d):
    left = rank(decay_linear(correlation(d["vwap"], ts_sum(_adv(d, 5), 26.4719), 4.58418), 7.18088))
    corr = correlation(rank(d["open"]), rank(_adv(d, 15)), 20.8187)
    right = rank(decay_linear(ts_rank(ts_argmin(corr, 8.62571), 6.95668), 8.07206))
    return left - right


def alpha99(d):
    left = rank(correlation(ts_sum((d["high"] + d["low"]) / 2, 19.8975), ts_sum(_adv(d, 60), 19.8975), 8.8136))
    right = rank(correlation(d["low"], d["volume"], 6.28259))
    return -_bool(left < right)


def alpha100(d):
    oscillator = (((d["close"] - d["low"]) - (d["high"] - d["close"])) / (
        d["high"] - d["low"]
    )) * d["volume"]
    first = indneutralize(indneutralize(rank(oscillator), d["subindustry"]), d["subindustry"])
    second = correlation(d["close"], rank(_adv(d, 20)), 5) - rank(ts_argmin(d["close"], 30))
    second = indneutralize(second, d["subindustry"])
    return -((1.5 * scale(first) - scale(second)) * (d["volume"] / _adv(d, 20)))


def alpha101(d):
    return (d["close"] - d["open"]) / ((d["high"] - d["low"]) + 0.001)


ALPHA_FUNCTIONS = {number: globals()[f"alpha{number}"] for number in range(1, 102)}


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


def compute_alpha(name_or_number: str | int, **wides) -> pd.DataFrame:
    definition = get_definition(name_or_number)
    data = {}
    for variable, (argument, _metric) in MARKET_INPUTS.items():
        if argument not in wides:
            continue
        values = wides[argument]
        data[variable] = values / 100.0 if variable == "returns" else values
    for variable, (argument, _scheme) in INDUSTRY_INPUTS.items():
        if argument in wides:
            data[variable] = wides[argument]
    result = definition.compute(data)
    return clean_inf(result).reindex_like(next(iter(wides.values())))


__all__ = [
    "ALPHA_DEFINITIONS",
    "ALPHA_FUNCTIONS",
    "AlphaDefinition",
    "FORMULAS",
    "MARKET_INPUTS",
    "INDUSTRY_INPUTS",
    "REQUIRED_HISTORY_BARS",
    "compute_alpha",
    "get_definition",
]
