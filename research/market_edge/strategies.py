"""The eight pre-registered hypotheses (see HYPOTHESES.md) and their robustness grids.

Conventions: a signal computed with the close of day ``t`` is traded at the close of ``t+1``
(``target.shift(1)``) unless stated otherwise. All costs are 5 bp per one-way trade by default.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
from backtest import Result, TRADING_DAYS, extract_trades, simulate_close, simulate_open_close
from gs_quant.timeseries import moving_average, volatility

warnings.filterwarnings('ignore', category=FutureWarning)
COST_BPS = 5.0


# ----------------------------------------------------------------------------- helpers

def sma(px: pd.Series, n: int) -> pd.Series:
    """Simple moving average via gs_quant, NaN during warm-up."""
    return moving_average(px, n).reindex(px.index)


def rsi_wilder(px: pd.Series, n: int) -> pd.Series:
    delta = px.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + gain / loss)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    iso = index.isocalendar()
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).max().sort_values().values)


def _binary_holding(exposure: pd.Series, label: str) -> pd.Series:
    return pd.Series(np.where(exposure > 1e-9, label, None), index=exposure.index, dtype=object)


def _finish(name, hyp, sim, bench, rf, oos, benchmark, params, holding=None) -> Result:
    daily = sim.assign(bench=bench.reindex(sim.index), rf=rf.reindex(sim.index).ffill())
    trades = extract_trades(daily, holding) if holding is not None else None
    return Result(name=name, hypothesis=hyp, daily=daily, oos_start=pd.Timestamp(oos), benchmark=benchmark,
                  trades=trades, params=params)


def _spy_frame():
    spy = data.yahoo('SPY')
    rf = data.us_market()['RF'].reindex(spy.index).ffill()
    return spy, rf


# ----------------------------------------------------------------------------- H1 trend

def h1_trend(window: int = 200, cost_bps: float = COST_BPS) -> Result:
    m = data.us_market()
    px = (1 + m['MKT']).cumprod()
    ma = sma(px, window)
    signal = (px > ma).astype(float).where(ma.notna())
    start = ma.first_valid_index()
    sim = simulate_close(signal.shift(1).to_frame('MKT'), m[['MKT']].loc[start:], m['RF'], cost_bps)
    return _finish(f'Tendencia MM{window}', 'H1', sim, m['MKT'], m['RF'], '2007-01-01', 'MKT',
                   {'window': window}, _binary_holding(sim['exposure'], 'MKT'))


def h1_other_market(symbol: str, window: int = 200, cost_bps: float = COST_BPS) -> Result:
    """Same trend rule on another market (price index, cash at 0 %): robustness only."""
    px = data.yahoo(symbol)['adj_close']
    rets = px.pct_change()
    ma = sma(px, window)
    signal = (px > ma).astype(float).where(ma.notna())
    start = ma.first_valid_index()
    zero = pd.Series(0.0, index=px.index)
    sim = simulate_close(signal.shift(1).to_frame(symbol), rets.to_frame(symbol).loc[start:], zero, cost_bps)
    return _finish(f'Tendencia MM{window} {symbol}', 'H1', sim, rets, zero, '2007-01-01', symbol,
                   {'symbol': symbol, 'window': window}, _binary_holding(sim['exposure'], symbol))


# ----------------------------------------------------------------------------- H2 volatility management

def h2_vol_managed(lookback: int = 21, cap: float = 2.0, target_vol: float = 0.16,
                   cost_bps: float = COST_BPS) -> Result:
    m = data.us_market()
    px = (1 + m['MKT']).cumprod()
    vol = volatility(px, lookback).reindex(px.index) / 100.0  # gs_quant: annualised, in %
    weight = (target_vol / vol).clip(upper=cap)
    me = month_ends(px.index)
    signal = weight.where(px.index.isin(me))
    start = vol.first_valid_index()
    sim = simulate_close(signal.shift(1).to_frame('MKT'), m[['MKT']].loc[start:], m['RF'], cost_bps)
    return _finish(f'Vol. objetivo {lookback}d tope {cap:g}x', 'H2', sim, m['MKT'], m['RF'], '2017-01-01', 'MKT',
                   {'lookback': lookback, 'cap': cap})


# ----------------------------------------------------------------------------- H3 industry momentum

def h3_industry_momentum(formation: int = 6, top: int = 5, cost_bps: float = COST_BPS) -> Result:
    m = data.us_market()
    ind = data.industries(30).reindex(m.index)
    px = (1 + ind.fillna(0.0)).cumprod()
    me = month_ends(m.index)
    pm = px.loc[me]
    form = pm / pm.shift(formation) - 1.0
    ranks = form.rank(axis=1, ascending=False, method='first')
    weights = (ranks <= top).astype(float).div(top).where(form.notna().any(axis=1), axis=0)
    weights = weights.dropna(how='all')
    target = weights.reindex(m.index).shift(1)
    start = weights.index[0]
    sim = simulate_close(target, ind.loc[start:], m['RF'], cost_bps)
    return _finish(f'Momentum industrias {formation}m top{top}', 'H3', sim, m['MKT'], m['RF'], '1999-01-01', 'MKT',
                   {'formation': formation, 'top': top})


# ----------------------------------------------------------------------------- H4 turn of the month

def h4_turn_of_month(before: int = 1, after: int = 3, cost_bps: float = COST_BPS) -> Result:
    m = data.us_market()
    idx = m.index
    key = pd.Series(idx.year * 12 + idx.month, index=idx)
    pos = key.groupby(key).cumcount() + 1
    count = key.map(key.value_counts())
    from_end = pos - count - 1  # -1 = last trading day of the month
    in_window = ((pos <= after) | (from_end >= -before)).astype(float)
    target = in_window.shift(-1)  # calendar is known in advance: buy at the close before the window
    sim = simulate_close(target.to_frame('MKT'), m[['MKT']], m['RF'], cost_bps)
    return _finish(f'Cambio de mes [-{before},+{after}]', 'H4', sim, m['MKT'], m['RF'], '1988-01-01', 'MKT',
                   {'before': before, 'after': after}, _binary_holding(sim['exposure'], 'MKT'))


# ----------------------------------------------------------------------------- H5 short-term reversal (RSI-2)

def h5_rsi2(threshold: float = 10, exit_ma: int = 5, cost_bps: float = COST_BPS) -> Result:
    spy, rf = _spy_frame()
    c = spy['adj_close']
    ma200, ma_exit, rsi = sma(c, 200), sma(c, exit_ma), rsi_wilder(c, 2)
    cv, m2, mx, rv = c.to_numpy(), ma200.to_numpy(), ma_exit.to_numpy(), rsi.to_numpy()
    after_close = np.zeros(len(c))
    pos = 0.0
    for t in range(len(c)):
        if np.isnan(m2[t]):
            continue
        if pos == 0.0 and cv[t] > m2[t] and rv[t] < threshold:
            pos = 1.0
        elif pos == 1.0 and cv[t] > mx[t]:
            pos = 0.0
        after_close[t] = pos
    w_intraday = pd.Series(after_close, index=c.index).shift(1).fillna(0.0)  # enter/exit at next open
    w_overnight = w_intraday.shift(1).fillna(0.0)
    r_on = spy['adj_open'] / c.shift(1) - 1.0
    r_id = c / spy['adj_open'] - 1.0
    start = ma200.first_valid_index()
    sim = simulate_open_close(w_overnight, w_intraday, r_on, r_id, rf, cost_bps).loc[start:]
    held = (w_overnight + w_intraday).loc[start:]
    return _finish(f'RSI(2)<{threshold:g} salida MM{exit_ma}', 'H5', sim, c.pct_change(), rf, '2009-01-01', 'SPY',
                   {'threshold': threshold, 'exit_ma': exit_ma}, _binary_holding(held, 'SPY'))


# ----------------------------------------------------------------------------- H6 overnight

def h6_overnight(cost_bps: float = COST_BPS) -> Result:
    spy, rf = _spy_frame()
    c = spy['adj_close']
    r_on = (spy['adj_open'] / c.shift(1) - 1.0).iloc[1:]
    r_id = (c / spy['adj_open'] - 1.0).iloc[1:]
    ones = pd.Series(1.0, index=r_on.index)
    sim = simulate_open_close(ones, ones * 0.0, r_on, r_id, rf, cost_bps)
    holding = pd.Series(sim.index.strftime('%Y-%m-%d'), index=sim.index, dtype=object)  # one trade per night
    return _finish('Overnight (cierre→apertura)', 'H6', sim, c.pct_change(), rf, '2008-01-01', 'SPY', {}, holding)


# ----------------------------------------------------------------------------- H7 dual momentum

def h7_dual_momentum(lookback: int = 12, cost_bps: float = COST_BPS) -> Result:
    spy, rf = _spy_frame()
    px = pd.concat({s: data.yahoo(s)['adj_close'] for s in ('SPY', 'EFA', 'AGG')}, axis=1).reindex(spy.index).ffill()
    rets = px.pct_change()
    rf_idx = (1 + rf).cumprod()
    me = month_ends(px.index)
    me = me[me >= data.yahoo('AGG').index[0]]
    pm, rfm = px.loc[me], rf_idx.loc[me]
    mom, rf_mom = pm / pm.shift(lookback) - 1.0, rfm / rfm.shift(lookback) - 1.0
    valid = mom[['SPY', 'EFA']].notna().all(axis=1)
    choice = np.where(mom['SPY'] > rf_mom, np.where(mom['SPY'] >= mom['EFA'], 'SPY', 'EFA'), 'AGG')
    weights = pd.DataFrame(0.0, index=me, columns=px.columns)
    for d, ch in zip(me, choice):
        weights.loc[d, ch] = 1.0
    weights = weights[valid.values]
    target = weights.reindex(px.index).shift(1)
    start = weights.index[0]
    sim = simulate_close(target, rets.loc[start:], rf, cost_bps)
    held = pd.Series(choice[valid.values], index=weights.index).reindex(px.index).shift(2).ffill().loc[start:]
    held[sim['exposure'] < 1e-9] = None
    return _finish(f'Dual momentum {lookback}m', 'H7', sim, rets['SPY'], rf, '2014-01-01', 'SPY',
                   {'lookback': lookback}, held)


# ----------------------------------------------------------------------------- H8 machine learning

def _features(px: pd.Series, rf: pd.Series) -> pd.DataFrame:
    lp = np.log(px)
    dr = px.pct_change()
    f = pd.DataFrame(index=px.index)
    for n in (5, 21, 63, 126, 252):
        f[f'r{n}'] = lp.diff(n)
    f['mom_12_1'] = lp.shift(21) - lp.shift(252)
    f['vol21'] = dr.rolling(21).std() * np.sqrt(TRADING_DAYS)
    f['vol63'] = dr.rolling(63).std() * np.sqrt(TRADING_DAYS)
    f['vol_ratio'] = f['vol21'] / f['vol63']
    f['d50'] = px / px.rolling(50).mean() - 1.0
    f['d200'] = px / px.rolling(200).mean() - 1.0
    f['rsi14'] = rsi_wilder(px, 14)
    f['dd252'] = px / px.rolling(252).max() - 1.0
    f['rf'] = rf * TRADING_DAYS
    f['month'] = px.index.month
    return f


def _make_model(kind: str):
    if kind == 'gbm':
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=50,
                                              l2_regularization=1.0, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def h8_ml(model: str = 'gbm', horizon: str = 'week', first_oos_year: int = 1950,
          cost_bps: float = COST_BPS) -> Result:
    m = data.us_market()
    idx = m.index
    px = (1 + m['MKT']).cumprod()
    feats = _features(px, m['RF'])
    dec = week_ends(idx) if horizon == 'week' else month_ends(idx)
    dec = dec[feats.loc[dec].notna().all(axis=1).values]
    pos = idx.get_indexer(dec)
    exec_pos = pos + 1
    keep = exec_pos < len(idx)
    dec, exec_pos = dec[keep], exec_pos[keep]
    grow_m = np.concatenate([[1.0], np.cumprod(1 + m['MKT'].to_numpy())])
    grow_f = np.concatenate([[1.0], np.cumprod(1 + m['RF'].to_numpy())])
    nxt = np.append(exec_pos[1:], -1)
    label = np.full(len(dec), np.nan)
    label_end = pd.Series(pd.NaT, index=dec)
    for i, (e, n_) in enumerate(zip(exec_pos, nxt)):
        if n_ < 0:
            continue
        hold_ret = grow_m[n_ + 1] / grow_m[e + 1] - 1.0  # days e+1 .. n_
        hold_rf = grow_f[n_ + 1] / grow_f[e + 1] - 1.0
        label[i] = float(hold_ret > hold_rf)
        label_end.iloc[i] = idx[n_]
    X = feats.loc[dec]
    proba = pd.Series(np.nan, index=dec)
    for year in range(first_oos_year, idx[-1].year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        test = (dec >= year_start) & (dec < pd.Timestamp(year=year + 1, month=1, day=1))
        if not test.any():
            continue
        train = (label_end < year_start).to_numpy() & ~np.isnan(label)
        clf = _make_model(model).fit(X[train], label[train])
        proba[test] = clf.predict_proba(X[test])[:, 1]
    weights = (proba.dropna() > 0.5).astype(float)
    target = pd.Series(weights.values, index=idx[idx.get_indexer(weights.index) + 1])
    start = target.index[0]
    sim = simulate_close(target.to_frame('MKT'), m[['MKT']].loc[start:], m['RF'], cost_bps)
    names = {'gbm': 'Gradient boosting', 'logit': 'Logística'}
    return _finish(f'{names[model]} {"semanal" if horizon == "week" else "mensual"}', 'H8', sim, m['MKT'], m['RF'],
                   f'{first_oos_year}-01-01', 'MKT', {'model': model, 'horizon': horizon},
                   _binary_holding(sim['exposure'], 'MKT'))


# ----------------------------------------------------------------------------- registry

HYPOTHESES = {
    'H1': ('Seguimiento de tendencia', h1_trend, {'window': 200},
           [{'window': w} for w in (50, 100, 150, 200, 250, 300)]),
    'H2': ('Gestión por volatilidad', h2_vol_managed, {'lookback': 21, 'cap': 2.0},
           [{'lookback': lb, 'cap': c} for lb in (10, 21, 42, 63) for c in (1.0, 1.5, 2.0)]),
    'H3': ('Momentum de industrias', h3_industry_momentum, {'formation': 6, 'top': 5},
           [{'formation': f, 'top': k} for f in (3, 6, 9, 12) for k in (3, 5, 8)]),
    'H4': ('Efecto cambio de mes', h4_turn_of_month, {'before': 1, 'after': 3},
           [{'before': b, 'after': a} for b, a in ((1, 3), (1, 2), (2, 3), (1, 4), (3, 3))]),
    'H5': ('Reversión a corto plazo RSI(2)', h5_rsi2, {'threshold': 10, 'exit_ma': 5},
           [{'threshold': t, 'exit_ma': e} for t in (5, 10, 15, 20, 25) for e in (3, 5, 10)]),
    'H6': ('Efecto overnight', h6_overnight, {}, [{}]),
    'H7': ('Dual momentum', h7_dual_momentum, {'lookback': 12},
           [{'lookback': lb} for lb in (3, 6, 9, 12)]),
    'H8': ('Machine learning (control)', h8_ml, {'model': 'gbm', 'horizon': 'week'},
           [{'model': mo, 'horizon': h} for mo in ('gbm', 'logit') for h in ('week', 'month')]),
}

OTHER_MARKETS = ['^GSPC', '^N225', '^FTSE', '^GDAXI', '^HSI', '^IBEX', '^STOXX50E', 'BTC-USD']
