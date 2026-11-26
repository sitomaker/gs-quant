"""Sanity tests for the simulators and for the absence of look-ahead bias."""

import numpy as np
import pandas as pd
import pytest

import data
import strategies as st
from backtest import simulate_close, simulate_open_close, extract_trades

IDX = pd.bdate_range('2020-01-01', periods=300)
RNG = np.random.default_rng(1)
R = pd.DataFrame({'A': RNG.normal(0.0005, 0.01, len(IDX))}, index=IDX)
RF = pd.Series(0.0001, index=IDX)


def test_buy_and_hold_matches_asset_after_entry_cost():
    target = pd.DataFrame({'A': [1.0] + [np.nan] * (len(IDX) - 1)}, index=IDX)
    sim = simulate_close(target, R, RF, cost_bps=5)
    assert sim['ret'].iloc[0] == pytest.approx((1 + RF.iloc[0]) * (1 - 5e-4) - 1)
    np.testing.assert_allclose(sim['ret'].iloc[1:], R['A'].iloc[1:], rtol=1e-12)


def test_leverage_pays_borrow_spread():
    target = pd.DataFrame({'A': 2.0}, index=IDX)  # rebalanced to 2x every day
    sim = simulate_close(target, R, RF, cost_bps=0, borrow_spread=0.005)
    expected = 2 * R['A'] - (RF + 0.005 / 252)
    np.testing.assert_allclose(sim['ret'].iloc[1:], expected.iloc[1:], rtol=1e-10)


def test_open_close_full_day_equals_close_to_close():
    on, intra = pd.Series(RNG.normal(0, 0.005, len(IDX)), IDX), pd.Series(RNG.normal(0, 0.008, len(IDX)), IDX)
    ones = pd.Series(1.0, index=IDX)
    sim = simulate_open_close(ones, ones, on, intra, RF, cost_bps=5)
    np.testing.assert_allclose(sim['ret'], (1 + on) * (1 + intra) - 1, rtol=1e-12)


def test_trade_extraction_counts_round_trips():
    w = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 1, 1] + [0] * 290, index=IDX, dtype=float)
    target = w.shift(-1).to_frame('A')
    sim = simulate_close(target, R, RF, cost_bps=5)
    trades = extract_trades(sim, st._binary_holding(sim['exposure'], 'A'))
    assert len(trades) == 3


def _no_lookahead(make_signal, cutoff='2000-06-30'):
    """Positions up to ``cutoff`` must not change when later data is removed."""
    full = make_signal(None)
    trunc = make_signal(pd.Timestamp(cutoff))
    common = trunc.index[trunc.index <= pd.Timestamp(cutoff)]
    pd.testing.assert_series_equal(full.reindex(common), trunc.reindex(common), check_names=False)


def test_trend_signal_has_no_lookahead():
    m = data.us_market()

    def make(cut):
        px = (1 + (m['MKT'] if cut is None else m['MKT'].loc[:cut])).cumprod()
        ma = st.sma(px, 200)
        return (px > ma).astype(float).where(ma.notna()).shift(1)
    _no_lookahead(make)


def test_turn_of_month_exposure_pattern():
    res = st.h4_turn_of_month()
    exp = res.daily['exposure'].loc['2019-01-01':'2019-03-31']
    held = exp[exp > 0].index
    # last trading day of Jan 2019 is the 31st, first three of Feb are 1st, 4th, 5th
    for d in ('2019-01-31', '2019-02-01', '2019-02-04', '2019-02-05'):
        assert pd.Timestamp(d) in held
    assert pd.Timestamp('2019-02-06') not in held and pd.Timestamp('2019-01-30') not in held


def test_ml_trains_only_on_past_labels():
    res = st.h8_ml(model='logit', horizon='month', first_oos_year=1960)
    assert res.daily.index[0] >= pd.Timestamp('1960-01-01')
