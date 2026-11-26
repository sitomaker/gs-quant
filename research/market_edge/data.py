"""Download and cache the market data used by the study.

Sources:
  * Kenneth French Data Library: daily US market (CRSP value-weighted, total return),
    risk-free rate, Fama-French factors, momentum and 30 industry portfolios.
  * Yahoo Finance chart API: daily OHLC with dividend/split-adjusted close.

Everything is cached as CSV under ``data/`` so repeated runs are offline and reproducible.
"""

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / 'data'
KF_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip'
YAHOO_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
HEADERS = {'User-Agent': 'Mozilla/5.0 (research script)'}


def _get(url: str, params: dict = None, retries: int = 4) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _parse_kf_csv(text: str) -> pd.DataFrame:
    """Parse the first data block of a Kenneth French CSV (daily, YYYYMMDD index), in decimals."""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        first = line.split(',')[0].strip()
        if first.isdigit() and len(first) == 8:
            header_idx = i - 1
            break
    if header_idx is None:
        raise ValueError('No daily data block found')
    columns = [c.strip() for c in lines[header_idx].split(',')[1:]]
    rows = []
    for line in lines[header_idx + 1:]:
        parts = [p.strip() for p in line.split(',')]
        if not (parts[0].isdigit() and len(parts[0]) == 8):
            break
        rows.append(parts)
    df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows], format='%Y%m%d'),
                      columns=columns).astype(float)
    df = df.mask(df <= -99.99)  # missing-value sentinel
    return df / 100.0


def kenneth_french(name: str, refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f'kf_{name}.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    resp = _get(KF_URL.format(name=name))
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        text = zf.read(zf.namelist()[0]).decode('latin-1')
    df = _parse_kf_csv(text)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def yahoo(symbol: str, refresh: bool = False) -> pd.DataFrame:
    """Daily OHLC; ``adj_open``/``adj_close`` include dividends and splits (total return)."""
    path = DATA_DIR / f'yf_{symbol.replace("^", "IDX_")}.csv'
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    params = {'period1': -2208988800, 'period2': int(time.time()), 'interval': '1d',
              'events': 'div,split', 'includeAdjustedClose': 'true'}
    result = _get(YAHOO_URL.format(symbol=symbol), params=params).json()['chart']['result'][0]
    quote = result['indicators']['quote'][0]
    idx = pd.to_datetime(result['timestamp'], unit='s', utc=True).tz_convert(
        result['meta']['exchangeTimezoneName']).tz_localize(None).normalize()
    df = pd.DataFrame({k: quote[k] for k in ('open', 'high', 'low', 'close', 'volume')}, index=idx)
    adj = result['indicators'].get('adjclose', [{}])[0].get('adjclose')
    df['adj_close'] = adj if adj is not None else df['close']
    df = df[~df.index.duplicated(keep='last')].dropna(subset=['close'])
    factor = df['adj_close'] / df['close']
    df['adj_open'] = df['open'] * factor
    df['adj_high'] = df['high'] * factor
    df['adj_low'] = df['low'] * factor
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def us_market() -> pd.DataFrame:
    """Daily US market total return, RF and factors (decimals)."""
    ff = kenneth_french('F-F_Research_Data_Factors_daily')
    mom = kenneth_french('F-F_Momentum_Factor_daily')
    df = ff.join(mom, how='left')
    df.columns = [c.replace(' ', '') for c in df.columns]
    df = df.rename(columns={'Mom': 'MOM'})
    df['MKT'] = df['Mkt-RF'] + df['RF']
    return df


def industries(n: int = 30) -> pd.DataFrame:
    return kenneth_french(f'{n}_Industry_Portfolios_daily')


ETFS = ['SPY', 'EFA', 'AGG']
ROBUSTNESS = ['^GSPC', '^N225', '^FTSE', '^GDAXI', '^HSI', '^IBEX', '^STOXX50E', 'BTC-USD']


def download_all(refresh: bool = False) -> None:
    us_market()
    industries(30)
    for sym in ETFS + ROBUSTNESS:
        yahoo(sym, refresh=refresh)


if __name__ == '__main__':
    download_all(refresh=True)
    m = us_market()
    print('US market', m.index.min().date(), m.index.max().date(), len(m))
    ind = industries(30)
    print('Industries', ind.shape, ind.index.min().date(), ind.index.max().date())
    for sym in ETFS + ROBUSTNESS:
        d = yahoo(sym)
        print(sym, d.index.min().date(), d.index.max().date(), len(d))
