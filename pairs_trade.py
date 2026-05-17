import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# commodity futures -> related equities
pairs = {
    'CL=F': ['XOM', 'CVX', 'COP', 'OXY'],
    'NG=F': ['EQT', 'AR'],
    'GC=F': ['NEM', 'GOLD', 'AEM'],
    'SI=F': ['PAAS', 'AG'],
    'HG=F': ['FCX', 'SCCO'],
    'ZC=F': ['ADM', 'MOS'],
    'LBS=F': ['WY'],
}


class MLPairsTrader:
    def __init__(self, commodity, stock, start, end):
        self.commodity = commodity
        self.stock = stock
        self.start = start
        self.end = end
        self.data = None
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

    def fetch_data(self):
        raw = yf.download(
            [self.commodity, self.stock],
            start=self.start,
            end=self.end,
            progress=False
        )
        price_col = 'Adj Close' if 'Adj Close' in raw.columns else 'Close'
        self.data = raw[price_col].ffill().dropna()  # ffill handles commodity holidays

    def build_features(self):
        df = self.data.copy()
        comm_ret = df[self.commodity].pct_change()
        stock_ret = df[self.stock].pct_change()

        for w in [3, 5, 10, 21]:
            df[f'comm_mom_{w}'] = df[self.commodity] / df[self.commodity].shift(w)
            df[f'stock_mom_{w}'] = df[self.stock] / df[self.stock].shift(w)

        df['corr_10'] = comm_ret.rolling(10).corr(stock_ret)
        df['corr_21'] = comm_ret.rolling(21).corr(stock_ret)
        ma10 = df[self.commodity].rolling(10).mean()
        ma50 = df[self.commodity].rolling(50).mean()
        df['comm_trend'] = np.where(ma10 > ma50, 1, -1)

        self.data = df

    def label(self, horizon=5, min_return=0.005):
        df = self.data.copy()
        fwd = df[self.stock].shift(-horizon) / df[self.stock] - 1
        # 0.5% for fees
        df['target'] = (fwd > min_return).astype(int)
        self.data = df.dropna()

    def _feature_cols(self):
        skip = {self.commodity, self.stock, 'target'}
        return [c for c in self.data.columns if c not in skip]

    def run(self, split_date, confidence=0.58):
        feats = self._feature_cols()

        train = self.data.loc[:split_date]
        test = self.data.loc[split_date:].copy()

        print(f"  train={len(train)}d  test={len(test)}d")

        self.model.fit(train[feats], train['target'])

        test['prob'] = self.model.predict_proba(test[feats])[:, 1]
        test['pred'] = (test['prob'] > confidence).astype(int)

        print(classification_report(test['target'], test['pred'], zero_division=0))

        return self._backtest(test)

    def _backtest(self, test):
        df = test.copy()
        df['ret'] = df[self.stock].pct_change()
        df['position'] = df['pred'].shift(1).fillna(0)  # trade next day's open
        df['strat_ret'] = df['position'] * df['ret']

        df['cum_strat'] = (1 + df['strat_ret'].fillna(0)).cumprod() - 1
        df['cum_bh'] = (1 + df['ret'].fillna(0)).cumprod() - 1

        sharpe = (
            df['strat_ret'].mean() / df['strat_ret'].std() * np.sqrt(252)
            if df['strat_ret'].std() > 0 else 0
        )
        max_dd = (
            df['cum_strat'] - df['cum_strat'].cummax()
        ).min()

        print(f"  days in market : {int(df['position'].sum())}")
        print(f"  strategy return: {df['cum_strat'].iloc[-1]:.2%}")
        print(f"  buy & hold     : {df['cum_bh'].iloc[-1]:.2%}")
        print(f"  sharpe         : {sharpe:.2f}")
        print(f"  max drawdown   : {max_dd:.2%}")

        return df


def main():
    start, end, split = '2016-01-01', '2023-01-01', '2021-01-01'

    # iterate through every commodity\stock relationship
    for commodity, stocks in pairs.items():
        for stock in stocks:
            print(f"{commodity} → {stock}")
            trader = MLPairsTrader(commodity, stock, start, end)
            trader.fetch_data()
            trader.build_features()
            trader.label(horizon=3)
            trader.run(split_date=split)


if __name__ == '__main__':
    main()