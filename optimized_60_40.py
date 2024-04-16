
#%%
#
# 60/40 portfolio on optimized MA cross parameters using RealTest software
#
#

import warnings
warnings.filterwarnings("ignore")
def action_with_warnings():
    warnings.warn("should not appear")
with warnings.catch_warnings(record=True):
    action_with_warnings()
import norgatedata
import quantstats        as qs
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt


#%%



#%%


#%%
# Preparing data using Norgate

# Stocks
spy_df = norgatedata.price_timeseries(
    "SPY",
    stock_price_adjustment_setting = norgatedata.StockPriceAdjustmentType.TOTALRETURN,
    padding_setting                = norgatedata.PaddingType.NONE,
    start_date                     = "1990-01-01",
    timeseriesformat               = 'pandas-dataframe',
)

# Bonds
agg_df = norgatedata.price_timeseries(
    "AGG",
    stock_price_adjustment_setting = norgatedata.StockPriceAdjustmentType.TOTALRETURN,
    padding_setting                = norgatedata.PaddingType.NONE,
    start_date                     = "1990-01-01",
    timeseriesformat               = 'pandas-dataframe',
)



#%%


#%%
def macross_backtest(df_, ma_fast, ma_slow):
    df = df_.copy()
    df['FillPrice'] = df['Open' ].shift(-1)
    df['Date'     ] = df.index
    df['DateIn'   ] = df['Date' ].shift(-1)
    df['DateOut'  ] = df['Date' ].shift(-1)
    df.fillna(method='ffill', inplace=True)
    df['EntrySetup'] = 0
    df['ExitRule'  ] = 0
    df['MaFast'] = df['Close'].rolling(ma_fast).mean()
    df['MaSlow'] = df['Close'].rolling(ma_slow).mean()
    df.dropna(inplace=True)
    df.loc[((df['MaFast']>df['MaSlow']) & (df['MaFast'].shift(1)<=df['MaSlow'].shift(1))), 'EntrySetup'] = 1
    df.loc[((df['MaFast']<df['MaSlow']) & (df['MaFast'].shift(1)>=df['MaSlow'].shift(1))), 'ExitRule'  ] = 1
    df.loc[df.index[ 0], 'EntrySetup'] = 1
    df.loc[df.index[-1], 'ExitRule'  ] = 1

    # Position tracking
    in_position      = False
    date_in          = None
    entry_fill_price = 0
    date_out         = None
    exit_fill_price  = 0
    position_history = []
    for index, row in df.iterrows():
        # EntrySetup
        if row['EntrySetup'] == 1:
            date_in          = row['DateIn'   ]
            entry_fill_price = row['FillPrice']
        # ExitRuel
        if row['ExitRule'  ] == 1:
            date_out        = row['DateOut'  ]
            exit_fill_price = row['FillPrice']
            pct_change      = (exit_fill_price - entry_fill_price)/entry_fill_price
            bars            = len(df[date_in:date_out])-1
            position_history.append((
                date_in, 
                date_out, 
                entry_fill_price, 
                exit_fill_price, 
                pct_change,
                bars
                ))

    position_df = pd.DataFrame(position_history, columns=['DateIn', 'DateOut', 'PriceIn', 'PriceOut', 'Return', 'Bars'])
    position_df = position_df.set_index(pd.DatetimeIndex(position_df['DateIn']))

    df['Return'] = df['Close'].pct_change()

    df['StratReturn'] = 0.0
    for index, row in position_df.iterrows():
        sub_df = df[row['DateIn']:row['DateOut']]["Return"]
        df.loc[sub_df.index, 'StratReturn'] = sub_df

    return df['StratReturn']



#%%


#%%
spy_df['StratRet'] = macross_backtest(spy_df, ma_fast=40, ma_slow=260)
spy_df['StratRet'].cumsum().plot()


#%%
agg_df['StratRet'] = macross_backtest(agg_df, ma_fast=30, ma_slow=100)
agg_df['StratRet'].cumsum().plot()


#%%
qs.stats.sharpe(spy_df['StratRet']), qs.stats.sharpe(agg_df['StratRet'])



#%%


#%%
df = pd.DataFrame(index=spy_df.index)
df['SPY_ret'] = spy_df['StratRet']
df['AGG_ret'] = agg_df['StratRet']
df['SPY_cum'] = df['SPY_ret'].cumsum()
df['AGG_cum'] = df['AGG_ret'].cumsum()

df.dropna(inplace=True)

df['SPY_ret'] = df['SPY_cum'].pct_change()
df['SPY_cum'] = df['SPY_ret'].cumsum()

df[['SPY_cum', 'AGG_cum']].plot()

#%%


#%%


#%%


#%%
# 2 BPS for trading cost proxy using Interactive Brokers fee model, very optimistic
rebalancing_cost        = 0.02/100.0
rebalancing_cost_log    = np.log(1-rebalancing_cost)

previous_rebalance_date = None
rebalance_frequency     = 'W-MON' # Weekly rebalance
stock_weight            = 0.6
bond_weight             = 0.4
portfolio_dates         = []
portfolio_returns       = []

for date, group in df.groupby(pd.Grouper(freq=rebalance_frequency)):
    group_df = group.copy()
    group_df['SPY_freq_cum'] = group_df['SPY_ret'].cumsum()
    group_df['AGG_freq_cum'] = group_df['AGG_ret'].cumsum()
    return_at_rebalance = stock_weight * group_df.iloc[-1]['SPY_freq_cum'] + bond_weight * group_df.iloc[-1]['AGG_freq_cum']
    portfolio_dates.append(date)
    portfolio_returns.append(return_at_rebalance)

portfolio_df = pd.DataFrame(index=portfolio_dates)
portfolio_df.index = pd.to_datetime(portfolio_df.index)
portfolio_df['ret'             ] = portfolio_returns
portfolio_df['log_ret'         ] = (1+portfolio_df['ret']).apply(np.log)
portfolio_df['adjusted_log_ret'] = portfolio_df['log_ret'] + rebalancing_cost_log
portfolio_df['cum'             ] = portfolio_df['log_ret'].cumsum().apply(np.exp)
portfolio_df['adjusted_cum'    ] = portfolio_df['adjusted_log_ret'].cumsum().apply(np.exp)
portfolio_df['adjusted_ret'    ] = portfolio_df['adjusted_cum'].pct_change()

portfolio_df



#%%
portfolio_df[['cum', 'adjusted_cum']].plot()
plt.legend(['Rebalanced portfolio', f"Rebalancing cost {round(rebalancing_cost*100,2)}%"])


#%%


#%%
qs.plots.snapshot(portfolio_df['adjusted_ret'], title=f"60/40 portfolio with {round(rebalancing_cost*100,2)}% cost", show=True);


#%%
qs.plots.drawdown(portfolio_df['adjusted_ret'])


#%%
qs.plots.drawdowns_periods(portfolio_df['adjusted_ret'])


#%%
qs.plots.histogram(portfolio_df['adjusted_ret'])


#%%
qs.plots.monthly_heatmap(portfolio_df['adjusted_ret'])


#%%
qs.plots.yearly_returns(portfolio_df['adjusted_ret'])


#%%
qs.stats.sharpe(portfolio_df['adjusted_ret'])


#%%


#%%
plt.plot(df['SPY_cum'])
plt.plot(df['AGG_cum'])
plt.plot(portfolio_df['adjusted_ret'].cumsum())
plt.legend(['SPY MA(40,260) performance', 'AGG MA(30,100) performance', f"60/40 portfolio after {round(rebalancing_cost*100.0, 2)}% cost performance"])
plt.show()


#%%



#%%
qs.reports.full(portfolio_df['adjusted_ret'])


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%


#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%




#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%


#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%




#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%


#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%




#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%


#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



#%%


#%%



