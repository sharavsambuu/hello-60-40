#%%
#
#  60/40 portfolio on raw proxies
#  SPY for the stocks
#  AGG for the bonds
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

df = pd.DataFrame(index=spy_df.index)
df['SPY'] = spy_df['Close']

# Bonds
agg_df = norgatedata.price_timeseries(
    "AGG",
    stock_price_adjustment_setting = norgatedata.StockPriceAdjustmentType.TOTALRETURN,
    padding_setting                = norgatedata.PaddingType.NONE,
    start_date                     = "1990-01-01",
    timeseriesformat               = 'pandas-dataframe',
)

df['AGG'] = agg_df['Close']

df



#%%
df.dropna(inplace=True)

df

#%%


#%%
df[['SPY', 'AGG']].plot()


#%%
df['SPY_ret'] = df['SPY'    ].pct_change()
df['AGG_ret'] = df['AGG'    ].pct_change()
df['SPY_cum'] = df['SPY_ret'].cumsum()
df['AGG_cum'] = df['AGG_ret'].cumsum()

df[['SPY_cum', 'AGG_cum']].plot()


#%%


#%%


#%%
# 2 BPS for trading cost proxy using Interactive Brokers fee model, very optimistic
rebalancing_cost        = 0.02/100.0
rebalancing_cost_log    = np.log(1-rebalancing_cost)

previous_rebalance_date = None
rebalance_frequency     = '3W' # rebalancing frequency
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

plt.plot(df['SPY_cum'])
plt.plot(df['AGG_cum'])
plt.plot(portfolio_df['adjusted_ret'].cumsum())
plt.legend(['SPY raw performance', 'AGG raw performance', f"60/40 portfolio after {round(rebalancing_cost*100.0, 2)}% cost performance, freq {rebalance_frequency}"])
plt.show()



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


#%%



