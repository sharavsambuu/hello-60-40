#%%
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


pd.options.display.float_format = '{:.4}'.format


#%%



#%%
start_date = "2000-01-01"

assets = ['SPY', 'AGG']

def download_history(symbol, start_date):
    return norgatedata.price_timeseries(
        symbol,
        stock_price_adjustment_setting = norgatedata.StockPriceAdjustmentType.TOTALRETURN,
        padding_setting                = norgatedata.PaddingType.NONE,
        start_date                     = start_date,
        timeseriesformat               = 'pandas-dataframe',
        )


prices_df = pd.DataFrame()
for symbol in assets:
    prices_df[symbol] = download_history(symbol, start_date)['Close']
prices_df.dropna(inplace=True)

returns_df = pd.DataFrame(index=prices_df.index)
for symbol in assets:
    returns_df[symbol] = prices_df[symbol].pct_change()
returns_df.dropna(inplace=True)

prices_df


#%%
returns_df


#%%
returns_df.cumsum().plot(legend=False)
plt.show();


#%%
prices_df.plot(legend=False)
plt.show();


#%%


#%%
portfolio_value      = 10000
rebalance_frequency  = "4W"
optimization_months  = 12
cost_ib_per_shares   = 0.005  # USD per share https://www.interactivebrokers.com/en/pricing/commissions-home.php
cost_slippage_spread = 0.002  # Other cost
debug                = False

volatility_window    = 40

optimization_window  = pd.Timedelta(days=optimization_months*30) # approximately many months
dataset_start        = prices_df.index[0]
optimization_dates   = []
portfolio_values     = []
rebalancing_costs    = []

for group_start_date, group_data in prices_df.groupby(pd.Grouper(freq=rebalance_frequency)):

    if (group_start_date-dataset_start)<optimization_window:
        continue

    # Forecast section
    optimization_prices_df  = prices_df[group_start_date-optimization_window:group_start_date].copy()
    optimization_returns_df = optimization_prices_df.copy().pct_change()
    optimization_returns_df.dropna(inplace=True)

    inverse_volatility_df = 1.0/(optimization_returns_df.rolling(volatility_window).std() * np.sqrt(252))

    if debug:
        print(f"[{group_start_date-optimization_window} - {group_start_date}]", end='  ')
    
    # Number of shares calculation
    w = pd.DataFrame(index=['SPY', 'AGG'])
    w['weight'           ] = [0.6, 0.4]
    w['weight'           ] = w['weight'] * inverse_volatility_df.iloc[-1]
    w['weight'           ] = w['weight'] / w['weight'].sum()

    w['investment_amount'] = w['weight'] * portfolio_value
    w['last_price'       ] = optimization_prices_df.iloc[-1]
    w['shares'           ] = round((w.investment_amount/w.last_price), 6)

    # Interactive Brokers dollar cost approximation MAX(1, 0.005*shares)
    w['shares_value'     ] = w['shares'] * w['last_price']
    w['ib_cost_value'    ] = w['shares'       ] * cost_ib_per_shares
    w['ib_cost_value'    ] = w['ib_cost_value'].apply(lambda x: max(1.0, x))
    total_ib_cost = w['ib_cost_value'].sum()

    if debug:
        print(f"pre ${round(w['shares_value'].sum(),2)}, cost ${round(total_ib_cost, 2)}", end=' -> ')


    # Evaluation section
    group_prices_df    = group_data.copy()
    group_returns_df   = group_prices_df.copy().pct_change()
    group_returns_df.dropna(inplace=True)

    if len(group_returns_df)<=0:
        continue

    group_equity_df    = group_returns_df.cumsum()
    group_returns_df   = group_equity_df - cost_slippage_spread

    group_end_returns  = group_returns_df.iloc[-1]
    group_values       = w['shares_value'] + w['shares_value']*group_end_returns

    portfolio_value    = group_values.sum() - total_ib_cost

    if debug:
        print(f"value ${round(portfolio_value,2)}")

    optimization_dates.append(group_start_date)
    portfolio_values.append(portfolio_value)
    rebalancing_costs.append(total_ib_cost)

    pass


portfolio_df = pd.DataFrame(index=optimization_dates)
portfolio_df['value'   ] = portfolio_values
portfolio_df['cost'    ] = rebalancing_costs
portfolio_df['cost_cum'] = portfolio_df['cost'].cumsum()

fig, axes = plt.subplots(nrows=2, height_ratios=[3, 1])
axes[0].set_title(f"Rebalance {rebalance_frequency}, Opt. months {optimization_months}, slippage and spread {cost_slippage_spread}, IB cost per share {cost_ib_per_shares}")
portfolio_df['value'   ].plot(ax=axes[0])
axes[1].set_title("Trading cost")
portfolio_df['cost_cum'].plot(ax=axes[1])
plt.show();


#%%


#%%
qs.reports.full(portfolio_df['value'].pct_change())


#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%%




