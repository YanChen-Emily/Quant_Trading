#Operating on BigQuant Platform

import numpy as np
import pandas as pd 
import talib

start_date = '2019-01-03'
end_date = '2021-01-22'
stock_list = pd.read_csv('stock_list.csv', header=None)
instruments = stock_list.values[:,0].tolist()

#initial capitalabs
capital_base = 10000

#benchmark '沪深300'
benchmark = '000300.INDX'

#account rebalance period
rebalance_period = 20

#number of stocks held
stock_count = 10


#Strategy:

#crossing point of MACD:
#DIF short = 12; DIF long = 26; DEA=9; DIF cross above DEA (long); otherwise short

#MA long:
#MA short = 5; MA long = 20; MA short crosses above MA long (long); otherwise short


def initialize(context):
    #transaction fee
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0003, min_cost=5))
    #short moving average (DIF)
    context.short_period = 12 #短期均线
    #long moving average (DIF)
    context.long_period=26 #长期均线
    #DEA
    context.smooth_period = 9
    
    #observation length for historical data
    context.observation=50
    
    context.ma_short_period = 5 #MA short
    context.ma_long_period = 20 #MA long
    
    #context.hold_days = 20 #transfer position period
    #context.trade_day = context.observation + context.observation % context.hold_days
    
    #context.trade_day = context.observation + context.hold_days
    #context.stock_count = 10 #maximum number of stocks purchased
    context.stock_weights = T.norm([1/stock_count for i in range(0,stock_count)]) #equal weights of capital on stocks

    
def handle_data(context,data):
    if context.trading_day_index < context.observation: 
        return
    
    #today is the first transaction day
    today = data.current_dt.strftime('%Y-%m-%d')
    
    #cash = context.portfolio.cash
    
    # context.trading_day_index：the index of trading date, the first is 0
    # will only trade every 20 days (hold_day)
    if context.trading_day_index % rebalance_period != 0:
        return
     
    equities = context.portfolio.positions
    
    #sell all the stocks in the portfolio
    for equity in equities:
        if data.can_trade(equity):
            context.order_target(equity, 0)
            
    #buy the top 10 stocks
    stock_candidates = []
    for i, instrument in enumerate(instruments):
        sid = context.symbol(instruments[i]) #stock index
        price = data.current(sid, 'price') # current price
        # historical daily prices
        prices = data.history(sid, 'price', context.observation, '1d')
        if np.isnan(price):
            continue
        # 用Talib计算MACD取值，得到三个时间序列数组，分别为macd, signal 和 hist
        macd, signal, hist = talib.MACD(np.array(prices), context.short_period, context.long_period, context.smooth_period)

        # long term and short term MA
        short_mavg = data.history(sid, 'price',context.ma_short_period, '1d').mean() # 短期均线值
        long_mavg = data.history(sid, 'price',context.ma_long_period, '1d').mean() # 长期均线值

        # 计算现在portfolio中股票的仓位
        cur_position = context.portfolio.positions[sid].amount

        # sell all the stocks every hold_day (20) days
        if cur_position > 0 and data.can_trade(sid):
            context.order_target_value(sid, 0)

        # buy (cross above)
        if macd[-1] - signal[-1] > 0 and macd[-2] - signal[-2] < 0:
            # 如果短期均线大于长期均线形成金叉，并且MA 短线在 MA 长线上方时，并且该股票可以交易
            if short_mavg > long_mavg:
                if data.can_trade(sid): #the stock is tradable
                    stock_candidates.append((sid, price))

        stocks = stock_candidates[: stock_count]
        stocks_len =  len(stocks)
        if stocks_len > 0:
            context.stock_weights = T.norm([1 / stocks_len for i in range(0, stocks_len)])
            stock_weights = context.stock_weights
            sid_prices = list(stocks)
            for i, instrument in enumerate(sid_prices):
                cash = capital_base * stock_weights[i]
                if cash > 0:
                    order_unit =  int(cash/instrument[1]/100)*100
                    context.order(instrument[0], order_unit)
                    
        else:
            pass    
    
    
m_macd = M.trade.v2( 
    instruments=instruments,
    start_date=start_date,
    end_date=end_date,
    initialize=initialize,
    handle_data=handle_data,
    order_price_field_buy='open',
    order_price_field_sell='open',
    capital_base=capital_base,
    benchmark=benchmark
)    
    
    
    
