import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers

start_date = '2019-01-03'
end_date = '2021-01-22'
stock_list = pd.read_csv('stock_list.csv', header=None)
instruments = stock_list.values[:,0].tolist()

#initial capitalabs
capital_base = 10000

#benchmark '沪深300'
benchmark = '000300.INDX'

#Number of stocks in the portfolio
N = len(instruments)

data = D.history_data(instruments,start_date,end_date,
                     fields=['open'])
# visualize the data
data = pd.pivot_table(data,values='open',index=['date'],columns=['instrument'])
T.plot(data)

#optimization function

#reference: https://bigquant.com/wiki/doc/cvxopt-3j4UqJw4us


def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # objective function
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    
    #constraints
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n,1))
    A = opt.matrix(1.0, (1,n))
    b = opt.matrix(1.0)
    
    #solve for optimal x
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    
    ## efficient margin and risk
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # 计算最优组合
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks



    def initialize(context):
    
    context.days = 0
    context.ins = instruments
     
def handle_data(context, data):
    
    context.days += 1
    if context.days < 5:
        return
    # rebalance every 20 days
    if context.days % 20 != 0:
        return
    # window size=5, calculate asset return per stock
    prices = data.history(context.symbols(context.ins[0],context.ins[1],context.ins[2],context.ins[3],context.ins[4],context.ins[5],context.ins[6],context.ins[7],context.ins[8],context.ins[9],context.ins[10]), 'price',5, '1d').dropna()
    returns = prices.pct_change().dropna()
    try:
        # Markowitz portfolio optimization
        weights, _, _ = optimal_portfolio(returns.T)
#         print(weights)
        # adjusting weights for each stock
        for stock, weight in zip(prices.columns, weights):
            if data.can_trade(stock):
                order_target_percent(stock, weight[0])
    except ValueError as e:
        pass   

m=M.trade.v2( 
    instruments=instruments,
    start_date=start_date,  
    end_date=end_date,
    initialize=initialize,
    handle_data=handle_data,
    order_price_field_buy='open',
    order_price_field_sell='open',
    capital_base=10000,
    benchmark='000300.INDX',
)
    
        
    