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

#max weight of each stock
context.max_weight_per_stock = 0.2

#max weight of each sector
context.max_weight_per_sector = 0.5

#Number of stocks in the portfolio
N = len(instruments)

#optimization function




def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # objective function
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    
    #constraints
    G_1 = -np.eye(n)
    G_2 = np.eye(n)
    G_3 = np.zeros((3,n))
    G_3[0, :10] = 1
    G_3[1, 10:20] = 1
    G_3[2, 20:] = 1
    G_np = np.concatenate((G_1, G_2, G_3), axis=0) 
    #print(G.shape)
    G = opt.matrix(G_np)
    
    h_1 = np.zeros((n,1)) 
    h_2 = np.ones((n,1)) * context.max_weight_per_stock # 0.2
    h_3 = np.ones((3,1)) * context.max_weight_per_sector # 0.5
    h_np = np.concatenate((h_1, h_2, h_3), axis=0) 
    #print(h_np.shape)
    h = opt.matrix(h_np)

    
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # 使用凸优化计算有效前沿
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## 计算有效前沿的收益率和风险
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
    # rebalance every 10 days
    if context.days % 10 != 0:
        return
    # window size=5, calculate asset return per stock
    prices = data.history(context.symbols(context.ins[0],context.ins[1],context.ins[2],context.ins[3],context.ins[4],context.ins[5],context.ins[6],context.ins[7],context.ins[8],context.ins[9],context.ins[10],context.ins[11],context.ins[12],context.ins[13],context.ins[14],context.ins[15],context.ins[16],context.ins[17],context.ins[18],context.ins[19]), 'price',5, '1d').dropna()
    returns = prices.pct_change().dropna()
    try:
        # Markowitz portfolio optimization
        weights, _, _ = optimal_portfolio(returns.T)
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
    
        
    