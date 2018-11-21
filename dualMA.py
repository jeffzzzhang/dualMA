# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:37:37 2017
双均线策略, double MA
reference https://www.joinquant.com/post/1398?f=study&m=algorithm
@author: Jeff
"""
import pandas as pd
import numpy as np
import time
import os
import pickle
import math

start_time = time.time()
def MovingAverage(number,closing_price):
    MA_n = np.array(closing_price)
    for stock_id in range(len(closing_price.columns)):
        this_stock_data = closing_price[closing_price.columns[stock_id]]
        for i in range(len(this_stock_data.index)):
            if type(this_stock_data[i]) == str:
                if this_stock_data[i] != '--':
                    continue
                else:
                    MA_n[i,stock_id] = '--'
                    continue
            else:
                if i < 2 + (number-1): # assume the head 2 lines are stock code and name
                    MA_n[i,stock_id] = '--'
                else:
                    tmp = list(this_stock_data[i-number+1:i+1])
                    tmp_cntr = 0
                    for j in range(number):
                        if type(tmp[j]) == str:
                            tmp_cntr = 1
                            break
                    if tmp_cntr == 1:
                        MA_n[i,stock_id] = '--'
                        continue
                    MA_n[i,stock_id] = sum(tmp)/number # this_stock_data[i-number:i].mean()
    MA = pd.DataFrame(MA_n,index=closing_price.index,columns=closing_price.columns)
    return MA

def ma_cross(ma_small,ma_big,stock_list):
    '''
    是否长线MA大于短线MA的时间序列
    '''    
    compare_array = np.array(ma_big[stock_list])
    starting_index = 0
    for i in range(len(ma_big.index)):
        if type(ma_big.iloc[i,0]) == str:
            continue
        else:
            starting_index = i
            break
    for stock in range(len(stock_list)):
        tmp_compare_bit = []
        tmp_compare_bit = list(np.sign(ma_big.ix[starting_index:,stock_list[stock]]-ma_small.ix[starting_index:,stock_list[stock]]))
        compare_array[starting_index:,stock] = tmp_compare_bit
        
    return pd.DataFrame(compare_array,index=ma_big.index,columns=stock_list)
    
def buy_sell_based_on_ma_cross(ma_cross_results):
    '''
    根据长短期MA的交叉决定是否买进
    '''
    buy_sell_action = np.array(ma_cross_results)
    stock_list = ma_cross_results.columns
    for stock in range(len(stock_list)):
        previous_bit = None
        for i in range(len(ma_cross_results.index)):
            if type(ma_cross_results.ix[i,stock]) == str: #停牌不清仓
                continue
            
            if previous_bit == None:
                buy_sell_action[i,stock] = '--'
                previous_bit = ma_cross_results.ix[i,stock]
                continue
            
            if previous_bit == ma_cross_results.ix[i,stock]:
                if buy_sell_action[i-1,stock] == 'buy' or buy_sell_action[i-1,stock] == 'hold':
                    buy_sell_action[i,stock] = 'hold'
                else:
                    buy_sell_action[i,stock] = '--'
            elif previous_bit - ma_cross_results.ix[i,stock] == 2: # golden cross
                buy_sell_action[i,stock] = 'buy'
                previous_bit = ma_cross_results.ix[i,stock]
            elif previous_bit - ma_cross_results.ix[i,stock] == -2: # dead cross
                buy_sell_action[i,stock] = 'sell'
                previous_bit = ma_cross_results.ix[i,stock]
            elif previous_bit - ma_cross_results.ix[i,stock] == 1 and ma_cross_results.ix[i,stock] == -1:
                j = 2
                while j<i:
                    if ma_cross_results.ix[i-j,stock] - ma_cross_results.ix[i,stock] == 2:
                        buy_sell_action[i,stock] = 'buy'
                        break
                    elif ma_cross_results.ix[i-j,stock] - ma_cross_results.ix[i,stock] == 2:
                        continue
                    else: # ma_cross_results.ix[i-j,stock] 和 ma_cross_results.ix[i,stock] 一样都是-1
                        if buy_sell_action[i-j,stock] == 'buy' or buy_sell_action[i-j,stock] == 'hold':
                            buy_sell_action[i,stock] = 'hold'
                            break
                        else:
                            buy_sell_action[i,stock] = '--'
                            break
                    j += 1
                previous_bit = ma_cross_results.ix[i,stock]
            elif previous_bit - ma_cross_results.ix[i,stock] == 1 and ma_cross_results.ix[i,stock] == 0:
                if buy_sell_action[i-1,stock] == 'buy' or buy_sell_action[i-1,stock] == 'hold':
                    buy_sell_action[i,stock] = 'hold'
                else:
                    buy_sell_action[i,stock] = '--'
                # 这时候因为ma_cross_results.ix[i,stock]情况特殊，所以previous_bit不改变
            elif previous_bit - ma_cross_results.ix[i,stock] == -1 and ma_cross_results.ix[i,stock] == 1:
                j = 2 
                while j<i:
                    if ma_cross_results.ix[i-j,stock] - ma_cross_results.ix[i,stock] == -2:
                        buy_sell_action[i,stock] = 'sell'
                        break
                    elif ma_cross_results.ix[i-j,stock] - ma_cross_results.ix[i,stock] == 0:
                        continue
                    else: # 都是1
                        if buy_sell_action[i-j,stock] == 'buy' or buy_sell_action[i-j,stock] == 'hold':
                            buy_sell_action[i,stock] = 'hold'
                            break
                        else:
                            buy_sell_action[i,stock] = '--'
                            break
                    j += 1
                previous_bit = ma_cross_results.ix[i,stock]
            elif previous_bit - ma_cross_results.ix[i,stock] == -1 and ma_cross_results.ix[i,stock] == 0:
                if buy_sell_action[i-1,stock] == 
                
    return pd.DataFrame(buy_sell_action,index=ma_cross_results.index,columns=ma_cross_results.columns)
    
def gain_and_loss(buy_sell_action,opening_price):
    # 根据buy_sell_based_on_ma_cross的结果和开盘价来计算损失和收益
    position_status = np.array(buy_sell_action)
    stock_list = buy_sell_action.columns
    money_allocation = 100000
    remaining_cap = np.array(buy_sell_action)
    starting_index = None
    for i in range(len(buy_sell_action.index)):
        if type(buy_sell_action.index[i]) != str:
            starting_index = i
            break
    remaining_cap[starting_index:,:] = np.zeros((len(buy_sell_action.index)-starting_index,len(buy_sell_action.columns))) + money_allocation
    remaining_cap = pd.DataFrame(remaining_cap,index=buy_sell_action.index,columns=buy_sell_action.columns)
    
    for stock in range(len(stock_list)):
        hands = None
        buying_cost = None
        for i in range(2,len(buy_sell_action.index)):
            if buy_sell_action.iloc[i,stock] == 'buy':
                if buy_sell_action.iloc[i-1,stock] == '--':
                    position_status[i,stock] = 0
                    hands = None
                    buying_cost = None
                elif buy_sell_action.iloc[i-1,stock] == 'sell':
                    remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] + hands * opening_price.ix[i,stock_list[stock]] * 100
                    position_status[i,stock] = 0
                    hands = None
                    buying_cost = None
            elif buy_sell_action.iloc[i,stock] == 'hold':
                if buy_sell_action.iloc[i-1,stock] == 'buy':
                    hands = math.trunc(remaining_cap.iloc[i,stock]/(opening_price.ix[i,stock_list[stock]]*100))
                    position_status[i,stock] = hands * opening_price.ix[i,stock_list[stock]] *100
                    buying_cost = hands * opening_price.ix[i,stock_list[stock]] *100
                    remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] - buying_cost
                else: # hold yesterday, hold today
                    position_status[i,stock] = hands * opening_price.iloc[i,stock] * 100
                    # remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] - buying_cost
            elif buy_sell_action.iloc[i,stock] == 'sell':
                if buy_sell_action.iloc[i-1,stock] == 'buy':
                    hands = math.trunc(remaining_cap.iloc[i,stock]/(opening_price.ix[i,stock_list[stock]]*100))
                    position_status[i,stock] = hands * opening_price.ix[i,stock_list[stock]] *100
                    buying_cost = hands * opening_price.ix[i,stock_list[stock]] *100
                    remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] - buying_cost
                elif buy_sell_action.iloc[i-1,stock] == 'hold':
                    position_status[i,stock] = hands * opening_price.ix[i,stock_list[stock]] * 100
                else:
                    print('check here')
            elif buy_sell_action.iloc[i,stock] == '--':
                if buy_sell_action.iloc[i-1,stock] == 'sell':
                    remaining_cap.iloc[i:,stock] = remaining_cap.iloc[i:,stock] + hands * opening_price.ix[i,stock_list[stock]] * 100
                    position_status[i,stock] = 0
                    hands = None
                    buying_cost = None
                elif buy_sell_action.iloc[i-1,stock] == '--':
                    position_status[i,stock] = 0
                    hands = None
                    buying_cost = None
                elif buy_sell_action.iloc[i-1,stock] != '--' and type(buy_sell_action.iloc[i-1,stock]) == str:
                    position_status[i,stock] = 0
                    hands = None 
                    buying_cost = None
    return pd.DataFrame(position_status,index=buy_sell_action.index,columns=buy_sell_action.columns),remaining_cap #两个array在同一天的值相加即为总资产

if __name__ == '__main__':
    os.chdir('C:\\Users\\talen_000\\Desktop\\dualMAstrategy')
    closing_price = pd.read_excel('个股日收盘价2016till20170606.xlsx')
    opening_price = pd.read_excel('个股日开盘价2016till20170606.xlsx')
    # ma20 = MovingAverage(20,closing_price)
    # ma60 = MovingAverage(60,closing_price)
    # ma5 = MovingAverage(5,closing_price)
    sector_and_stocks = pickle.load(open('C:/Users/talen_000/Desktop/行业和个股/sectorandstocks.txt','rb'))
    ma5 = pickle.load(open('C:/Users/talen_000/Desktop/dualMAstrategy/MA5_2016till20170606.txt','rb'))
    ma20 = pickle.load(open('C:/Users/talen_000/Desktop/dualMAstrategy/MA20_2016till20170606.txt','rb'))
    ma60 = pickle.load(open('C:/Users/talen_000/Desktop/dualMAstrategy/MA60_2016till20170606.txt','rb'))

    print('time collapsed: ',time.time()-start_time,' seconds')
