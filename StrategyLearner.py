"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import random
from indicators_fun import indicators
import pdb
import QLearner as Ql
from marketsimcode import *
import tensorflow as tf
from tensorflow import keras
from indicators_fun import indicators
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# Helper libraries



#%%

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.005,commission=9.95):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
 
            
    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 1000000): 
        # add your code to do learning here
        (normalized_values ,bbp,moving_avarage,rsi_val,rsi_spy,momentum,sma_cross) = indicators(sd=sd,ed=ed,syms=[symbol],allocs=[1],sv=sv,gen_plot=False)
        
        norm_val          = normalized_values.copy()
        
        normalized_values = pd.DataFrame(data=pd.qcut(normalized_values,10,labels=False),index=normalized_values.index)
        bbp               = pd.DataFrame(data=pd.qcut(bbp,10,labels=False),index=bbp.index)
        moving_avarage    = pd.DataFrame(data=pd.qcut(moving_avarage,10,labels=False),index=moving_avarage.index)
        rsi_val           = pd.DataFrame(data=pd.qcut(rsi_val,10,labels=False),index=rsi_val.index)
        rsi_spy           = pd.DataFrame(data=pd.qcut(rsi_spy,10,labels=False),index=rsi_spy.index)
        momentum          = pd.DataFrame(data=pd.qcut(momentum,10,labels=False),index=momentum.index)
        sma_cross         = pd.DataFrame(data=pd.cut(sma_cross,3,labels=False),index=sma_cross.index) # REVISAR ESTE , CREO QUE ESTA MAL
        #start             = pd.DataFrame(data=[[0] + [1]*(len(sma_cross)-1)][0],index=normalized_values.index)
        
        #states            = pd.concat([normalized_values,bbp,moving_avarage,rsi_val,rsi_spy,momentum,sma_cross],axis=1).apply(lambda x : x.fillna(0).astype(int).astype(str).str.cat(),axis=1).to_frame().iloc[13:,:]

        states            = pd.concat([bbp,moving_avarage,rsi_val,rsi_spy,momentum,sma_cross],axis=1).apply(lambda x : x.fillna(0).astype(int).astype(str).str.cat(),axis=1).to_frame().iloc[13:,:]
        
        
        robot             = Ql.QLearner(num_states = (10**6)+(10**5)+(10**4)+(10**3)+(10**2)+3 , num_actions=5) # PENDIENTE MEJORAR PARA QUE SEA DEL TAMANIO DE LAS BINS FILA 29 a la 35
        
        max_iter          = 5000
        actions_df        = pd.DataFrame(index=states.index,data=[0]*len(states))
        iter_num          = 0
        converged         = False
        dates = pd.date_range(sd,ed)
        prices_all = ut.get_data([symbol], dates)[symbol] 
        
        while not converged and iter_num<max_iter :
            #pdb.set_trace()
            
            holdings_actions = 0
            syms=[symbol]
            X = int(states.iloc[0])
            action = 2 #robot.querysetstate(int(states.iloc[0]))
            
            
            holdings_actions_1        = 0 
            
            
            
            for key,row in states.iterrows():
            
                #pdb.set_trace()
                    
                    #change = ((norm_val.loc[key]/norm_val.iloc[norm_val.index.get_loc(key)-1])-1).values[0]
                    
                    #holdings.append(holdings[-1] + holdings_actions*prices_all.loc[key])
                    
                    #reward = ((holdings[-1]/(holdings[-2]+holdings_actions*prices_all[prices_all.index.get_loc(key)-1])-1)*1000)
                    
                    
                holdings_actions              = {0:1000,1:500,2:0,3:-500,4:-1000}[action]
                if(key==states.index[0]):
                    holdings_diff             = 0
                    
                else:
                    
                    price_t                   = prices_all.iloc[prices_all.index.get_loc(key)-1]
                    price_t_plus_1            = prices_all.loc[key]
                    cash                      = -1*(holdings_actions-holdings_actions_1) * price_t
                    holdings_diff             = holdings_actions * price_t_plus_1 - holdings_actions_1 * price_t + cash
                    
                    
                    
                if(holdings_actions_1-holdings_actions)!=0:
                    holdings_diff     = holdings_diff - self.commission
                       
                reward = holdings_diff
                    
                X = int(states.loc[key])
                
                holdings_actions_1 = holdings_actions
                
                action = robot.query(X,reward)
                
                actions_df.loc[key,iter_num] = holdings_actions
                
                    
                        
                        
                        #holdings_actions[1] = holdings_actions[1]*(1-self.impact)
                        
                        
                        
            iter_num += 1
        #Check convergence    
            converged=False    
        pdb.set_trace()
        previous_days = 13
        trades = pd.DataFrame(data = actions_df.iloc[:,-1],index = actions_df.index).diff().shift(-1).fillna(0)
        #trades = pd.concat([pd.DataFrame(data=[[0]]*previous_days,index=normalized_values.index[0:previous_days],columns=trades.columns.tolist()),trades])
        pdb.set_trace()
        #trades = pd.concat([pd.DataFrame(data=actions_df.iloc[0,-1],index=normalized_values.index[0],columns=trades.columns.tolist()),trades])
        #trades = trades.append(pd.DataFrame(data=[actions_df.iloc[:,-1][0]],columns=trades.columns.tolist(),index=[trades.index[0]]))
        trades.sort_index(axis =0 , inplace=True)
        trades.iloc[-1] = -1*trades.iloc[:-1].sum()
        
        trades.columns = ['Shares']
    
        
        
        trades['Symbol'] = symbol
        trades['Order'] = trades['Shares'].to_frame().applymap(lambda x : {-2000:'SELL',-1500:'SELL',-1000:'SELL',-500:'SELL',0:0,500:"BUY",1000:"BUY",1500:"BUY",2000:"BUY"}[x])
        trades['Shares'] = trades['Shares'].abs()
        trades['Date'] = trades.index
        self.robot = robot
        
        
        
        ## example usage of the old backward compatible util function
        #syms=[symbol]
        #dates = pd.date_range(sd,ed)
        #prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        #prices = prices_all[syms]  # only portfolio symbols
        #prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        #if self.verbose: print prices
        #
        ## example use with new colname 
        #volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        #volume = volume_all[syms]  # only portfolio symbols
        #volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        #if self.verbose: print volume
        
    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        
        # here we build a fake set of trades
        # your code should return the same sort of data
        #dates = pd.date_range(sd, ed)
        #prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        #trades = prices_all[[symbol,]]  # only portfolio symbols
        #trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        #trades.values[:,:] = 0 # set them all to nothing
        #trades.values[0,:] = 1000 # add a BUY at the start
        #trades.values[40,:] = -1000 # add a SELL 
        #trades.values[41,:] = 1000 # add a BUY 
        #trades.values[60,:] = -2000 # go short from long
        #trades.values[61,:] = 2000 # go long from short
        #trades.values[-1,:] = -1000 #exit on the last day
        #if self.verbose: print type(trades) # it better be a DataFrame!
        #if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

if __name__=="__main__":
    
    nuevo = StrategyLearner()
    nuevo.addEvidence()
    print "One does not simply think up a strategy"
