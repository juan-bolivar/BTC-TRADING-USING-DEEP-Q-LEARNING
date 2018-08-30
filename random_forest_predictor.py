from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import pandas as pd
import matplotlib.pyplot as plt

import datetime as dt
import pandas as pd
import util as ut
import random
from indicators_fun import indicators
import pdb
import QLearner as Ql
from marketsimcode import *


sd=dt.datetime(2008,1,1)
ed=dt.datetime(2009,1,1)
symbol = 'JPM'
sv = 1000000

(normalized_values ,bbp,moving_avarage,rsi_val,rsi_spy,momentum,sma_cross) = indicators(sd=sd,ed=ed,syms=[symbol],allocs=[1],sv=sv,gen_plot=False)

nuevo  = normalized_values.diff().shift(-1).fillna(0).applymap(lambda x : 1 if x>0  else 0)

nuevo.columns = ['Y']


data = pd.concat([normalized_values,bbp,moving_avarage,rsi_val,rsi_spy,momentum,sma_cross,nuevo],axis=1)
data.columns = ['normalized_values','bbp','moving_avarage','rsi_val','rsi_spy','momentum','sma_cross','nuevo']

dataset = data.iloc[14:,:]


X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]
test_size = 0.33
X_train , X_test , y_train , y_test = train_test_split (X,Y,test_size=test_size)
model = XGBClassifier(booster='gbtree',max_depth=1)#, n_estimators = 200,silent=False , n_jobs = 3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
datos = pd.DataFrame(columns=['y_test','y_predict'])
predictions = [value for value in y_pred]
pdb.set_trace()
datos['y_test'] = y_test
datos['y_predict'] = predictions

datos = pd.concat([datos , pd.DataFrame(data=model.predict_proba(X_test),columns=['1','2'],index=datos.index.tolist())],axis=1)

datos = datos.loc[datos['1'] >= 0.65 | datos['2'] >= 0.65]



#datos.plot(x='y_test',y='y_predict',kind='scatter')
#plt.show()
#
#rmse            = math.sqrt(((y_test - predictions) ** 2 ).sum()/y_test.shape[0])
#
#print "Out of sample results"
#print "RMSE: " , rmse
#c = np.corrcoef(predictions, y =y_test)
#print "corr: " , c[0,1]



