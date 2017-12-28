import poloniex
import numpy as np
import pandas as pd
import time
from sklearn import ensemble
from sklearn.metrics import accuracy_score

APIKey = ""
Secret = ""

# Get historic data from Poloniex
polo = poloniex.Poloniex(APIKey, Secret)

PAIR = 'BTC_LTC'
# The CANDLESTICK_PERIOD options are 300, 900, 1800, 7200, 14400, and 86400 sec
CANDLESTICK_PERIOD = 300
# Use http://epochconverter.com to get Epoch timestamp
CHART_PERIOD_START = 1483225200    # 01-jan-17 00:00:00 GMT+01
CHART_PERIOD_END = 1514156400      # 25-dec-17 00:00:00 GMT+01

data = polo.returnChartData(PAIR,
                            CANDLESTICK_PERIOD,
                            CHART_PERIOD_START,
                            CHART_PERIOD_END)

# Put the data in a Pandas DF
data = pd.DataFrame(data)
data.index = data['date']
del data['date']
data = data.astype(float)

# Create the classifications we want to predict
data['classification'] = 0

data.loc[data['close'] > data['close'].shift(-1), 'classification'] = -1
data.loc[data['close'] <= data['close'].shift(-1), 'classification'] = 1

# Use the first half as train data, second half as test data:
train_data = data[:int(len(data)*0.9)]
test_data = data[len(train_data):]

train_data_x = train_data.iloc[:, 0:-1]
train_data_y = train_data.iloc[:, -1]

test_data_x = test_data.iloc[:, 0:-1]
test_data_y = test_data.iloc[:, -1]

# Apply random forest classifier:
rfc = ensemble.RandomForestClassifier(n_estimators=50)
rfc.fit(train_data_x, train_data_y)

predictions = rfc.predict(test_data_x)

confidence = rfc.predict_proba(test_data_x)

importance = list(zip(list(test_data_x.columns), rfc.feature_importances_))

print("Accuracy score: "+str(accuracy_score(test_data_y.values, predictions)))
print("\nImportance of each variable in the prediction:")
print(importance)

'''
# Use the fitted model to do Real-Time trading:

while True:
    t=int(time.time());
    data2 = polo.returnChartData(
                            PAIR,
                            CANDLESTICK_PERIOD,
                            t-300,
                            t)
    data2 = pd.DataFrame(data2)
    data2.index = data2['date']
    del data2['date']
    data2 = data2.astype(float)
    pred = rfc.predict(data2)
    conf = rfc.predict_proba(data2)

    if pred==1;

    time.sleep(350)

'''
'''
start = dt.datetime(2017,12,1,17,0).strftime('%s')
end = dt.datetime(2017,12,31,17,0).strftime('%s')
# print(end)
CHART_PERIOD_START = start # 01-dec-17 17:00 | https://www.epochconverter.com/
CHART_PERIOD_END = end   # 24-dec-17 17:00

data = polo.returnChartData(PAIR, CS_PERIOD, CHART_PERIOD_START, CHART_PERIOD_END)


# Put them in a Pandas DF:
data = pd.DataFrame(data)

# conversion. does not work
data['date'] = pd.to_datetime(data['date'], unit='s')
'''
