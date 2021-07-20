import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

X = pd.read_csv("training.csv")
name = pd.get_dummies(X.name)
dt = pd.to_datetime(X.time)
dt = dt.dt
period = [3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]
season = [3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
day = {'Sunday':0, 'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6}
season, day, hour = pd.get_dummies(dt.month.apply(lambda x: season[x - 1])), dt.day_name().apply(lambda x: day[x]), dt.hour
period = pd.get_dummies(hour.apply(lambda x: period[x]))
weekend = []
for d, h in zip(day.tolist(), hour.tolist()):
    if d == 6 or (d == 0 and h < 20) or (d == 5 and h >= 20):
        weekend.append(1)
    else:
        weekend.append(0)
weekend = pd.DataFrame({'weekend':weekend})
X = pd.concat([name, X.iloc[: , 3:], period, weekend, season], axis=1).values
Y, X = X[:, -1], X[:, :-1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15)
regressor = xgb.XGBRegressor(tree_method = 'exact')
regressor.fit(x_train, y_train)

print('RMSE on Training Set', np.mean((regressor.predict(x_train) - y_train) ** 2) ** 0.5)
print('RMSE on Validation Set', np.mean((regressor.predict(x_test) - y_test) ** 2) ** 0.5)
