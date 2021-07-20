import pandas as pd
import xgboost as xgb

x, x_test = pd.read_csv("training.csv"), pd.read_csv("testing.csv")
test_id = x_test.id.tolist()
name, name_test = pd.get_dummies(x.name), pd.get_dummies(x_test.name)
dt, dt_test = pd.to_datetime(x.time), pd.to_datetime(x_test.time)
dt, dt_test = dt.dt, dt_test.dt
period, season = [3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3], [3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
day = {'Sunday':0, 'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6}
season_test, day_test, hour_test = pd.get_dummies(dt_test.month.apply(lambda x: season[x - 1])), dt_test.day_name().apply(lambda x: day[x]), dt_test.hour
season_test = pd.concat([pd.DataFrame({0:[0] * len(x_test), 1:[0] * len(x_test)}), season_test], axis=1)
season, day, hour = pd.get_dummies(dt.month.apply(lambda x: season[x - 1])), dt.day_name().apply(lambda x: day[x]), dt.hour
period, period_test = pd.get_dummies(hour.apply(lambda x: period[x])), pd.get_dummies(hour_test.apply(lambda x: period[x]))
weekend = []
for d, h in zip(day.tolist(), hour.tolist()):
    if d == 6 or (d == 0 and h < 20) or (d == 5 and h >= 20):
        weekend.append(1)
    else:
        weekend.append(0)
weekend = pd.DataFrame({'weekend':weekend})
weekend_test = []
for d, h in zip(day_test.tolist(), hour_test.tolist()):
    if d == 6 or (d == 0 and h < 20) or (d == 5 and h >= 20):
        weekend_test.append(1)
    else:
        weekend_test.append(0)
weekend_test = pd.DataFrame({'weekend':weekend_test})
x, x_test = pd.concat([name, x.iloc[: , 3:], period, weekend, season], axis=1).values, pd.concat([name_test, x_test.iloc[: , 3:], period_test, weekend_test, season_test], axis=1).values
y, x = x[:, -1], x[:, :-1]
regressor = xgb.XGBRegressor(tree_method = 'exact')
regressor.fit(x, y)
pd.DataFrame({'Id':test_id, 'Predicted':regressor.predict(x_test).tolist()}).to_csv('xgb.csv',index = False)
