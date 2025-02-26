 # 数据预处理
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# 读取数据
df = pd.read_csv("hospital_visits.csv")  # 访客数据
df['visit_date'] = pd.to_datetime(df['visit_date'])  # 转换日期格式
df = df.set_index('visit_date')  # 设置索引

# 计算每日患者流量
df = df.resample('D').count()
df.rename(columns={'patient_id': 'patient_count'}, inplace=True)


#ARIMA 时间序列预测
# 训练 ARIMA 模型
arima_model = ARIMA(df['patient_count'], order=(5,1,0))  # (p,d,q) 选择 (5,1,0)
arima_fit = arima_model.fit()
df['arima_forecast'] = arima_fit.predict(start=len(df), end=len(df)+30, dynamic=False)

#XGBoost 时间序列预测
# 构造特征
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
df['lag_1'] = df['patient_count'].shift(1)
df = df.dropna()

# 划分训练集和测试集
train = df.iloc[:-30]
test = df.iloc[-30:]

# 训练 XGBoost 模型
xgb_model = XGBRegressor(objective='reg:squarederror')
xgb_model.fit(train[['day', 'month', 'year', 'lag_1']], train['patient_count'])

# 预测
df['xgb_forecast'] = xgb_model.predict(df[['day', 'month', 'year', 'lag_1']])

# tabpy
#在 Tableau 计算字段 中调用 Python
SCRIPT_REAL("
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
model = ARIMA(_arg1, order=(5,1,0)).fit()
return model.forecast(30)", 
SUM([patient_count])
)

# 计算误差
mape_arima = mean_absolute_percentage_error(test['patient_count'], test['arima_forecast'])
mape_xgb = mean_absolute_percentage_error(test['patient_count'], test['xgb_forecast'])

rmse_arima = np.sqrt(mean_squared_error(test['patient_count'], test['arima_forecast']))
rmse_xgb = np.sqrt(mean_squared_error(test['patient_count'], test['xgb_forecast']))

print(f"ARIMA MAPE: {mape_arima:.2%}, RMSE: {rmse_arima}")
print(f"XGBoost MAPE: {mape_xgb:.2%}, RMSE: {rmse_xgb}")
