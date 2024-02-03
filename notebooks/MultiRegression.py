from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import xgboost as xgb
import lightgbm as lgb
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


def test_dataset(x_train, x_test, y_train, y_test):


    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)


    print("\nRandom Forest Regression:")
    rf = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    rf.fit(x_train, y_train.ravel())
    pred2_rf = rf.predict(x_test)
    mse_rf = mean_squared_error(y_test.ravel(), pred2_rf)
    mae_rf = mean_absolute_error(y_test.ravel(), pred2_rf)
    r2_rf = r2_score(y_test.ravel(), pred2_rf)
    print('mse: %.4f' % mse_rf)
    print('mae: %.4f' % mae_rf)
    print('r2: %.4f' % r2_rf)

    print("\nDecision Tree Regression:")
    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train.ravel())
    pred2_dt = dt.predict(x_test)
    mse_dt = mean_squared_error(y_test.ravel(), pred2_dt)
    mae_dt = mean_absolute_error(y_test.ravel(), pred2_dt)
    r2_dt = r2_score(y_test.ravel(), pred2_dt)
    print('mse: %.4f' % mse_dt)
    print('mae: %.4f' % mae_dt)
    print('r2: %.4f' % r2_dt)

    # print("\nK Neighbors Regression:")
    # knr = KNeighborsRegressor(n_neighbors=50)
    # knr.fit(x_train, y_train.ravel())
    # pred2_knr = knr.predict(x_test)
    # mse_knr = mean_squared_error(y_test.ravel(), pred2_knr)
    # mae_knr = mean_absolute_error(y_test.ravel(), pred2_knr)
    # r2_knr = r2_score(y_test.ravel(), pred2_knr)
    # print('mse: %.4f' % mse_knr)
    # print('mae: %.4f' % mae_knr)
    # print('r2: %.4f' % r2_knr)

    print("\nLinear Regression:")
    lir = LinearRegression()
    lir.fit(x_train, y_train.ravel())
    pred2_lir = lir.predict(x_test)
    print(pred2_lir)
    mse_lir = mean_squared_error(y_test.ravel(), pred2_lir)
    mae_lir = mean_absolute_error(y_test.ravel(), pred2_lir)
    r2_lir = r2_score(y_test.ravel(), pred2_lir)
    print('mse: %.4f' % mse_lir)
    print('mae: %.4f' % mae_lir)
    print('r2: %.4f' % r2_lir)

    print("\nGradient Boosting Regression:")
    gbr = GradientBoostingRegressor()
    gbr.fit(x_train, y_train.ravel())
    pred2_gbr = gbr.predict(x_test)
    mse_gbr = mean_squared_error(y_test.ravel(), pred2_gbr)
    mae_gbr = mean_absolute_error(y_test.ravel(), pred2_gbr)
    r2_gbr = r2_score(y_test.ravel(), pred2_gbr)
    print('mse: %.4f' % mse_gbr)
    print('mae: %.4f' % mae_gbr)
    print('r2: %.4f' % r2_gbr)

    print("\nLGBMRegression:")
    gmb = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=20, force_col_wise=True, verbose=-1)
    gmb.fit(x_train, y_train.ravel())
    pred2_gmb = gmb.predict(x_test)
    mse_gmb = mean_squared_error(y_test.ravel(), pred2_gmb)
    mae_gmb = mean_absolute_error(y_test.ravel(), pred2_gmb)
    r2_gmb = r2_score(y_test.ravel(), pred2_gmb)
    print('mse: %.4f' % mse_gmb)
    print('mae: %.4f' % mae_gmb)
    print('r2: %.4f' % r2_gmb)

    print("\nXGBoost:")
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', max_depth = 3, learning_rate = 0.1, n_estimators = 100,  n_jobs = -1)
    xg_reg.fit(x_train, y_train)
    pred_xg = xg_reg.predict(x_test)
    mse_xg = mean_squared_error(y_test, pred_xg)
    mae_xg = mean_absolute_error(y_test, pred_xg)
    r2_xg = r2_score(y_test, pred_xg)
    print("XGBoost Regression:")
    print('mse: %.4f' % mse_xg)
    print('mae: %.4f' % mae_xg)
    print('r2: %.4f' % r2_xg)


    return