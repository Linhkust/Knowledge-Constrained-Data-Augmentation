import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math
from scipy.stats import pearsonr
import time
from tabpfn import TabPFNRegressor

def model_performance(y_test, y_pred):
    # ML perspective
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    performance_result = {'RMSE': math.sqrt(mse),
                          'NRMSE': (((math.sqrt(mse)) / abs(np.mean(y_test))) * 100),
                          'MAE': mae,
                          'MAPE': mean_absolute_percentage_error(y_test, y_pred),
                          'R2': r2,
                          'std': y_pred.std(),
                          'rho': pearsonr(y_pred, y_test)[0],
                          'ref': y_test.std()
                          }

    return performance_result


# TabPFN
def _tabpfn_(train, test, target):
    start = time.time()

    regressor = TabPFNRegressor(ignore_pretraining_limits=True)
    regressor.fit(train.drop(target, axis=1), train[target])

    # Predict on the test set
    predictions = regressor.predict(test.drop(target, axis=1))

    # Evaluate the model
    performance = model_performance(test[target].values, predictions)
    finish = time.time()
    running_time = ("%.2f" % ((finish - start) / 60))

    performance['Framework'] = 'TabPFN'
    performance['Training time'] = running_time
    return performance