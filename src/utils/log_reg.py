import numpy as np
import copy
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator


def logistic(x, b):
    return 1 / (1 + np.exp(-b[1] * (x - b[0]))) + 0


def logistic_loss(x, *args):
    return args[1] - logistic(args[0], x)


def logistic_regression(x, y, ax=None):
    b1 = 1
    b0 = 0
    init_params = [b0, b1]
    opt_res = least_squares(logistic_loss, init_params, args=(x, y), diff_step=np.finfo(np.float32).eps ** (1/3))
    #     print(init_params, opt_res)
    if ax is not None:
        range_arr = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(range_arr, logistic(range_arr, opt_res.x))

    return opt_res


def norm_x(x):
    x_mean = np.mean(x)
    new_x = x - x_mean
    x_std = np.std(new_x)
    new_x = new_x / x_std
    return new_x, x_mean, x_std


def norm_y(y):
    y_min = np.min(y)
    y = y - y_min
    y_max = np.max(y)
    y = y / y_max
    return y, y_min, y_max


def denorm_x(x, mean, std):
    return (x * std) + mean


def log_reg_function(x, y):
    new_x, x_mean, x_std = norm_x(x)
    new_y, y_min, y_max = norm_y(y)
    opt_res = logistic_regression(new_x, new_y)
    return lambda k, params=opt_res.x, std=x_std, mean=x_mean: logistic((k - mean) / std, params) * y_max + y_min

# Logistic regression wrapper for sklearn estimators
# The wrapping makes subclass params available in parent class
def lr_wrap_regressor(subclass):
    class WrappedRegressor(subclass):
        def __init__(self, **kwargs):
            self.log_reg_f_ = None
            super().__init__(**kwargs)

        def fit(self, x, y):
            super().fit(x, y)
            ypred = super().predict(x)
            self.log_reg_f_ = log_reg_function(ypred, y)

        def predict(self, x):
            return self.log_reg_f_(super().predict(x))
        
        def predict_unfitted(self, x):
            return super().predict(x)
        
        def predict_fit(self, x):
            return self.log_reg_f_(x)

        def get_params(self, deep=True):
            params = super().get_params(deep)
            # Hack to make get_params return base class params...
            cp = copy.copy(self)
            cp.__class__ = subclass
            params.update(subclass.get_params(cp, deep))
            return params
    return WrappedRegressor

class IdentityMethod(BaseEstimator):
    def predict(self, x):
        return x
    
    def fit(self, x, y):
        pass