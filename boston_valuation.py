import pandas as pd 
import numpy as np

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Gather data
boston_data = load_boston()
data = pd.DataFrame(data = boston_data.data, columns = boston_data.feature_names)

features = data.drop(["INDUS","AGE"],axis = 1)

log_prices = np.log(boston_data.target)
target = pd.DataFrame(log_prices,columns = ["PRICE"])
features.head()


# Run Linearregression model

reg = LinearRegression().fit(features,target)
fitted_values = reg.predict(features)

# lets calculate the MSE and RMSE
MSE = mean_squared_error(target,fitted_values)
RMSE = np.sqrt(MSE)


crime_idx = 0
zn_idx = 1
chas_idx = 2
room_idx = 4
ptratio_idx = 8


proparty_state = np.ndarray(shape=(1,11))
proparty_state = features.mean().values.reshape(1,11)


def get_log_estimate(nr_rooms, student_per_classroom, next_to_river = False, high_confidance = True):
    
    # Configure proparty
    
    proparty_state[0][room_idx] = nr_rooms
    proparty_state[0][ptratio_idx] = student_per_classroom
    
    if next_to_river is True:
        proparty_state[0][chas_idx] = 1
    else:
        proparty_state[0][chas_idx] = 0
    
    # Estimating price 
    
    log_estimate = reg.predict(proparty_state)[0][0]
    
    # calculate Range 
    
    if high_confidance:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    return log_estimate , upper_bound, lower_bound , interval



# Lets change this log price to todays price

ZELLO_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZELLO_MEDIAN_PRICE/ np.median(boston_data.target)
SCALE_FACTOR


def get_doller_estimate(rm, ptratio, river = False, high_diff = True):
    """Estimate the Current price of a proparty in Boston.
    
    key arguments :
    
    rm :  Number of Room per house.
    ptratio : Number of stident per classroom.
    river : House that are next to the Charles River.
    high_deff : The confidentiality of price Interval in normal distribution
    
    """
    
    
    if rm < 1 or ptratio < 1:
        print("INPUT is Unrealistic . Please Try Again.")
        return
    
    log_est, upper, lower, intv = get_log_estimate(rm, ptratio,next_to_river=river,high_confidance = high_diff)
    
    #revarting log in to real price 
    doller_est = np.e**log_est * SCALE_FACTOR * 1000
    doller_high = np.e**upper *SCALE_FACTOR * 1000
    doller_low = np.e**lower * SCALE_FACTOR * 1000
    
    #round the price 
    round_est = np.around(doller_est,-3)
    round_high = np.around(doller_high,-3)
    round_low = np.around(doller_low,-3)
    
    print(f"House Price at Current Market Value is {round_est}")
    print(f"Higher Bound at Current Market Value is {round_high}")
    print(f"Lower Bound at Current Market Value is {round_low}")
    
    
