import importlib.util
spec = importlib.util.spec_from_file_location("RegressionAnalysis", \
"/Users/Jiao/Desktop/Greenware Service/code_preoject_convenience_stores/Regression/code.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
RegressionAnalysis = foo.RegressionAnalysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
df = pd.read_excel('model_data.xlsx').set_index('MLS')
df = df[['Sold_Price', 'Area_Size', 'Lottery_Dummy', \
        'Occupation_Dummy', 'Days_Open_5', 'Size_Franchise_Order2']]

class MonteCarloSimulations(RegressionAnalysis):
    def __init__(self, df):
        RegressionAnalysis.__init__(self, df)
    
    def ols_model(self, y, X):
        return super().linear_regression(y, X)
    
    def sum_of_square(self, data):
        return super().calculate_sum_of_squares(data)
    
    def generate_coef_samples(self, params, stds, n=1000):
        # ** params, stds are from OLS model results
        # ** pandas series type 
        coef_samples_df = pd.DataFrame(columns=params.index)
        for i in range(len(params)):
            var_name = params.index[i]
            mu = params[i]
            sigma = stds[i]
            coef_samples_df[var_name] = np.random.normal(mu, sigma, n)
        return coef_samples_df

    def y_simulation_with_single_observation(self, obs, coef_samples_df, n=1000, check_one_var=None):
        # ** obs, pandas series type
        obs = obs.values
        X = np.insert(obs, 0, 1)
        coef_samples_array = coef_samples_df.values
        col_number = coef_samples_array.shape[1]
        Y = []

        for i in range(n):
            coefs = []
            for c in range(col_number):
                coef = np.random.choice(coef_samples_array[:, c])
                coefs.append(coef)
            y = np.dot(np.transpose(X), coefs)
            Y.append(y)
        return Y
    
    def confidence_interval_calculation(self, data, alpha=0.95):
        mean = np.mean(data)
        ss = self.sum_of_square(data)
        sigma = math.sqrt(ss/(len(data) - 1))
        upper_ci = mean + st.norm.ppf(alpha)*sigma
        lower_ci = mean - st.norm.ppf(alpha)*sigma
        return mean, sigma, upper_ci, lower_ci
    
    def gernerate_X_samples(self, var, n=1000):
        max_v = max(var)
        min_v = min(var)
        X = np.random.uniform(min_v, max_v, size=n)
        return np.sort(X)

    def y_simulation_with_single_variable(self, var, var_position, obs, coef_samples_df, n=2000):
        X_generated_sample = self.gernerate_X_samples(var)
        obs = obs.values
        X = np.insert(obs, 0, 1)

        coef_samples_array = coef_samples_df.values
        col_number = coef_samples_array.shape[1]
        
        x_repeat = 100
        Y_with_repeated_x =  np.zeros(shape=(len(X_generated_sample), x_repeat))
        row = 0
        for x in X_generated_sample:
            Y = []
            for i in range(x_repeat):
                coefs = []
                for c in range(col_number):
                    coef = np.random.choice(coef_samples_array[:, c])
                    coefs.append(coef)
                X[var_position] = x
                y = np.dot(np.transpose(X), coefs)
                Y.append(y)
            Y_with_repeated_x[row] = Y
            row = row + 1
        return Y_with_repeated_x


ms = MonteCarloSimulations(df)
ols_model = ms.ols_model(df['Sold_Price'], df[['Area_Size', 'Lottery_Dummy', \
                            'Occupation_Dummy', 'Days_Open_5', 'Size_Franchise_Order2']])

coef_samples = ms.generate_coef_samples(ols_model.params, ols_model.bse)

observation_choosen = df.loc['W1359413'][1:]

# Y_simulated = ms.y_simulation_with_single_observation(observation_choosen, coef_samples)
# plt.plot(Y_simulated)
# plt.show()

# mean, sigma, upper_bound, lower_bound = ms.confidence_interval_calculation(Y_simulated)
# print('95% confidence interval,','mean:', mean, 'standard deviation', sigma, \
#     '\n', 'lower bound:', lower_bound, 'upper bound:', upper_bound)

x_array = ms.gernerate_X_samples(df['Area_Size'])
Y_simulated = ms.y_simulation_with_single_variable(df['Area_Size'], 1, observation_choosen,coef_samples)
# print(Y_simulated.shape)

# ** calculate ci
ci_lower_bound = []
ci_upper_bound = []
M = []
for i in range(Y_simulated.shape[0]):
    mean, sigma, upper_ci, lower_ci = ms.confidence_interval_calculation(Y_simulated[i])
    ci_lower_bound.append(lower_ci)
    ci_upper_bound.append(upper_ci)
    M.append(mean)

flat_y = Y_simulated.flatten()
# x_axis = np.random.uniform(0, len(flat_y), 1000)
plt.plot(mean)
plt.plot(ci_lower_bound)
plt.plot(ci_upper_bound)
plt.show()
# print(len(ci_upper_bound))