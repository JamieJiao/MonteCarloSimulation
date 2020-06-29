import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import statsmodels.api as sm

class MonteCarloSimulations():
    
    def linear_regression(self, y, X):
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        return results
    
    def calculate_sum_of_squares(self, data):
        sum_of_squares = []
        mean_value = np.mean(data)
        for value in data:
            sum_of_squares.append((value - mean_value)**2)
        return sum(sum_of_squares)
    
    def generate_coef_samples(self, params, stds, n=1000):
        # params, stds are from OLS model results, pandas series type 
        coef_samples_df = pd.DataFrame(columns=params.index)
        for i in range(len(params)):
            var_name = params.index[i]
            mu = params[i]
            sigma = stds[i]
            coef_samples_df[var_name] = np.random.normal(mu, sigma, n)
        # coef_samples_df, pandas dataframe
        return coef_samples_df
    
    def parameters_setup(self, obs, coef_samples_df):
        # obs is one observation, pandas series type
        # change to numpy array, add constant term
        obs = obs.values
        var_values = np.insert(obs, 0, 1)
        coef_samples_array = coef_samples_df.values
        coef_count = coef_samples_array.shape[1]
        return var_values, coef_samples_array, coef_count

    def y_simulation_with_single_observation(self, obs, coef_samples_df, n=1000):
        var_values, coef_samples_array, coef_count = self.parameters_setup(\
                                                obs, coef_samples_df)
        Y = []
        for i in range(n):
            coefs = []
            for c in range(coef_count):
                coef = np.random.choice(coef_samples_array[:, c])
                coefs.append(coef)
            y = np.dot(np.transpose(var_values), coefs)
            Y.append(y)
        return Y
    
    def confidence_interval_calculation(self, data, alpha=0.95):
        mean = np.mean(data)
        ss = self.calculate_sum_of_squares(data)
        sigma = math.sqrt(ss/(len(data) - 1))
        upper_ci = mean + st.norm.ppf(alpha)*sigma
        lower_ci = mean - st.norm.ppf(alpha)*sigma
        return mean, sigma, upper_ci, lower_ci
    
    def generate_X_sample_uniform(self, var, n=1000):
        max_value = max(var)
        min_value = min(var)
        X = np.random.uniform(min_value, max_value, size=n)
        return np.sort(X)

    def y_simulation_with_repeating_one_variable(self, var, var_position, \
                                        obs, coef_samples_df, x_repeat_times):
        # var input data, pandas series type
        X_generated_sample = self.generate_X_sample_uniform(var)
        var_values, coef_samples_array, coef_count = self.parameters_setup(\
                                                obs, coef_samples_df)
        
        Y_with_repeating_x = np.zeros(shape=(len(X_generated_sample), \
                                                x_repeat_times))
        row = 0
        for x in X_generated_sample:
            Y = []
            var_values[var_position] = x
            # ** repeat x to generate its correspoding y sample
            for i in range(x_repeat_times):
                coefs = []
                for c in range(coef_count):
                    coef = np.random.choice(coef_samples_array[:, c])
                    coefs.append(coef)
                # X[var_position] = x
                y = np.dot(np.transpose(var_values), coefs)
                Y.append(y)

            Y_with_repeating_x[row] = Y
            row = row + 1
        return Y_with_repeating_x