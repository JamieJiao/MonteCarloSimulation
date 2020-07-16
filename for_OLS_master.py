import importlib.util
spec = importlib.util.spec_from_file_location("RegressionAnalysis", \
"/Users/Jiao/Desktop/Greenware Service/code_preoject_convenience_stores/Regression/code.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
RegressionAnalysis = foo.RegressionAnalysis

import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
df = pd.read_excel('model_data.xlsx').set_index('MLS')
df = df[['Sold_Price', 'Area_Size', 'Rental_Month', 'Lottery_Dummy', \
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
    
    def parameters_setup(self, obs, coef_samples_df):
        # ** obs is one observation, pandas series type
        # ** change to numpy array
        obs = obs.values
        X = np.insert(obs, 0, 1)
        coef_samples_array = coef_samples_df.values
        coef_count = coef_samples_array.shape[1]
        return X, coef_samples_array, coef_count

    def y_simulation_with_single_observation(self, obs, coef_samples_df, \
                                            n=10000, check_one_var=None):
        X, coef_samples_array, coef_count = self.parameters_setup(\
                                                obs, coef_samples_df)
        Y = []
        for i in range(n):
            coefs = []
            for c in range(coef_count):
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
    
    def gernerate_X_sample_uniform(self, var, n=1000):
        max_value = max(var)
        min_value = min(var)
        X = np.random.uniform(min_value, max_value, size=n)
        return np.sort(X)

    def gernerate_X_sample_random(self, var, n=10000):
        max_value = max(var)
        min_value = min(var)
        X = np.random.randint(min_value, max_value, size=n)
        return X

    def y_simulation_with_single_variable(self, var, var_position, \
                                        obs, coef_samples_df, x_repeat_times):
        # X_generated_sample = self.gernerate_X_sample_uniform(var)
        X_generated_sample = self.gernerate_X_sample_random(var)
        X, coef_samples_array, coef_count = self.parameters_setup(\
                                                obs, coef_samples_df)
        
        # Y_with_repeating_x = np.zeros(shape=(len(X_generated_sample), \
        #                                         x_repeat_times))
        Y_with_repeating_x = []
        row = 0
        for x in X_generated_sample:
            Y = []
            X[var_position] = x
            # ** repeat x to generate its correspoding y sample
            # for i in range(x_repeat_times):
            coefs = []
            for c in range(coef_count):
                coef = np.random.choice(coef_samples_array[:, c])
                coefs.append(coef)
            # X[var_position] = x
            y = np.dot(np.transpose(X), coefs)
            # Y.append(y)
            Y_with_repeating_x.append(y)

            # Y_with_repeating_x[row] = Y
            # row = row + 1
        return Y_with_repeating_x
    
    def y_simulation_with_multiple_variables(self, obs, coef_samples_df, \
                                        x_repeat_times, var1, var1_posi, \
                                            var2=None, var2_posi=None, n=10000):
        X_generated_sample1 = self.gernerate_X_sample_random(var1)
        X_generated_sample2 = self.gernerate_X_sample_random(var2)
        X, coef_samples_array, coef_count = self.parameters_setup(\
                                                obs, coef_samples_df)

        # Y_with_repeating_x = np.zeros(shape=(len(X_generated_sample1), \
        #                                         x_repeat_times))
        Y_with_repeating_x = []

        X1_choose = []
        X2_choose = []
        row = 0
        for r in range(n):
            Y = []
            X1 = np.random.choice(X_generated_sample1)
            X2 = np.random.choice(X_generated_sample2)
            X1_choose.append(X1)
            X2_choose.append(X2)
        # X1_choose = sorted(X1_choose)
        # X2_choose = sorted(X2_choose)

        # ** repeat x to generate its correspoding y sample
        for loop_n in range(n):
            X[var1_posi] = X1_choose[loop_n]
            X[var2_posi] = X2_choose[loop_n]
            # Y = []
            # for repeat_same_x in range(x_repeat_times):
            coefs = []
            for coef_position in range(coef_count):
                coef = np.random.choice(coef_samples_array[:, coef_position])
                coefs.append(coef)
            y = np.dot(np.transpose(X), coefs)
            # Y.append(y)
            Y_with_repeating_x.append(y)
            # Y_with_repeating_x[row] = Y
            # row = row + 1
        return Y_with_repeating_x, X1_choose, X2_choose

ms = MonteCarloSimulations(df)
df_x = df[['Area_Size', 'Rental_Month', 'Lottery_Dummy', \
            'Occupation_Dummy', 'Days_Open_5', 'Size_Franchise_Order2']]
ols_model = ms.ols_model(df['Sold_Price'], df_x)

x_array = ms.gernerate_X_sample_uniform(df['Rental_Month'])
coef_samples = ms.generate_coef_samples(ols_model.params, ols_model.bse)

observation_choosen = df_x.loc['W1359413']

# Y_simulated = ms.y_simulation_with_single_observation(observation_choosen, coef_samples)
# plt.plot(Y_simulated)
# plt.show()

# mean, sigma, upper_bound, lower_bound = ms.confidence_interval_calculation(Y_simulated)
# print('95% confidence interval,','mean:', mean, 'standard deviation', sigma, \
#     '\n', 'lower bound:', lower_bound, 'upper bound:', upper_bound)
def results_display(x_repeat_time, var_name):
    
    Y_simulated = ms.y_simulation_with_single_variable(df[var_name], 2, \
                                observation_choosen, coef_samples, x_repeat_time)
    
    # ** calculate ci
    # ci_lower_bound = []
    # ci_upper_bound = []
    # M = []
    # length = Y_simulated.shape[0]
    # for i in range(length):
    #     mean, sigma, upper_ci, lower_ci = \
    #                         ms.confidence_interval_calculation(Y_simulated[i])
    #     ci_lower_bound.append(lower_ci)
    #     ci_upper_bound.append(upper_ci)
    #     M.append(mean)

    plt.plot(Y_simulated)
    # plt.xlabel(var_name)
    # plt.ylabel('Simulated Sold Price')
    plt.title('Simulated Sold Price Model Without Rental')
    plt.show()

    # plt.plot(x_array, M, label='Mean')
    # plt.xlabel(var_name)
    # plt.ylabel('Simulated Sold Price')
    # plt.plot(x_array, ci_lower_bound, label='Lower Bound')
    # plt.plot(x_array, ci_upper_bound, label='Upper Bound')
    # plt.legend()
    # plt.title('confidence interval, x repeated times:{}'. format(x_repeat_time))
    # plt.show()

x_repeat_times = [100, 1000, 10000]
x_repeat_time = x_repeat_times[0]

# results_display(x_repeat_time, 'Area_Size')


df_x = df[['Area_Size', 'Rental_Month', 'Lottery_Dummy', \
                    'Occupation_Dummy', 'Days_Open_5', 'Size_Franchise_Order2']]
ols_model = ms.ols_model(df['Sold_Price'], df_x)

observation_choosen = df_x.loc['W1359413']
# print(observation_choosen)

coef_samples = ms.generate_coef_samples(ols_model.params, ols_model.bse)
# print(ols_model.params, ols_model.bse)


x_array_random1 = ms.gernerate_X_sample_random(df['Area_Size'])
x_array_random2 = ms.gernerate_X_sample_random(df['Rental_Month'])
# plt.plot(x_array_random1)

Y_simulated, X1, X2 = ms.y_simulation_with_multiple_variables(observation_choosen, \
                                        coef_samples, 0, \
                                        df['Area_Size'], 1, \
                                        df['Rental_Month'], 2)

# ci_lower_bound = []
# ci_upper_bound = []
# M = []
# for i in range(Y_simulated.shape[0]):
#     mean, sigma, upper_ci, lower_ci = \
#                         ms.confidence_interval_calculation(Y_simulated[i])
#     ci_lower_bound.append(lower_ci)
#     ci_upper_bound.append(upper_ci)
#     M.append(mean)

# plt.plot(Y_simulated)
# plt.title('Simulated Sold Price Model With Size and Rental')
# plt.show()

# plt.plot(M, label='Mean')

# plt.ylabel('Simulated Sold Price')
# plt.plot(ci_lower_bound, label='Lower Bound')
# plt.plot(ci_upper_bound, label='Upper Bound')
# plt.legend()
# plt.title('confidence interval, x repeated times:{}'. format(x_repeat_time))
# plt.show()


def distribution_plot(data):
    mu = np.mean(data)
    sigma = np.sqrt(ms.sum_of_square(data) / len(data))
    count, bins, ignored = plt.hist(data, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    plt.show()

# distribution_plot(Y_simulated.flatten())

# plt.plot(df_x['Area_Size'], df_x['Rental_Month'], 'bv', c='r')
# plt.show()

# regression model between rental and size
regression_rental_size = ms.ols_model(df_x['Rental_Month'], df_x['Area_Size'])
# print(regression_rental_size.summary())
# create quadratic term for size
df_area_size_quadratic = df[['Area_Size']]
df_area_size_quadratic['Area_Size_Quadratic'] = df['Area_Size']**2
regression_rental_size_quadratic = ms.ols_model(df_x['Rental_Month'], df_area_size_quadratic['Area_Size_Quadratic'])
# print(regression_rental_size_quadratic.summary())

size = len(df['Area_Size'])
constant_coef_mean = regression_rental_size.params[0]
area_size_coef_mean = regression_rental_size.params[1]
constant_coef_std = regression_rental_size.bse[0]
area_size_coef_std = regression_rental_size.bse[1]

area_size_mean = np.mean(df['Area_Size'])
ss_area_size = ms.sum_of_square(df['Area_Size'])
size = len(df['Area_Size'])
area_size_std = math.sqrt(ss_area_size/(size - 1))

area_size_quadratic_mean = np.mean(df_area_size_quadratic['Area_Size_Quadratic'])
ss_area_size_quadratic = ms.sum_of_square(df_area_size_quadratic['Area_Size_Quadratic'])
area_size_quadratic_std = math.sqrt(ss_area_size_quadratic/ (size - 1))



def simulate_rental(area_size_generated, mu2, mu3, sigma2, sigma3):
    
    constant_coef_generated = np.random.normal(mu2, sigma2, 1)
    area_size_coef_generated = np.random.normal(mu3, sigma3, 1)
    rental_simulated = constant_coef_generated*1 + area_size_coef_generated*area_size_generated

    return rental_simulated

def simulate_rental_quadratic(area_size_quadratic_mean, area_size_quadratic_std, \
                                coefs_mean, coefs_std_error):
    rental_simulated = 0
    variables = [1]
    # area_size_generated = np.random.normal(area_size_mean, area_size_std, 1)
    # variables.append(area_size_generated)
    area_size_quadratic_generated = np.random.normal(area_size_quadratic_mean, area_size_quadratic_std, 1)
    variables.append(area_size_quadratic_generated)

    n = len(coefs_mean)
    for i in range(n):
        coef_generated = np.random.normal(coefs_mean[i], coefs_std_error[i], 1)
        term = coef_generated * variables[i]
        rental_simulated = rental_simulated + term

    return rental_simulated

sample = []

def simulate_sold_price(obs, area_size_posi, area_size_generated, rental_simulated, coefs_mean, coefs_std_error):
    sold_price_simulated = 0

    variables = np.array(obs)
    variables = np.insert(variables, 0, 1)
    
    variables[area_size_posi] = area_size_generated

    for i in range(len(coefs_mean)):
        coef_generated = np.random.normal(coefs_mean[i], coefs_std_error[i], 1)
        if coefs_mean.index[i] != 'Rental_Month': 
            term = coef_generated * variables[i]
        else:
            term = coef_generated * rental_simulated
        sold_price_simulated = sold_price_simulated + term
    
    return sold_price_simulated

# def sold_price_sample_generated():


# run the full model
full_regression = ms.ols_model(df['Sold_Price'], df_x)
# print(full_regression.summary())
# exclude rental from the observation
# observation_choosen_no_rental = [observation_choosen[i] for i in range(len(observation_choosen)) if i != 1]
full_coefs = full_regression.params
full_coefs_std_error = full_regression.bse
# exclude the rental's coefficient, because we want to use the generated rental 
# coefs_mean = [full_coefs[i] for i in range(len(full_coefs)) if full_coefs.index[i] != 'Rental_Month']
# coefs_std_error = [full_coefs_std_error[i] for i in range(len(full_coefs_std_error)) \
#                     if full_coefs_std_error.index[i] != 'Rental_Month']

def monte_carlo_with_x_repeat():
    sample_size = 1000
    sold_price_simulated_sample = []
    rental_simulated_sample = []
    area_size_simulated_sample = []

    x_repeat_time = 100

    ci_lower_bound = []
    ci_upper_bound = []
    M = []

    for n in range(sample_size):
        # simulate rental based on the regression between rental and size
        sold_price_simulated_sub_sample = []
        
        area_size_generated = np.random.normal(area_size_mean, area_size_std, 1)

        if area_size_generated < min(df['Area_Size']) or area_size_generated > max(df['Area_Size']):
            continue

        else:
            area_size_simulated_sample.append(area_size_generated[0])
            rental_simulated = simulate_rental(area_size_generated, constant_coef_mean, area_size_coef_mean, \
                                                    constant_coef_std, area_size_coef_std)
            rental_simulated_sample.append(rental_simulated[0])

            # rental_simulated = simulate_rental_quadratic(area_size_mean, area_size_std, \
            #                                                 area_size_quadratic_mean, area_size_quadratic_std, \
            #                                                 regression_rental_size_quadratic.params, \
            #                                                 regression_rental_size_quadratic.bse)
            
            for repeat in range(x_repeat_time):

                sold_price_simulated = simulate_sold_price(observation_choosen, 1, \
                                                    area_size_generated, rental_simulated, \
                                                full_coefs, full_coefs_std_error)

                sold_price_simulated_sub_sample.append(sold_price_simulated)

            mean, sigma, upper_ci, lower_ci = \
                                ms.confidence_interval_calculation(sold_price_simulated_sub_sample)
            ci_lower_bound.append(lower_ci)
            ci_upper_bound.append(upper_ci)
            M.append(mean)

    # area_size_simulated_sample = np.sort(area_size_simulated_sample)
    rental_simulated_sample = np.sort(rental_simulated_sample)
    M = np.sort(M)
    ci_lower_bound = np.sort(ci_lower_bound)
    ci_upper_bound = np.sort(ci_upper_bound)

    plt.plot(rental_simulated_sample, M, label='Mean')

    plt.ylabel('Simulated Sold Price')
    # plt.xlabel('Area Size')
    plt.xlabel('Rental')

    plt.plot(rental_simulated_sample, ci_lower_bound, label='Lower Bound')
    plt.plot(rental_simulated_sample, ci_upper_bound, label='Upper Bound')
    plt.legend()
    plt.title('Confidence Intervals, x Repeated Times:{}'. format(x_repeat_time))
    plt.show()

monte_carlo_with_x_repeat()

sample_size = 1000
sold_price_simulated_sample = []
rental_simulated_sample = []
area_size_simulated_sample = []

def monte_carlo_no_x_repeat():

    for n in range(10000):
        
        area_size_generated = np.random.normal(area_size_mean, area_size_std, 1)
        area_size_simulated_sample.append(area_size_generated)

        rental_simulated = simulate_rental(area_size_generated, constant_coef_mean, area_size_coef_mean, \
                                            constant_coef_std, area_size_coef_std)

        # rental_simulated = simulate_rental_quadratic(area_size_quadratic_mean, area_size_quadratic_std, \
        #                                                 regression_rental_size_quadratic.params, \
        #                                                 regression_rental_size_quadratic.bse)
        
        rental_simulated_sample.append(rental_simulated)

        sold_price_simulated = simulate_sold_price(observation_choosen, 1, \
                                                    area_size_generated, rental_simulated, \
                                                full_coefs, full_coefs_std_error)

        # rental_coef_mean = full_coefs[2]
        # rental_coef_std_error = full_coefs_std_error[2]

        # rental_coef = np.random.normal(rental_coef_mean, rental_coef_std_error, 1)

        # sold_price_simulated = sold_price_simulated_no_rental + rental_coef * rental_simulated

        sold_price_simulated_sample.append(sold_price_simulated)

    mean, sigma, upper_ci, lower_ci = \
                                ms.confidence_interval_calculation(sold_price_simulated_sample)
    print(mean, lower_ci, upper_ci)
    plt.plot(area_size_simulated_sample, sold_price_simulated_sample, 'bv')
    plt.title('Simulated Sold Price Model With Size Quadratic To Rental')
    plt.show()
# monte_carlo_no_x_repeat()

