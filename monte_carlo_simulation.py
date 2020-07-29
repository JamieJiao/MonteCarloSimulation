import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import statsmodels.api as sm

def linear_regression(y, X):
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def calculate_sum_of_squares(data):
    sum_of_squares = []
    mean_value = np.mean(data)
    for value in data:
        sum_of_squares.append((value - mean_value)**2)
    return sum(sum_of_squares)

def confidence_interval_calculation(data, alpha=0.95):
    mean = np.mean(data)
    ss = calculate_sum_of_squares(data)
    sigma = math.sqrt(ss/(len(data) - 1))
    upper_ci = mean + st.norm.ppf(alpha)*sigma
    lower_ci = mean - st.norm.ppf(alpha)*sigma
    return mean, upper_ci, lower_ci

def simulate_rental(square_feet_generated, constant_coef_mean, square_feet_coef_mean, \
                                                    constant_coef_std, square_feet_coef_std):
    """
    Based on regression between reponse variable, Rental, 
    and explanatory variable, SquareFeet,
    to generate simulated Rental.
    """

    constant_coef_generated = np.random.normal(constant_coef_mean, constant_coef_std, 1)
    square_feet_coef_generated = np.random.normal(square_feet_coef_mean, square_feet_coef_std, 1)
    rental_simulated = constant_coef_generated*1 + square_feet_coef_generated*square_feet_generated

    return rental_simulated

def simulate_sold_price(observation_choosen, square_feet_posi, \
                        square_feet_generated, rental_simulated, \
                        coefs_mean, coefs_std_error, simualated_var_name='Rental_Month'):
    """
    Use a specific observation record,
    based on regression, 
    SoldPrice= β_0+ β_1*SquareFeet+ β_2*Rental+ β_3*LotteryDummy ... + ε_i,
    to simulate Sold Price, y.
    """

    sold_price_simulated = 0
    """ change to numpy array"""
    observation_choosen = np.array(observation_choosen)
    """ add constant to the observation """
    observation_choosen = np.insert(observation_choosen, 0, 1)
    
    observation_choosen[square_feet_posi] = square_feet_generated

    for i in range(len(coefs_mean)):
        coef_generated = np.random.normal(coefs_mean[i], coefs_std_error[i], 1)
        """ exculde Rental coefficient, use simulated Rental instead"""
        if coefs_mean.index[i] != simualated_var_name: 
            term = coef_generated * observation_choosen[i]
        else:
            term = coef_generated * rental_simulated
        sold_price_simulated = sold_price_simulated + term
    
    return sold_price_simulated

def sold_price_subsample_generate(observation_choosen, square_feet_generated, \
                                    rental_simulated, full_coefs_mean, full_coefs_std, \
                                    square_feet_posi=1, x_repeat_times=100):
    """
    By repeating X, generate a sample of Y, Sold Price.
    """

    simulated_sold_price_subsample = []
    for repeat in range(x_repeat_times):

        simulated_sold_price = simulate_sold_price(observation_choosen, square_feet_posi, \
                                            square_feet_generated, rental_simulated, \
                                        full_coefs_mean, full_coefs_std)

        simulated_sold_price_subsample.append(simulated_sold_price)
    return simulated_sold_price_subsample

def Simulation_Results(observation_choosen, square_feet_mean, square_feet_td, \
                        constant_coef_mean, constant_coef_std, \
                        square_feet_coef_mean, square_feet_coef_std, \
                        full_coefs_mean, full_coefs_std, sample_size=1000):
    """
    Mean and confidence intervals are generated in each iteration.

    Parameters:
    Observation_choosen (a observation record),
    square_feet_posi (the position of the Square Feet variable in the observation),
    ...
    coefs_mean (coefficients of all variables in the Sold Price regression from statsmodels, 
    Pandas Series like),
    coefs_std_error (standard errors of the coefficientsfrom statsmodels, Pandas Dataseries like).

    Return:
    mean values, lower and upper confidence interval.
    """

    simulated_rental_sample = []
    generated_square_feet_sample = []
    
    ci_lower_bound = []
    ci_upper_bound = []
    means = []
    for n in range(sample_size):
        """ simulate Square Feet """
        square_feet_generated = np.random.normal(square_feet_mean, square_feet_td, 1)
        if square_feet_generated <= 0:
            continue
        else:
            generated_square_feet_sample.append(square_feet_generated[0])
            """ simulate Rental based on the regression between Rental and Square Feet """
            rental_simulated = simulate_rental(square_feet_generated, constant_coef_mean, \
                                                square_feet_coef_mean, \
                                                constant_coef_std, square_feet_coef_std)
            simulated_rental_sample.append(rental_simulated[0])

            simulated_sold_price_subsample = sold_price_subsample_generate(observation_choosen, \
                                                                    square_feet_generated, \
                                                                    rental_simulated, \
                                                                    full_coefs_mean, full_coefs_std)
            mean, upper_ci, lower_ci = \
                                confidence_interval_calculation(simulated_sold_price_subsample)
            ci_lower_bound.append(lower_ci)
            ci_upper_bound.append(upper_ci)
            means.append(mean)

    results_display(means, ci_upper_bound, ci_lower_bound, \
                    simulated_rental_sample, generated_square_feet_sample)

    return means, upper_ci, lower_ci

def results_display(means, ci_upper_bound, ci_lower_bound, \
                    simulated_rental_sample, generated_square_feet_sample):

    generated_square_feet_sample = np.sort(generated_square_feet_sample)
    simulated_rental_sample = np.sort(simulated_rental_sample)
    means = np.sort(means)
    ci_lower_bound = np.sort(ci_lower_bound)
    ci_upper_bound = np.sort(ci_upper_bound)

    plt.plot(generated_square_feet_sample, means, label='Mean')
    plt.ylabel('Simulated Sold Price')
    plt.xlabel('Square Feet')

    plt.plot(generated_square_feet_sample, ci_lower_bound, label='Lower Bound')
    plt.plot(generated_square_feet_sample, ci_upper_bound, label='Upper Bound')
    plt.legend()
    plt.title('Confidence Intervals')
    plt.show()

    plt.plot(simulated_rental_sample, means, label='Mean')
    plt.ylabel('Simulated Sold Price')
    plt.xlabel('Rental')

    plt.plot(simulated_rental_sample, ci_lower_bound, label='Lower Bound')
    plt.plot(simulated_rental_sample, ci_upper_bound, label='Upper Bound')
    plt.legend()
    plt.title('Confidence Intervals')
    plt.show()


df = pd.read_excel('model_data.xlsx').set_index('MLS')
df = df[['Sold_Price', 'Area_Size', 'Rental_Month', 'Lottery_Dummy', \
        'Occupation_Dummy', 'Days_Open_5', 'Size_Franchise_Order2']]

df_x = df[['Area_Size', 'Rental_Month', 'Lottery_Dummy', \
            'Occupation_Dummy', 'Days_Open_5', 'Size_Franchise_Order2']]

observation_choosen = df_x.loc['W1359413']

area_size_mean = np.mean(df['Area_Size'])
ss_area_size = calculate_sum_of_squares(df['Area_Size'])
size = len(df['Area_Size'])
area_size_std = math.sqrt(ss_area_size/(size - 1))

rental_mean = np.mean(df['Rental_Month'])
ss_rental = calculate_sum_of_squares(df['Rental_Month'])
rental_std = math.sqrt(ss_rental/(size - 1))

regression_rental_size = linear_regression(df_x['Rental_Month'], df_x['Area_Size'])

size = len(df['Area_Size'])
constant_coef_mean = regression_rental_size.params[0]
area_size_coef_mean = regression_rental_size.params[1]
constant_coef_std = regression_rental_size.bse[0]
area_size_coef_std = regression_rental_size.bse[1]

full_regression = linear_regression(df['Sold_Price'], df_x)
full_coefs = full_regression.params
full_coefs_std_error = full_regression.bse


Simulation_Results(observation_choosen, area_size_mean, area_size_std, constant_coef_mean, constant_coef_std, \
                    area_size_coef_mean, area_size_coef_std, \
                        full_coefs, full_coefs_std_error)