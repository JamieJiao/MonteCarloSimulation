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

def simulate_rental(square_feet_generated, coefs_mean, coefs_std_error, simualated_var_name='Square_Feet'):
    """
    ----------------------------
    Based on regression between reponse variable, Rental, 
    and explanatory variable, SquareFeet,
    to generate simulated Rental.
    ----------------------------
    """

    rental_simulated = 0
    for i in range(len(coefs_mean)):
        coef_generated = np.random.normal(coefs_mean[i], coefs_std_error[i], 1)
        if coefs_mean.index[i] != simualated_var_name: 
            term = coef_generated * 1
        else:
            term = coef_generated * square_feet_generated
        rental_simulated += term

    return rental_simulated

def simulate_sold_price(observation_choosen, square_feet_posi, \
                        square_feet_generated, rental_simulated, \
                        coefs_mean, coefs_std_error, simualated_var_name='Rental'):
    """
    ----------------------------
    Use a specific observation record,
    based on regression, 
    SoldPrice= β_0+ β_1*SquareFeet+ β_2*Rental+ β_3*LotteryDummy ... + ε_i,
    to simulate Sold Price, y.
    ----------------------------
    """

    sold_price_simulated = 0
    # convert to numpy array
    observation_choosen = np.array(observation_choosen)
    # add constant to the observation
    observation_choosen = np.insert(observation_choosen, 0, 1)
    # replace variable by simulated one
    observation_choosen[square_feet_posi] = square_feet_generated

    for i in range(len(coefs_mean)):
        coef_generated = np.random.normal(coefs_mean[i], coefs_std_error[i], 1)
        # exculde Rental coefficient, use simulated Rental instead
        if coefs_mean.index[i] != simualated_var_name: 
            term = coef_generated * observation_choosen[i]
        else:
            term = coef_generated * rental_simulated
        sold_price_simulated += term
    
    return sold_price_simulated

def sold_price_subsample_generate(observation_choosen, square_feet_generated, \
                                    rental_simulated, full_coefs_mean, full_coefs_std, \
                                    square_feet_posi=1, x_repeat_times=100):
    """
    ----------------------------
    By repeating X, generate a sample of Y, Sold Price.
    
    Parameter:
    square_feet_posi (the position of the Square Feet variable in the observation),
    ----------------------------
    """

    simulated_sold_price_subsample = []
    for repeat in range(x_repeat_times):

        simulated_sold_price = simulate_sold_price(observation_choosen, square_feet_posi, \
                                            square_feet_generated, rental_simulated, \
                                        full_coefs_mean, full_coefs_std)

        simulated_sold_price_subsample.append(simulated_sold_price)
    return simulated_sold_price_subsample

def Simulation_Results(observation_choosen, square_feet_mean, square_feet_std, \
                        coefs_mean, coefs_std, \
                        full_coefs_mean, full_coefs_std, sample_size=1000):
    """
    ----------------------------
    Mean and confidence intervals are generated in each iteration.

    Parameters:
    Observation_choosen (a observation record);
    ...
    coefs_mean (coefficients of all variables in the regression between Rental and Square Feet),
    coefs_std (standard errors of the coefficients in the regression between Rental and Square Feet),
    full_coefs_mean (coefficients of all variables in the Sold Price regression),
    full_coefs_std (standard errors of the coefficients in the Sold Price regression),
    Pandas Series like, from statsmodels.
    
    Return:
    mean values, lower and upper confidence interval.
    ----------------------------
    """

    simulated_rental_sample = []
    generated_square_feet_sample = []
    
    ci_lower_bound = []
    ci_upper_bound = []
    means = []
    for n in range(sample_size):
        # simulate Square Feet
        square_feet_generated = np.random.normal(square_feet_mean, square_feet_std, 1)
        if square_feet_generated <= 0:
            continue
        else:
            generated_square_feet_sample.append(square_feet_generated[0])
            # simulate Rental based on the regression between Rental and Square Feet
            rental_simulated = simulate_rental(square_feet_generated, coefs_mean, coefs_std)
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
    distribution_plot(means)

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
    plt.xlabel('Simulated Square Feet')

    plt.plot(generated_square_feet_sample, ci_lower_bound, label='Lower Bound')
    plt.plot(generated_square_feet_sample, ci_upper_bound, label='Upper Bound')
    plt.legend()
    plt.title('Confidence Intervals')
    plt.show()

    plt.plot(simulated_rental_sample, means, label='Mean')
    plt.ylabel('Simulated Sold Price')
    plt.xlabel('Simulated Rental')

    plt.plot(simulated_rental_sample, ci_lower_bound, label='Lower Bound')
    plt.plot(simulated_rental_sample, ci_upper_bound, label='Upper Bound')
    plt.legend()
    plt.title('Confidence Intervals')
    plt.show()

def distribution_plot(data):
    mu = np.mean(data)
    sigma = np.sqrt(calculate_sum_of_squares(data) / len(data))
    count, bins, ignored = plt.hist(data, 30, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    plt.title('Density Function of Simulated Sold Price Means')
    plt.show()


