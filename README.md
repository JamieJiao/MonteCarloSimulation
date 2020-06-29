# Monte Carlo Simulations for OLS
## Simulated y
Choose one observation, keep other variables fixed, with generated coefficient samples (normal distribution) and one generated variable (Area Size) sample (uniform distribution), loop through Area Size sample with each iteration repeating x 100 times, the true relationship is shown between y (Sold Price) and x (Area Size).

![](images/simulated_y_as_x_inreases.png)

## Confidence intervals for simulated y
As the times of x repeatiton increse, from 100 to 10000, the simulated confidence intervals (95%) tend to have less fluctuation and are more close to straight lines (which are the true confidence intervals for the simulated y). In each iteration, the simulated y sample size gets bigger and the simulated y distribution gets more close to the true distribution.

![](images/confidence_interval_x_repeat_100.png)

![](images/confidence_interval_x_repeat_1000.png)

![](images/confidence_interval_x_repeat_10000.png)

## Confidence intervals when there is a strong correlation among two variables

Predictor Area Size and Rental have a significant correlation.
The true relationships between y and X are violated. And simulated y distribution is no more normal distribution.

![](images/simulated_y_as_area_size_increases.png)

![](images/simulated_y_as_rental_increases.png)

Also, with the presence of high multicollinearity, the confidence intervals tend to become wide as predictors have less power to explain the response variable. And the confidence intervals are not paralleled with the mean values.

![](images/CI_x_repeat_100_area_size.png)

![](images/CI_x_repeat_10000_area_size.png)

![](images/CI_x_repeat_100_rental.png)

![](images/CI_x_repeat_10000_rental.png)
