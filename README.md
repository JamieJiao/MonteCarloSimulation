# Monte Carlo Simulation for Calculating Confidence Intervals with Presence of Multicollinearity in Model

In the model, variables Rental and Square Feet are highly correlated but they both have significant contributions to the response variable Sold Pice.

In order to overcome the multicollinearity issue, linear transformation for Rental is used and all numerical variables (Rental, Square Feet) and coefficients in the model are simulated.

![](images/formulas1.png)
![](images/formulas2.png)

## Confidence Intervals when Rental is correlated to Square Feet

![](images/SoldPrice_vs_Rental_no_simu.png)
![](images/SoldPrice_vs_SquareFeet_no_simu.png)

## Sold Price probability density function

![](images/Density_Function_no_simu.png)

## Simulated Confidence Intervals

![](images/SoldPrice_vs_Rental.png)
![](images/SoldPrice_vs_SquareFeet.png)

## Simulated Sold Price probability density function

![](images/Density_Function.png)

When there is a strong correlation between explanatory variables, the model confidence intervals for the reponse variable tend to be much wider.

After Monte Carlo simulation, the simulated y has much narrower confidence intervals. And the true relationships between y and X are shown.