# Machine-Learning--Andrew-Ng
The projects in online course: Machine Learning from Andrew Ng

# Project 1: Linear Regression
The first project is for Linear Regression is to implement linear regression with multiple variables to predict the prices of houses.
The file ex1data2.txt contains a training set of housing prices in Portland, Oregon. 
The first column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house.
We use two different methods: the gradient descent method and the numerial equation and compare their performances.

Both of the two methods can calculate the optimal theta from linear regression which can match the dataset best.
In this project, we get totally the same result ($293081.464335 predicted price when it's a 1650 sq-ft, 3 br house) based on this two methods.

The numerial equation gives the optimal theta in analysis.The equation in detail is : theta = pinv(X' * X) * X' * Y, which is very easy to execute and do not 
need to do any feature scaling or finding the "best" learning rate and num_iterations. But if the num of features getting larger (> 10^4), the "pinv" pesudo inverse
may cost a lot.In this situation, the gradient descent method would be a better choice.

In the gradient descent method, to improve the convergence of the gradient, we use feature scaling to normalize the dataset.So for new features, we need to first normalize then 
get the predicted price. We find the "best" learning rate by plotting the figures of cost function w.r.t num_iterations of different learning rate. We try from the lr = [0.01 0.03 0.1 0.3] and num_iteration = 50 situation. Finally find the ideal parameter is lr/alpha = 0.2, num_iteration = 300.

Improvement:
In this setting, the predicted value produced by this two methods are totally the same. However, if we want to save more expenses, we can just set num_iteration = 200.
This may lead to errors of only 0.002, which I think is still acceptable.

