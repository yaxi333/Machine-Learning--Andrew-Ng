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

# Project 2: Logistic Regression
The second project is to solve the simple classification problems based on Logistic Regression to implement on two different dataset. 
One for linear decision boundary and the other one for nonlinear decision boundary.

The first task is to build a logistic regression model to predict whether a student gets admitted into a university. And you have the dataset of the previous years applicant’s chance of admission based on their results on two exams. So we use this to build our logistic regression model.
We generally first plot the 2D figures of the training set(in this case, we only have 2 features in both datasets). Based on the plot, we can get a general idea of this datdaset is that our hypothesis is a line. So we do not need to add our features to build an nonlinear classifier model.
We use advanced optimizaton method (based on fminunc() function) rather than gradient descent method. This codes for this medthod is as follows:

% First build the costFunction to calculate the cost and gradients w.r.t each features(theta)
function [jVal, gradient] = costFunction(theta, X, y)
         jVal = [code to compute the cost function J(theta)];
         gradient(1) = [code to compute the gradient for theta(0)];
         gradient(2) = [code to compute the gradient for theta(1)];
         ......
         ......
         gradient(n+1) = [code to compute the gradient for theta(n)];
end

% In the main function, we call the costFunction
% 'GradObj','on' : provide gradient to this algorithm
% 'MaxIter', 100 : the maximum iteration times 100
% optTheta : the final optimal theta we finally find
% functionVal : the final cost w.r.t the opitimal value
% exitFlag : verify wether the function has converged
% initialTheta: pay attention that the dimension should >= 2

options = optimset('GradObj','on','MaxIter', 100);
initialTheta = zeros(n+1,1);
[optTheta, functionVal(costVal), exitFlag] = fminunc(@(t)costFunction, initialTheta, options);

After finding the optimial theta, we can draw the decision boundary for the plotted data figures.

And we can use this theta to predict any new applicants. The idea is that we need to first calculate the hypothesis (Attention: this value should be sent into sigmoid function, which is the point different from linear regression). Then from the figure of sigmoid function, we know that if h>=0.5, then it shoulf classified as label 1(the positive one).

The second task is to classify a dataset which is not linear. Thus we need to add more features to build a more complex hypothesis model. So we add the feature of [1, x_1,x_2,x_1*x_2,x_1^2,x_2^2,x_1^3,x_2^3,x_1^2*x_2......x_1^6,x_2^6]
More feature will easier to cause the problem of overfitting, so we need to use the idea of regulization to reduce the value of each theta by adding the regulization parameter:lamda.

After experiment, we conclude that if the lamda is too small, then it may lead to overfitting; if it is too large, it will underfitting, which means even not fitting the training dataset.
