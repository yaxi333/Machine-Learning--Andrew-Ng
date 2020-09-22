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

# Project 3:  Multi-class Classification and Neural Networks
In this project, we implement one-vs-all logistic regression and neural networks to recognize hand-written digits (from 0 to 9). We use multiple one-vs-all logistic regression models to build a multi-class classifier. Since there are 10 classes, we trained 10 separate logistic regression classifiers in the vectorizing form which means that we do not employ any for loops. In this task we use fmincg() function to do optimization, which is more efficient for dealing with a large number of parameters than fminunc(). Finaly, the output is a probability of each class, we choose the maximum one of them as our final predicted label.

As logistic regression cannot form more complex hypotheses as it is only a linear classifier, we introduce another method : The neural network that is able to represent complex models that form non-linear hypotheses. And in this method, we do not need to do optimization, instead we use the Feedforward Propagation Medthod.

We need to pay attention that since we do prediction of 5000 examples at the same time. So we need to use the vectorizing method. Transfer all this examples and their features at the same time in the form of Matrix. And also do the Feedforward Propagation at the same time. In each layer we need to transfer our results through the activation function to get the probability. Finally, in the output layer, we need to choose the class with the max probability as the class of the corresponding example.


# Project 4:

# Project 5: Regularized Linear Regression and Bias v.s. Variance
We use regularized linear regression and use it to study models with different bias-variance properties. In the first half of the exercise, we implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir. In the next half, we go through some diagnostics of debugging learning algorithms and examine the effects of bias v.s. variance.

We know that an important concept in machine learning is the bias-variance tradeoff. Models with high bias are not complex enough for the data and tend to underfit,
while models with high variance overfit to the training data. And we often plot training and cross-validation errors on a learning curve to diagnose bias-variance problems.

To plot the learning curve, we first trained the model with lambda = 0/ lambda = some values. Then we can get the optimal theta parameters. We then test this well-trained model on the training set and validation set. Calculate the corresponding errors. And we repeat this based on different number of training examples(the first 1:i:m training examples in training set). Pay attention that to calculate the J_train and J_val also later J_test, we can not add the regulation factor, since what we waht to get is the practical/ true error. And we calculate the error on the selected subset of the training set.

After plot the learning curve,we can know it's high bias or variance problem. If both the train error and cross validation error are high when the number of training examples is increased. This reflects a high bias problem in the model – the linear regression model is too simple and is unable to fit our dataset well. We address this problem by adding more features. Here we use the polynomial regression to fit the data which means adding features of [x x^2 x^3 x^4 x^5 x^6 x^7 x^8]. Pay attention that in this case as the features would be badly scaled, we need to use feature normalization.

If there is a gap between the training and cross validation errors, indicating a high variance problem. One way to combat the overfitting (high-variance) problem is to add regularization to the model. We can select λ(regularization factor) by using a cross validation set. That is to plot a cross validation curve of error v.s. λ that allows you select which λ parameter to use. Remember it is important to evaluate the “final” model on a test set that was not used in any part of training (that is, it was neither used to select the λ parameters, nor to learn the model parameters θ).

Finally we also tried to Plotting learning curves with randomly selected examples.
