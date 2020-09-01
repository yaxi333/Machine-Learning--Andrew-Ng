function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = X * theta;
h = sigmoid(z);

% Calculate the cost function J(theta)
J = (-y' * log(h)-(1-y)' * log(1-h))/m   +  (lambda/(2*m)) * (theta(2:end)'*theta(2:end));

% Calculate the gradient w.r.t theta(j)
% Do not regularize the parameter theta(1)
grad(1) = ((h-y)' * X(:,1)) / m;

% Regularize the parameter theta(2):theat(n+1)
for i = 2:size(theta)
    grad(i) = ((h-y)' * X(:,i)) / m + (lambda * theta(i))/m;
end


% =============================================================

end