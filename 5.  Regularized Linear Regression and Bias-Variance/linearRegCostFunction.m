function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

Hx = X * theta;
J = (((Hx-y)') * (Hx-y)) / (2 * m) + (lambda/(2*m)) * (theta(2:end)'*theta(2:end));

% Calculate the gradient w.r.t theta(j)
% Do not regularize the parameter theta(1)
grad(1) = ((Hx-y)' * X(:,1)) / m;

% Regularize the parameter theta(2):theat(n+1)
for i = 2:size(theta)
    grad(i) = ((Hx-y)' * X(:,i)) / m + (lambda * theta(i)) / m;
end

% =========================================================================

grad = grad(:);

end
