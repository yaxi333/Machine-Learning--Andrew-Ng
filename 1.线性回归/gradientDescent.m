function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% J_history is the cost vector w.r.t the iteration of num_iters times


% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    n = size(X,2); % Since X has been add the first column exactly its n = n+1 here 
    temp = zeros(n , 1);
    Hx = X * theta;
    for j = 1:n
        temp(j) = theta(j) - ((Hx - y)' * X(:,j)) * (alpha / m); 
    end
    theta = temp;    
    % temp is set for applying the simultaneously update of theta

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
