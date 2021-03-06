function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));



% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% compute hypothesis using the FWD propagation algorithm

X = [ones(size(X,1),1), X]; %add one column of ones (bias)
a2 = sigmoid(X * Theta1'); % activation values at layer 2
a2 = [ones(size(a2,1),1), a2]; %add one column of ones (bias)
a3 = sigmoid(a2 * Theta2'); % activation values at layer 3, output layer

% we need to transform y in a m x k matrix instead m x 1
y_matrix = zeros(m, size(a3,2));

for i = 1:m
    
    y_matrix(i,y(i)) = 1;
    
    for k = 1:size(y_matrix,2)
    
      J = J + ( (-y_matrix(i,k) * log(a3(i,k)) ) - ((1-y_matrix(i,k)) * log(1-a3(i,k))) )/m;
      
    end
end

% Let's add the regularization to the cost function J

J = J + (lambda/(2*m)) * (sum(sum(Theta1(:,2:size(Theta1,2)) .^2)) + sum(sum(Theta2(:,2:size(Theta2,2)) .^2)));

% -------------------------------------------------------------

% =========================================================================

% Backpropagation algorithm

Delta_2 = 0;
Delta_1 = 0;
%iterate over each single training set
for t = 1:m
    
    % Set input layers values a_1 to the t-th training example x
    a_1 = X(t,:); % is alread biased
    
    %start the feed forward algorithm
    z_2 = a_1 * Theta1';
    % apply the sigmoid function
    a_2 = sigmoid(z_2);
    % add the bias element
    a_2 = [1 a_2];
    z_2 = [0 z_2];
    % compute z at layer 3
    z_3 = a_2 * Theta2';
    % apply the sigmoid function to compute the activation
    a_3 = sigmoid(z_3);
    
    % Compute errors 
    delta_3 = a_3 - y_matrix(t,:);
      
    % hidden layer
    delta_2 = (delta_3 * Theta2) .* sigmoidGradient(z_2);
    
    % remove the zero element 
    delta_2 = delta_2(2:end);
    %delta_3 = delta_3(2:end);
    
    Delta_2 = Delta_2 + (a_2' * delta_3);
    
    Delta_1 = Delta_1 + (a_1' * delta_2);
    
end

Theta1_grad = (Delta_1' + lambda * Theta1)/m;
Theta1_grad(:,1) = (Delta_1(1,:)')/m;

Theta2_grad = (Delta_2' + lambda * Theta2)/m;
Theta2_grad(:,1) = (Delta_2(1,:)')/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
