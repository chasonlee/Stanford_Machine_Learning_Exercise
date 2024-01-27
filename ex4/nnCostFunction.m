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

Y = zeros(size(y, 1),size(Theta2, 1));
for i = 1:m
    Y(i, y(i)) = 1;
end

ReTheta1 = Theta1;
ReTheta1(:, 1) = 0;
ReTheta2 = Theta2;
ReTheta2(:, 1) = 0;

J = sum(1/m * sum(-Y .* log(sigmoid([ones(m, 1) sigmoid([ones(m, 1) X] * Theta1')] * Theta2')) - ...
         (1 - Y) .* log(1 - sigmoid([ones(m, 1) sigmoid([ones(m, 1) X] * Theta1')] * Theta2')))) + ...
         lambda / (2 * m) * (sum(sum(ReTheta1 .^ 2)) + sum(sum(ReTheta2 .^ 2)));

% -------------------------------------------------------------

a_1 = [ones(m, 1) X];
z_2 = a_1 * Theta1';
a_2 = [ones(m, 1) sigmoid(z_2)];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

det_3 = a_3 - Y;
det_2 = det_3 * Theta2;
det_2 = det_2(:,2:end) .* sigmoidGradient(z_2);

Theta1_grad = det_2' * a_1 / m;
Theta2_grad = det_3' * a_2 / m;

% Implement regularization
ReTerm1 = ones(size(Theta1_grad));
ReTerm2 = ones(size(Theta2_grad));
ReTerm1(:, 1) = 0;
ReTerm2(:, 1) = 0;
ReTerm1(:, 2:end) = ReTerm1(:, 2:end) * lambda / m .* Theta1(:, 2:end);
ReTerm2(:, 2:end) = ReTerm2(:, 2:end) * lambda / m .* Theta2(:, 2:end);
Theta1_grad = Theta1_grad + ReTerm1;
Theta2_grad = Theta2_grad + ReTerm2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
