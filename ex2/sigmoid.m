function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

tmp1 = ones(size(z));
tmp1 = tmp1 * exp(1);
tmp1 = tmp1 .^ -z;
tmp1 = 1./(tmp1 + 1);
g = tmp1;

% =============================================================

end
