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

X = [ones(m,1) X];
const = ones(m, 1);
i = size(Theta2, 1);
Y = repmat([1:i], size(y, 1), 1);
for i = 1:m
  Y(i,:) = Y(i,:) == y(i);
endfor
Z_2 = X*Theta1';
A = sigmoid(Z_2);
A = [ones(size(A,1) , 1) A];
H = sigmoid(A*Theta2');
J = -(1/m)*((Y.*log(H) + (1-Y).*log(1-H))'*const);
const = ones(size(J));
J = J'*const;
Theta1_z = Theta1;
Theta2_z = Theta2;
Theta1_z(:,1) = zeros(size(Theta1,1),1);
Theta2_z(:,1) = zeros(size(Theta2,1),1);
Theta1_sq = Theta1_z*Theta1_z';
Theta1_sq = Theta1_sq.*eye(size(Theta1_sq));
Theta2_sq = Theta2_z*Theta2_z';
Theta2_sq = Theta2_sq.*eye(size(Theta2_sq));
Theta2_sum = (Theta2_sq*ones(size(Theta2_sq,1),1))'*(ones(size(Theta2_sq,1),1));
Theta1_sum = (Theta1_sq*ones(size(Theta1_sq,1),1))'*(ones(size(Theta1_sq,1),1));

J = J + (lambda/(2*m))*(Theta1_sum + Theta2_sum);
error = H - Y;


del_3 = H - Y;
del_2 = (del_3*Theta2);
del_2 = del_2(:, [2:end]).*sigmoidGradient(Z_2);

for i = 1:m
  z_2 = Theta1*X(i,:)';
  del_3 = (H(i, :) - Y(i, :))';
  del_2 = Theta2'*del_3;
  del_2 = del_2([2:end]);
  del_2 = (del_2).*[sigmoidGradient(z_2)];
  Theta1_grad = Theta1_grad + (del_2)*X(i,:);
  Theta2_grad = Theta2_grad + (del_3)*([1; sigmoid(z_2)]');
endfor
Theta1_grad = Theta1_grad/m + (lambda/m)*(Theta1_z);
Theta2_grad = Theta2_grad/m + (lambda/m)*(Theta2_z);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
