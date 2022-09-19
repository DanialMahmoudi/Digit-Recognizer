function g = sigmoid(z)
  
% SIGMOID Compute sigmoid function
% G = SIGMOID(z) computes the sigmoid of z.
% This funtions computes the sigmoid function which is used in neural networks 
% and logistic regressions

g = 1.0 ./ (1.0 + exp(-z));
end