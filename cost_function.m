function [J grad] = cost_function(params, ts, y, inp, hid, out)

% This functions uses output layer activation units computed with forward 
% propagation and our true must-be-anticipated class values for each training
% example to cost cost function and apply back propagation algorithm on each 
% training example

theta1 = reshape(params(1:(hid * (inp + 1))), hid, (inp + 1));
theta2 = reshape(params((1 + (hid * (inp + 1))):end), out, (hid + 1));
X = ts;
m = size(X, 1);
X = [ones(m, 1), X];
J = 0;
theta1_grad = zeros(size(theta1));
theta2_grad = zeros(size(theta2));
total = 0;

A2 = [ones(1, m); sigmoid(theta1 * X')];
A3 = sigmoid(theta2 * A2);
for k = 1:out
  tmp = (y == k);
  total += (((-1 * tmp)' * log(A3(k,:))') - (((1 .- tmp)') * log(1 .- A3(k,:))')) ./ m;
end
J = total;

Delta1 = zeros(size(theta1));
Delta2 = zeros(size(theta2));
for i = 1:m
  a1 = (X(i, :))';
  z2 = theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];
  z3 = theta2 * a2;
  a3 = sigmoid(z3);
  delta3 = zeros(out, 1);
  delta2 = zeros(hid + 1, 1);
  tmp_y = zeros(out, 1);
  for k = 1:out
    if k == y(i)
      tmp_y(k, 1) = 1;
    else
      tmp_y(k, 1) = 0;
    endif
  endfor
  delta3 = a3 - tmp_y;
  delta2 = (theta2' * delta3) .* (a2 .* (1 .- a2));
  Delta1 = Delta1 + (delta2(2:end) * (a1)');                %Remove delta(l)0's from hidden layers to fit into Deltas and Partial derivatives
  Delta2 = Delta2 + (delta3 * (a2)');
end
theta1_grad = Delta1 ./ m;
theta2_grad = Delta2 ./ m;

grad = [theta1_grad(:); theta2_grad(:)];