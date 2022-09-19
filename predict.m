function p = predict(Theta1, Theta2, X)

% In this function we use final thetas to firstly, calculate the accuracy of our 
% algorithm and its efficiency and secondly, use random training example and 
% predict their class in action

m = size(X, 1);
out = size(Theta2, 1);
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
z2 = Theta1 * X';
a2 = sigmoid(z2);
a2 = [ones(1, m); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
a3 = a3';
[ma, in] = max(a3, [], 2);
p = in;
end