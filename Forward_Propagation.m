function A3 = Forward_Propagation(params, ts, inp, hid, out)

% This function runs feed forward AKA Forward_Propagation algorithm in order toc
% compute activation units in each layer in a vectorized form in order to compute
% Activation units in output layer consequently for us to ba able to use them to
% compute cost function later in the next chapter.
% Note that the activation units are our hypothesis funtions and to compute the 
% cost function we need our hypothesis's output in last (output) layer (A3)

tmp = params;
theta1 = reshape(tmp(1:hid * (inp + 1)), hid, inp + 1);
theta2 = reshape(tmp(hid * (inp + 1) + 1: end), out, hid + 1);
X = ts;
m = size(ts);
X = [ones(m, 1) X];
A2 = [ones(1,m); sigmoid(theta1 * X')];
A3 = sigmoid(theta2 * A2);

end