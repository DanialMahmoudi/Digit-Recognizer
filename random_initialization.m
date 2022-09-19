function W = random_initialization(params, inp, hid, out)

% In this function we first receive unrolled parameters as a vector the we reshape 
% them into 2 matrices (theta1 & theta2) in order to initialize them randomly &
% then we use two values (e1 & e2) to set some random parameters to them based on 
% their corresponding layer activation units

tmp = params;
theta1 = reshape(tmp(1:hid * (inp + 1)), hid, inp + 1);
theta2 = reshape(tmp(hid * (inp + 1) + 1: end), out, hid + 1);

% We use the following formula to calculate e1 & e2

e1 = sqrt(6) / sqrt(hid + inp + 1);
e2 = sqrt(6) / sqrt(out + hid + 1);

% We use following formula to initialize theta1 & theta2

theta1 = rand(hid, inp + 1) * 2 * e1 - e1;
theta2 = rand(out, hid + 1) * 2 * e2 - e2;

% Now we unroll them into "W" vector

W = [theta1(:); theta2(:)];