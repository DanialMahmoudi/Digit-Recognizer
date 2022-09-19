% ================= Digit Recognizer =================
%
% Welcome to the "Digit Recognizer" program
%
% This program allows you to anticipate and recognize the 0-9 digits from 
% gray scale pictures (20 * 20 pixels) using a 3-layer neural network
% 
% I hope you enjoy...
%
% ================= Chapter 1: Displaying The Data =================
%
% In this chapter we load the "Input.mat" which includes our pixels and desired anticipation into our program

clear; close all; clc

input_layer_size = 400;
hidden_layer_size = 25;
output_layer_size = 10;

fprintf('Loading and Visualizing Data ...\n')
load("Input.mat");
m = size(X, 1); % Number of our training set (5000 different pictures)

% We randomly select 100 different pictures to demonstrate using the "displayData() finction"
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

% ================= Chapter 2: Load & initialize neural network parameters (Theta1 & Theta2 AKA Weights) =================
%
% In this chapter first, we load some pre-initialized parameters into Theta1 &
% Theta2 matrices then we unroll these to matrices into a column vector & then
% use "random_initialization" function to randomly initialize these two matrices 

fprintf('\nLoading Saved Neural Network Parameters ...\n')

load("Weights.mat"); % We load "Weights.mat" into Theta1 & Theta2 Matrices which have (25 * 401 = 10025 parameters) & (10 * 26 = 260 parameters) accordingly

fprintf('\nInitializing weights with random numbers with respects to epsilon value ...\n')

unrolled_parameters = [Theta1(:); Theta2(:)]; % Here we vectorize Theta1 & Theta2 matrices into a column vector which has (10285 * 1) parameters 

unrolled_parameters = random_initialization(unrolled_parameters, input_layer_size, hidden_layer_size, output_layer_size); % We use "random_initialization" function to randomly initialize our weights

fprintf('\nVisualizing the first ten elements from Theta1 & Theta2 ...\n')

fprintf('\n Theta1\t\t\t\tTheta2\n\n')
for j = 1:10
  k = 10025 + j;
  fprintf(strcat(num2str(unrolled_parameters(j, 1)), '\t\t\t', num2str(unrolled_parameters(k, 1)), '\n\n'))
  end

fprintf('Program paused. Press enter to continue.\n');
pause;

% ================= Chapter 3: Forward Propagation =================
% 
% In this chapter we perform forward propagation inside our "Forward_Propagation" function in order to
% calculate all activation units in each layer.
% Note that each activation unit is our hypothesis function and we calculate them in
% vectorized form and use them to calculate our cost function later in the next chapter

output_layer_A_units = Forward_Propagation(unrolled_parameters, X, input_layer_size, hidden_layer_size, output_layer_size);

fprintf('\nVisualizing the first ten elements from A3 for each 10 classes for first training example ...\n\n')
for j = 1:10
  fprintf(strcat('Class\t', num2str(j), '\t===>\t', num2str(output_layer_A_units(j, 1)), '\n\n'))
  end

fprintf('Program paused. Press enter to continue.\n\n');
pause;

% ================= Chapter 4: Cost Function & Back Propagation =================
%
% Now we use output layer activation units (A3) to compute cost function and 
% partial derivaties for thetas with back propagation algorithm with 
% "cost_function" to use it for debugging and advanced optimization algorithms 
% later on the next chapters

J = cost_function(unrolled_parameters, X, y, input_layer_size, hidden_layer_size, output_layer_size);

fprintf('Our computed cost function with initialized thetas without regularization is ...\n\n');
J
fprintf('\nThe goal is to minimize cost function as much as possible with algorithms like gradient descent or more advanced algorithms\n\n')
fprintf('Program paused. Press enter to continue.\n\n');
pause;

% ================= Chapter 5: Back Propagation & Delta terms =================
%
% In this chapter we use back propagation algorithm in our "Back_Prpagation"
% function to calculate delta terms and use them to calculate Delta and 
% partial derivaties to be able to use them to obtain best theta values with
% helps of advanced optimization algorithms

pds = back_propagation(unrolled_parameters, X, y, input_layer_size, hidden_layer_size, output_layer_size);

fprintf("\nVisualizing the first ten elements from Theta1 & Theta2's partial derivaties ...\n")

fprintf('\n Theta1 partial derivaties\t\t\t\tTheta2 partial derivaties\n\n')
for j = 1:10
  k = 10025 + j;
  fprintf(strcat(num2str(unrolled_parameters(j, 1)), '\t\t\t\t\t\t\t', num2str(unrolled_parameters(k, 1)), '\n\n'))
  end

fprintf('Program paused. Press enter to continue.\n');
pause;

% ================= Chapter 6: Training Neural Network =================
%
% In this chapter we use more advanced and sophisticated algorithms to find the 
% best thetas for our training set to be able to learn and anticipate further 
% data
% Note that here we use "fminunc" function to calculate best thetas over 50 itertions
% with respect to the current cost function and thetas

options = optimset('MaxIter', 50);
cost = @(p) cost_function(p, X, y, input_layer_size, hidden_layer_size, output_layer_size);
[final_params, final_cost] = fmincg(cost, unrolled_parameters, options);

final_Theta1 = reshape(final_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, input_layer_size + 1);
final_Theta2 = reshape(final_params(hidden_layer_size * (input_layer_size + 1) + 1: end), output_layer_size, hidden_layer_size + 1);

fprintf('\nVisualizing the first ten elements from our final Theta1 & Theta2 ...\n')

fprintf('\n final_Theta1\t\t\t\tfinal_Theta2\n\n')
for j = 1:10
  k = 10025 + j;
  fprintf(strcat(num2str(final_params(j, 1)), '\t\t\t', num2str(final_params(k, 1)), '\n\n'))
end

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nVisualizing our final calculated cost function after 50 iterations ...\n')
final_cost(size(final_cost, 1))
% Now that we have calculated our cost function over 50 iterations and also obtained
% the best values for theta we can use these data to debug our algorithm and 
% anticipate other examples

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% ================= Chapter 7: Debugging algorithm using cost function =================
%
% In this chapter we plot our cost function values calculated in each iterations
% using "Plot" function to be able to debug our algorithm and assess its efficiency.

fprintf('\nVisualizing the cost per iteration plot ...\n')
itr = [1:50];
figure(2);
plot(itr, final_cost, 'r', 'linewidth', 3);
xlabel('Number of Iterations');
ylabel('Cost function value');
title('Cost per iteration');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


% ================= Chapter 8: prediction and efficiency =================
%
% In this chapter we use our calculated thetas in order to predict our training 
% example's class and compare them to their real class in order to be able to
% evaluate our algorithm's efficiency

fprintf('\nCalculating the training set accuracy ...\n')

pred = predict(final_Theta1, final_Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

% ================= Chapter 9: Working with neural network =================
%
% In this chapter we give you the wheel to be able to enjoy working with the 
% training set and scrutinize them separately and see how does our program work
% you can continue seeing more pictures and their anticipation if you press 
% "Enter" and you can quit whenever you want with pressing "q" button on the 
% keyboard
% Note that the flawlessness of prediction is related to accuracy
% Note that the pred returns index of class which is equal to "10" for 0s

rp = randperm(m);

fprintf('\nInitializing the prediction process ...\n')
for i = 1:m
    
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(final_Theta1, final_Theta2, X(rp(i),:));
    if pred == 10
      pred = 0
    endif
    fprintf('\nNeural Network Prediction is : %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press Enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

