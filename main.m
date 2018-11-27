%% Machine Learning Case: Deep Neural Network from First Principles 
% Written By: Nezar Assawiel
%--------------------------- 
 
% This is a 3 layer, fully connected neural network used for hand-written 
% digit recognition. Evaluated against MNIST
 
 
%% Set neural network parameters
 
clear; close all; clc
 
input_lyr_size  = 400;  % 20x20 Input Images of Digits
hid_lyr_size = 25;      % 25 hidden layers
out_lyr_size = 10;      % 10 layers (or labels) from 1 to 10. "0" was mapped to "10" and "1-9" were mapped
                        % to "1-9"
 
 
%% Load training data & (visualize it)
 
% load training data which has X and y variables as the data and labels
load('MNIST.mat');
X=train;
X_test=test;
y=ytrain;
y_test=ytest;

 
m=size(X,1);

% Select 100 random data examples to see
sel1 = randperm(m);
sel = sel1(1:100);
 
display_data(X(sel, :));
 
 
% shuffle data randomly
X= X(sel1,:);
y= y(sel1);
fprintf('Paused. To see 100 training examples, see the figure. Press enter to continue...\n');
pause;
 
 
%% Check correctness of implementation
% code of check_nn_gradients checks numerical gradient against analytical gradient
% for samll neural network and prints the results 
 
fprintf('\n Checking validity of backpropagation implementation...\n')
 
lambda = 2;
check_nn_gradients(lambda);
 
 
%% Randomly initialize parameters
 
initial_theta1 = rand_initialize_weights(input_lyr_size, hid_lyr_size);
initial_theta2 = rand_initialize_weights(hid_lyr_size, out_lyr_size);
 
% Unroll parameters
initial_nn_params = [initial_theta1(:); initial_theta2(:)];
 
 
 
 
%% Training the network
 
fprintf('\n Training the Neural Network...\n')
 
% Regulization parameter
lambda = 1;
 
% Shorer notation for the cost function 
cost_function = @(p) nn_cost_function(p, input_lyr_size,  hid_lyr_size, out_lyr_size, X, y, lambda);
 
% Get cost function                                   
options = optimset('MaxIter', 100);
[nn_params, cost] = fmincg(cost_function, initial_nn_params, options);
 
% Obtain theta1 and theta2 back from reshaping nn_params
 
theta1 = reshape(nn_params(1:hid_lyr_size * (input_lyr_size + 1)),  hid_lyr_size, (input_lyr_size + 1));                
theta2 = reshape(nn_params((1 + (hid_lyr_size * (input_lyr_size + 1))):end), out_lyr_size, (hid_lyr_size + 1));
                 
 
%% See hidden layer of the NN 
 
fprintf('\The hidden layer of the Neural Network is displayed\n')
 
display_data(theta1(:, 2:end));
 
%% Predict the hand-written digits of the training set
% here a test set can be examined as well
 
pred = predict(theta1, theta2, X);
fprintf('\n The Accuracy of the Training Set: %f\n', mean(double(pred == y))*100);

pred = predict(theta1, theta2, X_test);
fprintf('\n The Accuracy of the Test Set: %f\n', mean(double(pred == y_test))*100);



% end of “Machine Learning Case: Neural Network Training”
 
