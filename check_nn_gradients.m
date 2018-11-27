function check_nn_gradients(lambda)
 
%   check_nn_gradients(lambda) creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using cal_numeric_gradient). These two gradient computations should
%   result in very similar values.
%
 
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end
 
input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;
 
% generate test data 
Theta1 = initial_weights(hidden_layer_size, input_layer_size);
Theta2 = initial_weights(num_labels, hidden_layer_size);
 
% use initial_weights to generate X
X  = initial_weights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';
 
% unroll
nn_params = [Theta1(:) ; Theta2(:)];
 
% Short hand for cost function
cost_fun = @(p) nn_cost_function(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);
 
[cost, grad] = cost_fun(nn_params);
numgrad = cal_numeric_gradient(cost_fun, nn_params);
 
% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
 
% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in cal_numeric_gradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);
 
fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);
 
End


function W = initial_weights(lyr_out, lyr_in)
% Initialize the weights of a layer with lyr_in
%incoming connections and lyr_out outgoing connections using a fixed
%strategy
 
% Set W to zeros
W = zeros(lyr_out, 1 + lyr_in);
 
% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10; 
end
