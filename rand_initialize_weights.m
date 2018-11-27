function Weights = rand_initialize_weights(L_in, L_out)
 
%   This function randomly initializes the weights 
%   of a neural network layer with incoming connections (L_in) and outgoing 
%   connections (L_out). This is to break the symmetry while training the neural network.
%
 
% Preallocate  
% Note: 1st column of W corresponds to the bias terms
Weights = zeros(L_out, 1 + L_in);
 
epsilon_init = 0.12;
Weights = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
 
end 
