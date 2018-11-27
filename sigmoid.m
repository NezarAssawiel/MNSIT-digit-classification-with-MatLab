function g = sigmoid(z)
 
% computes the sigmoid of z and returns it as g.
 
g = 1.0 ./ (1.0 + exp(-z));
end 
