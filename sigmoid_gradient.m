function g = sigmoid_gradient(z)
% computes the gradient of the sigmoid function
 
G=1./(1+ exp(-z));
 
g=G.*(1-G);
 
end
