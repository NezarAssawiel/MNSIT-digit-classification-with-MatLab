function [J, grad] = nn_cost_function(nn_params,input_layer_size, hidden_layer_size, ...
                                   num_of_classes, X, y, lambda)                                  
% this function implements the neural network cost function for a two layer
%classification neural network and returns "grad": unrolled vector of the partial derivatives 
% of the neural network.
   
 
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for the 2 layers
 
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_of_classes, (hidden_layer_size + 1));
 
% get number of exmaples in traning data
m = size(X, 1);
 
% pre-allocate         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
 
 
% add bias terms
X = [ones(m, 1) X];
 
% apply activation function
a2= sigmoid(X*Theta1'); a2=[ones(size(a2,1),1) a2];
a3=sigmoid (a2*Theta2');
 
 
%Pre-allocate and perform One Hot Encoding (notice that true labels are in "y"
%while calucuated labels are in "Y"
Y=zeros(m, num_of_classes);
 
rows=1:m; cols= y';
Y(sub2ind(size(Y),rows,cols))=1;
 
% alternatively, one can do the following:  
%[~, loc] = ismember(y, unique(y));
% Y = ind2vec(loc')';
 
%% calculate cost function
J=(1/m)*sum(sum((-Y.*log(a3))-(1 - Y).*log(1-a3)));
RegTerm=(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+ sum(sum(Theta2(:,2:end).^2)));
J=J + RegTerm;
 
%% un-regulized gradient for Theta1 and Theta2
delta3 = a3 - Y;
delta2=delta3*Theta2(:,2:end).*sigmoid_gradient(X*Theta1');
 
D1=delta2'*X;   Theta1_grad=D1/m; 
D2=delta3'*a2;  Theta2_grad=D2/m;
 
%% add regulization 
Theta1(:,1)=0; Theta1=(lambda/m)*Theta1;
Theta2(:,1)=0; Theta2=(lambda/m)*Theta2;
 
Theta1_grad= Theta1_grad + Theta1 ;
Theta2_grad= Theta2_grad + Theta2 ;
 
 
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
  
end
