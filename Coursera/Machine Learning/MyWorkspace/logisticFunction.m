function [ G ] = logisticFunction( Theta, X )
%Returns the sigmoid or logistic function given the theta
%parameters and the dataset X

%Theta is a matrix with dim p x n, n 
%X is the training data matrix, with size m, n

Z = 1 + exp(-(Theta * X'));

G = 1./Z;

end

