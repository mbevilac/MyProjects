function [ G ] = logisticFunction( theta, X )
%Returns the sigmoid or logistic function given the theta
%parameters and the dataset X

%theta is a 1 col, n rows vector
%X is the parameters matrix, with size m, n

Z = X * theta;

Z = 1 + exp(-Z);

G = 1./Z;

end

