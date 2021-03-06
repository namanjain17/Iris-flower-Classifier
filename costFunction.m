function [J grad] = costFunction(params,X,y,n,lambda)
%no of hidden layer =1
%no of hidden layer units = 25
%n=no of features
Theta1 = reshape(params(1:(25*(n+1))),25,n+1);
Theta2 = reshape(params(25*(n+1)+1:end),3,26);
m=size(X,1);
Theta1_grad=zeros(25,5);
Theta2_grad=zeros(3,26);
J=0;

for i=1:m,
    x = [1 X(i,:)]';
    yi = zeros(3,1);
    yi(y(i))=1;
    a1 = [1;sigmoid(Theta1*x)];
    a2 = sigmoid(Theta2*a1);
    J+=-(yi'*log(a2)+(1-yi)'*log(1-a2));
    delta2 = (a2-yi);
    Theta2_grad += delta2*a1';
    delta1 = (Theta2'*delta2).*(a1.*(1-a1));
    Theta1_grad += delta1(2:end)*x';
end
J=J/m;
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
J+=(lambda/(2*m))*sum(sum(Theta1(:,2:end).^2));
J+=(lambda/(2*m))*sum(sum(Theta2(:,2:end).^2));
Theta1_grad(:,2:end) += (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) += (lambda/m)*Theta2(:,2:end);
grad = [Theta1_grad(:);Theta2_grad(:)];
end
