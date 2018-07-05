load('data.txt');
rp = randperm(150);
X=data(rp,1:4);
Y=data(rp,5);
X_train = X(1:100,:);
Y_train = Y(1:100,1);
X_test  = X(101:150,:);
Y_test = Y(101:150,1);
[X_norm,mu,sigma]=featureNormalize(X_train);
cf = @(t) costFunction(t,X_norm,Y_train,4,1);
Theta1 = randInit(25,5);
Theta2 = randInit(3,26);
initial_theta = [Theta1(:);Theta2(:)];
[theta] = grad_descent(cf,initial_theta,260);
Theta1 = reshape(theta(1:25*5),25,5);
Theta2 = reshape(theta(25*5+1:end),3,26);
[X_norm1,mu1,sigma1]=featureNormalize(X_test);
%To predict on your own dataset replace X_test with your feature 
%file and print pred where pred(i) belongs to [1,3]
%1 = Iris-setosa
%2 = Iris-versicolor
%3 = Iris-virginica                                   
pred = predict(Theta1, Theta2, X_norm1);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y_test)) * 100);




