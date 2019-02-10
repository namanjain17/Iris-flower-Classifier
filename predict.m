function pred = predict(Theta1,Theta2,X)
    
   X = [ones(size(X,1),1) X]; 
   a1 = sigmoid(Theta1*X');
   a2 = sigmoid(Theta2*[ones(1,size(a1,2));a1]);
   a2=a2';
   [r p] = max(a2,[],2);
   pred = p;
end


