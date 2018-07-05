function [theta] = grad_descent(cf,init_theta,max_iter)
   theta = init_theta;
alpha = .8;
   for iter = 1:max_iter
       [J grad] =  cf(theta);
       theta = theta -alpha*grad;
       fprintf('%4i | Cost: %4.6e\r', iter, J);
   end  
end 