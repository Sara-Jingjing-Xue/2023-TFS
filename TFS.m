% B        input anchor graph 
% U0       initialization label matrix of anchors 
% alpha    parameter of PGD
% maxiter  the maxmum iteration 


function [Label0, Label, F, U,alpha,LOSS,iter] = TFS(B, U0, alpha, maxiter)
    [n,m]=size(B);
    U=U0;
    [m,c]=size(U);
    F=B*U;
    E=F'*F;
    aa = sum(E,2)+ eps*ones(c,1);
    LOSS(1)= sum(diag(E)./aa);
    [~,Label0]=max(B*U0,[],2);
for iter=1:maxiter 
    F=B*U;
    E=F'*F;
    d=diag(E); 
    a=sum(d); 
    b=sum(F,2);  
    aa=sum(F,1);  
    bb=sum(E,2); 
    
    a1=zeros(n,1);
    for j=1:c
        a1=a1+d(j)/(aa(j)^2+ eps)*F(:,j);
    end
    
    for i=1:m
       for j=1:c   
          gradient(i,j)=(( 2*F(:,j)*aa(j)- repmat(d(j),n,1))./(aa(j)^2 + eps) )'*B(:,i);
       end
    end
    nn(iter)=norm(gradient, 'fro' ) ;
    U = U + alpha*gradient;  
    for i=1:m
        U(i,:)= EProjSimplex_new(U(i,:)); 
    end
    
    F=B*U;
    E=F'*F;
    aa = sum(E,2)+ eps*ones(c,1); 
    LOSS(iter+1)= sum(diag(E)./aa); 
    if (iter>1 && abs(LOSS(iter+1)-LOSS(iter)) < 10^-6)
       break;
    end 
end

 F=B*U;
 [~,Label]=max(F,[],2);
 

 

