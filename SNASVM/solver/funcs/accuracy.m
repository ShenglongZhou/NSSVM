function [acc,mis,sz] = accuracy(X,x,y)
if ~isempty(X)  
    z        = X*x(1:end-1)+x(end); 
    sz       = sign(z);
    sz(sz==0)= 1; 
    mis      = nnz(sz-y);
    acc      = 1-mis/length(y);
    clear X y
else
    acc = NaN;
    mis = NaN;
    sz  = NaN;
end
end