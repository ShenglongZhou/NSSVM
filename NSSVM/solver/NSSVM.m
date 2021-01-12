function Out = NSSVM(X,y,pars)
% Inputs:
%       X    -- the sample data, dimension, \in\R^{m-by-n}; (required)
%       y    -- the classes of the sample data, \in\R^m; (required)
%               y_i \in {+1,-1}, i=1,2,...,m
%       pars -- parameters (optional)
%               
%     pars:     Parameters are all OPTIONAL
%               pars.alpha0  --  Starting point of alpha \in\R^m,  (default, zeros(m,1)) 
%               pars.display --  =1. Display results for each iteration.(default)
%                                =0. No results are displayed for each iteration.
%               pars.maxit   --  Maximum number of iterations, (default,2000) 
%               pars.tol     --  Tolerance of the halting condition, (default,1e-6*sqrt(n*m))
%
% Outputs:
%     Out.alpha:         The sparse solution alpha
%     Out.w:             The solution of the primal problem, namely the classifier
%     Out.s:             Sparsity level of the solution Out.alpha
%     Out.sv:            Number of support vectors 
%     Out.time           CPU time
%     Out.iter:          Number of iterations
%     Out.acc:           Classification accuracy
%
%%%%%%%    Written by Shenglong Zhou on 18/06/2020 based on the algorithm proposed in
%%%%%%%    S.Zhou,Sparse SVM for Sufficient Data Reduction,arXiv:2005.13771,2020. 
%%%%%%%    Send your comments and suggestions to <<< shenglong.zhou@soton.ac.uk >>>                                  
%%%%%%%    Warning: Accuracy may not be guaranteed!!!!!               


if nargin<2; error('Inputs are not enough'); end

t0    = tic; 
[m,n] = size(X);

if issparse(X) && nnz(X)/m/n>0.1  
   X=full(X); 
end

if  n  <  3e4 
    Qt = y.*X;
else
    Qt = spdiags(y,0,m,m)*X; 
end

Q     = Qt';
yt    = y';
Fnorm = @(var)norm(var).^2;

[maxIt,tol,eta,s,C,c,alpha,display] = GetParameters(m,n);
if nargin<3;               pars    = [];             end
if isfield(pars,'s0');     s       = min(m,pars.s0); end
if isfield(pars,'tol');    tol     = pars.tol;       end
if isfield(pars,'maxIt');  maxIt   = pars.maxIt;     end
if isfield(pars,'alpha0'); alpha   = pars.alpha0;    end
if isfield(pars,'display');display = pars.display;   end
 
T1      = find(y==1);  nT1= nnz(T1);
T2      = find(y==-1); nT2= nnz(T2);
if  nT1 < s
    T  = [T1; T2(1:(s-nT1))];   
elseif nT2 < s
    T  = [T1(1:(s-nT2)); T2]; 
else 
    T  = [T1(1:ceil(s/2)); T2(1:(s-ceil(s/2)))];    
end
T       = sort(T(1:s));
 
mu0     = (nT1>=nT2)-(nT1<nT2);
mu      = mu0;
beta    = -ones(m,1);
maxACC  = 0;
flag    = 1;
ERR     = zeros(maxIt,1);  
ACC     = zeros(maxIt,1);  
ACC(1)  = 1-nnz(sign(mu)-y)/m; 

b0      = 0;
maxAcc0 = 0;

if  display
    fprintf('Run SNSVM ...... \n');   
    fprintf('------------------------------------------\n');
    fprintf('  Iter          Error           Accuracy  \n')
    fprintf('------------------------------------------\n');
end
for iter     = 1:maxIt
    
    if iter == 1 || flag
       QT    = Q(:,T);  
       QtT   = Qt(T,:); 
       yT    = y(T);
       ytT   = yt(T);
    end
  
    alphaT   =  alpha(T); 
    betaT    = -beta(T);  
    dm1      = -ytT*alphaT; 
    err      = (abs(Fnorm(alpha)-Fnorm(alphaT))+Fnorm(betaT)+dm1^2)/m; 
    ERR(iter)= sqrt(err);
    if  display
        fprintf('  %3d          %6.2e         %6.2f%%\n',iter,err,ACC(iter)*100); 
    end
    
    stop1    =  (iter>10 && max(ACC(iter-10:iter))<maxAcc0);   
    stop2    =  (iter>5  && std(ACC(1:iter))<1e-6);
    stop3    =  (iter>2  && ACC(iter)>= 0.99995); 
    stop4    =  (iter>2  && abs(ACC(iter)-maxAcc0)<2e-4...
                         && std(ACC(iter-2:iter))>1e-5);                      
    if  ACC(iter)>0 
        if stop1 || stop2 || stop3  || stop4     
           break;  
        end
    end
    
    ET     = (alphaT>=0)/C + (alphaT<0)/c;
    if  s <= n  
        if iter == 1 || flag
           PTT   = QtT*QT;  
        end    
        PTT      = PTT + spdiags(ET,0,s,s);
        d        = [PTT yT; ytT 0]\[betaT; dm1]; 
    else
        ETinv = 1./ET;
        if iter == 1 || flag         
           EQtT  = spdiags(ETinv,0,s,s)*QtT;  
           P0    = speye(n) + QT*EQtT; 
        end
        Ez       = ETinv.*betaT;
        Hz       = Ez-EQtT*(P0\(QT*Ez));  
        Ey       = ETinv.*yT;
        Hy       = Ey-EQtT*(P0\(QT*Ey));  
        dend     = (ytT*Hz-dm1)/(ytT*Hy);  
        d        = [Hz-dend*Hy; dend];  
    end

    alpha    = zeros(m,1);
    alphaT   = alphaT + d(1:s); 
    alpha(T) = alphaT;
    mu       = mu     + d(end); 
    
    w        = QT*alphaT;  
    Qtw      = Qt*w;
    beta     = Qtw - 1 + mu*y;
    ET       = (alphaT>=0)/C + (alphaT<0)/c;
    c        = max(1e-4,c/2);
    beta(T)  = alphaT.*ET + beta(T);  

    tmp      = y.*Qtw;
    b        = sum(y-tmp)/m; 
    j        = iter+1;
    
    ACC(j)    = 1-nnz(sign(tmp+b)-y)/m ;    
    if abs(ACC(j)-.25)<.25; ACC(j)=1-ACC(j); end
    
    if m <  6e6
       opt.MaxIter = 12*(m>=1e6)+30*(m<1e6);
       opt.Display = 'off';
       b0          = fminsearch(@(t)norm(sign(tmp+t(1))-y),b0,opt);
       ACC0        = 1-nnz(sign(tmp+b0)-y)/m; 
       if ACC(j)   < ACC0
          ACC(j)   = ACC0;  
          b        = b0;    
       end   
    end
     
    maxAcc0    = maxACC;
    if ACC(j)  > maxACC   
        maxACC = ACC(j);
        alpha0 = alpha; 
        w0     = [w;b];
    end 
    
    flag1 = (iter>4 && std(ERR(iter-4:iter))<1e-5);
    r     = 1;
    mark  = 0;
    if mod(iter,10)==0 || (flag1 && err<tol) 
       if flag1 ||  err<tol || mod(iter,20)==0 
       r     = 1.15;
       eta   = 1/m; 
       alpha = zeros(m,1);
       beta  = -ones(m,1);
       mu    = mu0;   
       mark  = 1;
       end
    end
  
    s     = min(m,ceil(r*s));    
    T0    = T;
    
    if s~=m
        if  m     < 5e7 
            [~,T] = maxk(abs(alpha-eta*beta),s);  
        else      
            [~,T] = sort(abs(alpha-eta*beta),'descend');
        end
            T     = sort(T(1:s));
   
        if  mark
            nT         = nnz(y(T)==1);
            if     nT == s 
                if nT2<= .75*s 
                T      = [T(1:s-ceil(nT2/2)); T2(1: ceil(nT2/2))]; 
                else
                T      = [T(1:ceil(s/4)); T2(1:s-ceil(s/4))]; 
                end
            elseif nT == 0
                if nT1<= .75*s 
                T      = [T(1:s-ceil(nT1/2)); T1(1: ceil(nT1/2))]; 
                else
                T      = [T(1:ceil(s/4)); T1(1:s-ceil(s/4))]; 
                end
            end
            T          = sort(T(1:s));
        end
    else
        T    = 1:m;
    end
    
    flag  = 1;
    if (nnz(T0)==s && isempty(setdiff(T,T0))) || nnz(T0)==m
       flag = 0; 
       T    = T0;
    end
    

end

if  iter == 1 
    w0      = [zeros(n,1); mu];
  	 alpha0  = alpha;
end

if  display
    fprintf('------------------------------------------\n');
end
Out.s     = s;
Out.w     = w0;
Out.sv    = nnz(alpha0);
Out.acc   = maxACC;
Out.iter  = iter; 
Out.time  = toc(t0);
Out.alpha = alpha0;
clear X y yt ytT yT Q Qt QtT QtT P0 PPT EQtT 
end

%--------------------------------------------------------------------------
function [maxIt,tol,eta,s0,C,c,alpha,display] = GetParameters(m,n)
maxIt = 1e3;
tol   = 1e-6*sqrt(n*m);  
mn    = m/n; 
if     mn < 10 ;  r = 1;
elseif mn < 1e2;  r = (1+1e-3*n)*log10(m);
elseif mn < 6e4;  r = max(5,0.1*n*log10(m));
elseif mn >=6e4;  r = max(5,5e1*n*log10(m)); 
end 
s0      = max(4,ceil(min(0.2*m,r*n)));   
C       = 1e0;
c       = 1e-2;
eta     = 1/m;
alpha   = zeros(m,1); 
display = 1;
end
