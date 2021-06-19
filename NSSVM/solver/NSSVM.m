function Out = NSSVM(X,y,pars)
% Inputs:
%       X    -- the sample data, dimension, \in\R^{m-by-n}; (required)
%       y    -- the classes of the sample data, \in\R^m; (required)
%               y_i \in {+1,-1}, i=1,2,...,m
%       pars -- parameters (optional)
%               
%     pars:     Parameters are all OPTIONAL
%               pars.alpha   --  Starting point of alpha \in\R^m,  (default, zeros(m,1)) 
%               pars.disp    --  =1. Display results for each iteration.(default)
%                                =0. No results are displayed for each iteration.
%               pars.s0      --  The initial sparsity level  
%                                An integer in [1,m] (default, n(log(m/n))^2) 
%               pars.tune    --  =1. Tune the sparsity level automatically.
%                                =0. Not tune the saprsity level.(default)
%               pars.C       --  A positive scalar in (0,1].(default, 1/4) 
%               pars.c       --  A positive scalar in (0,1].(default, 1/8) 
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
%%%%%%%    Shenglong Zhou, Sparse SVM for Sufficient Data Reduction, 2021,
%%%%%%%    IEEE Transactions on Pattern Analysis and Machine Intelligence, 10.1109/TPAMI.2021.3075339. 
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
Fnorm = @(var)norm(var,'fro')^2;

[maxit,alpha,tune,disp,tol,eta,s0,C,c] = GetParameters(m,n);
if nargin<3;               pars  = [];             end
if isfield(pars,'maxit');  maxit = pars.maxit;     end
if isfield(pars,'alpha');  alpha = pars.alpha;     end
if isfield(pars,'disp');   disp  = pars.disp;      end
if isfield(pars,'tune');   tune  = pars.tune;      end
if isfield(pars,'tol');    tol   = pars.tol;       end
if isfield(pars,'eta');    eta   = pars.eta;       end
if isfield(pars,'s0');     s0    = min(m,pars.s0); end
if isfield(pars,'C');      C     = pars.C;         end
if isfield(pars,'c');      c     = pars.c;         end

T1      = find(y==1);  nT1= nnz(T1);
T2      = find(y==-1); nT2= nnz(T2);
if  nT1 < s0
    T  = [T1; T2(1:(s0-nT1))];   
elseif nT2 < s0
    T  = [T1(1:(s0-nT2)); T2]; 
else 
    T  = [T1(1:ceil(s0/2)); T2(1:(s0-ceil(s0/2)))];    
end
T       = sort(T(1:s0));
s       = s0;
b       = (nT1>=nT2)-(nT1<nT2);
bb      = b;
w       = zeros(n,1);
gz      = -ones(m,1);
ERR     = zeros(maxit,1);  
ACC     = zeros(maxit,1);  
ACC(1)  = 1-nnz(sign(b)-y)/m; 
ET      = ones(s,1)/C;

maxACC  = 0;
flag    = 1;
j       = 1;
r       = 1.1; 
count   = 1;
count0  = 2;
iter0   = -1;

if  disp
    fprintf('Run NSSVM ...... \n');   
    fprintf('------------------------------------------\n');
    fprintf('  Iter          Error           Accuracy  \n')
    fprintf('------------------------------------------\n');
end
 
for iter     = 1:maxit
    
    if iter == 1 || flag
       QT    = Q(:,T);  
       QtT   = Qt(T,:); 
       yT    = y(T);
       ytT   = yT';  
    end
      
    alphaT   =  alpha(T);  
    gzT      = -gz(T);  
    alyT     = -ytT*alphaT; 
    
    err      = (abs(Fnorm(alpha)-Fnorm(alphaT))+Fnorm(gzT)+alyT^2)/(m*n); 
    ERR(iter)= sqrt(err);   
      
    if  tune  && iter < 30  && m<=1e8 
        stop1  = ( iter>5 && err < tol*s*log2(m)/100);  
        stop2  = ( s~=s0 && abs(ACC(iter)- max(ACC(1:iter-1))) <= 1e-4);   
        stop3  = ( s~=s0 && iter>10 &&  max(ACC(iter-5:iter)) < maxACC); 
        stop4  = ( count~=count0+1 && ACC(iter)>= ACC(1));   
        stop   = ( stop1 && (stop2 || stop3) &&  stop4 ); 
    else         
        stop1  = ( err <tol*sqrt(s)*log10(m) );   
        stop2  = ( iter>4 && std(ACC(iter-1:iter))<1e-4 );    
        stop3  = ( iter>20 && abs(max(ACC(iter-8:iter ))-maxACC)<=1e-4 );    
        stop   = ( stop1  && stop2) || stop3;
    end
    if  disp
        fprintf('  %3d          %6.2e         %7.5f\n',iter,err,ACC(iter)); 
    end
    
    if  ACC(iter)>0 && ( ACC(iter)>= 0.99999  || stop )
        break;  
    end
    
    ET0    = ET;
    ET     = (alphaT>=0)/C +(alphaT<0)/c; 
  
    if min(n,s) > 1e3
         d      = my_cg(QT,yT,ET,[gzT; alyT],1e-10,50,zeros(s+1,1));
         dT     = d(1:s);
         dend   = d(end); 
    else     
        if  s  <=   n   
            if iter == 1 || flag
               PTT0   = QtT*QT;  
            end         
            PTT  = PTT0 + spdiags(ET,0,s,s);
            d    = [PTT yT; ytT 0]\[gzT; alyT];  
            dT   = d(1:s);
            dend = d(end);  
        else           
            ETinv = 1./ET;           
            flag1  = nnz(ET0)~=nnz(ET);
            flag2  = nnz(ET0)==nnz(ET) && nnz(ET0-ET)==0; 
            if iter == 1 || flag || flag1 || ~flag2 
               EQtT  = spdiags(ETinv,0,s,s)*QtT;  
               P0    = speye(n) + QT*EQtT; 
            end     
            Ey   = ETinv.*yT;
            Hy   = Ey-EQtT*(P0\(QT*Ey));  
            dend = (gzT'*Hy-alyT)/(ytT*Hy); 
            tem  = ETinv.*(gzT-dend*yT); 
            dT   = tem-EQtT*(P0\(QT*tem));   
                       
        end    
    end
  
    alpha    = zeros(m,1);  
    alphaT   = alphaT + dT;  
    alpha(T) = alphaT;
    b        = b + dend; 
    
    w        = QT*alphaT;  
    Qtw      = Qt*w;    
    tmp      = y.*Qtw; 
        
    gz       = Qtw - 1 + b*y;
    ET1      = (alphaT>=0)/C + (alphaT<0)/c; 
    gz(T)    = alphaT.*ET1 + gz(T);  
          
    j        = iter+1;   
    ACC(j)   = 1-nnz(sign(tmp+b)-y)/m; 
    
    if m    <= 1e7
        bb   = mean(yT-tmp(T)); 
        ACCb = 1-nnz(sign(tmp+bb)-y)/m;         
        if  ACC(j) >= ACCb 
            bb      = b;  
        else
            ACC(j)  = ACCb; 
        end
    else
        bb = b;
    end
      
    if  m <  6e6 &&  ACC(j)<0.5
        opt.MaxIter = 10*(m>=1e6)+20*(m<1e6);
        opt.Display = 'off';
        b0          = fminsearch(@(t)sum((sign(tmp+t(1))-y).^2),bb,opt);     
        acc0        = 1-nnz(sign(tmp+b0)-y)/m;   
        if  ACC(j)  < acc0
            bb      = b0;   
            ACC(j)  = acc0;
        end  
    end
   
    if ACC(j) >= maxACC   
        maxACC = ACC(j);
        alpha0 = alpha;
        tmp0   = tmp;
        maxwb  = [w;bb];
    end 
    
    T0    = T;        
    mark  = 0;     
    if tune && ( err<tol || mod(iter,10)==0 ) && iter>iter0+2 && count<10
       count0   = count;  
       count    = count + 1;       
       s        = min(m,ceil(r*s));  
       iter0    = iter;
       if count > (m>=1e6 || n<3) + 1*(m<1e6 && n>=5)           
          alpha = zeros(m,1);
          gz    = -ones(m,1); 
          mark  = 1;  
       end   
    else
       count0   = count; 
    end   
     
    if s~= m      
        if  m     < 5e8 
            [~,T] = maxk(abs(alpha-eta*gz),s);  
        else      
            [~,T] = sort(abs(alpha-eta*gz),'descend');
        end
            T     = sort(T(1:s));

        if  mark
            nT  = nnz(y(T)==1);  
            if  nT == s 
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
    flag3 = nnz(T0)==s;
 
    if flag3
       flag3 = nnz(T-T0)==0;
    end
    if flag3 || nnz(T0)==m
       flag = 0; 
       T    = T0;  
    end   
 
end

% output results ----------------------------------------------------------

wb   = [w;bb];
acc  = ACC(j);  

if m <= 1e7 && iter >1
   opt.MaxIter = 20; 
   opt.Display = 'off';
   b0          = fminsearch(@(t)norm(sign(tmp0+t(1))-y),maxwb(end),opt);  
   acc0        = 1-nnz(sign(tmp0+b0)-y)/m;     
   if acc      < acc0
      wb       = [maxwb(1:end-1);b0];  
      acc      = acc0; 
   end   
end

if acc    < maxACC-1e-4  
   alpha  = alpha0;
   wb     = maxwb;  
   acc    = maxACC;
end

if  disp
    fprintf('------------------------------------------\n');
end
Out.s     = s;
Out.w     = wb;
Out.sv    = s;
Out.ACC   = acc;
Out.iter  = iter; 
Out.time  = toc(t0);
Out.alpha = alpha;
clear X y yt ytT yT Q Qt QtT QtT P0 PPT EQtT 
end

%Initial parameters --------------------------------------------------------
function [maxit,alpha,tune,disp,tol,eta,s0,C,c] = GetParameters(m,n)
maxit   = 1e3;
alpha   = zeros(m,1);
tune    = 0; 
disp    = 1;
tol     = 1e-6;  
eta     = min(1/m,1e-4);
if max(m,n)<1e4; beta = 1;
elseif m<=5e5;   beta = 0.05;
elseif m<=1e8;   beta = 10;   
end 
s0   = ceil(beta*n*(log2(m/n))^2);
if m > 5e6
   C = log10(m);  
else
   C = 1/2;      
end
c    = C/100; 
end

% Conjugate gradient method-------------------------------------------------
function x = my_cg(Q,y,E,b,cgtol,cgit,x)
    r = b;
    e = sum(r.*r);
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end
        p1 = p(1:end-1);
        w  =  [((Q*p1)'*Q)'+ E.*p1 + p(end)*y; sum(y.*p1)]; 
        a  = e/sum(p.*w);  
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end
end
