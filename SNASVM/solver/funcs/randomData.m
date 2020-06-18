function [Atr,ctr,Ate,cte] = randomData(type,m,n,r)

% This file aims at generating data of 3 examples
% Inputs:
%       type    -- can be '2D','3D' or 'nD' 
%       m       -- number of samples
%       n       -- number of features
%       r       -- flipping ratio
% Outputs:
%       Atr     --  training samples data,    m/2-by-n
%       ctr     --  training samples classes, n-by-1
%       Ate     --  testing  samples data,    m/2-by-n
%       cte     --  testing  samples classes, n-by-1
%
% written by Shenglong Zhou, 10/05/2020

m2    = ceil(m/2);
rng('shuffle');

switch type
    case '2D'
        A     = [ 0.5+sqrt(0.5)*randn(m2,1)  -3+sqrt(3)*randn(m2,1);
                 -0.5+sqrt(0.5)*randn(m2,1)   3+sqrt(3)*randn(m2,1)];
        c     = [-ones(m2,1); ones(m2,1)];    
    case '3D'
        rho   = .5 + 0.03*randn(m2,1); 
        t     = 2*pi*rand(m2,1);   
        data1 = [rho.*cos(t), rho.*sin(t) rho.*rho];

        rho     = .5 + 0.03*randn(m2,1); 
        t     = 2*pi*rand(m2,1);      
        data2 = [rho.*cos(t), rho.*sin(t) -rho.*rho];
        A     = [data1; data2];
        c     = [ones(m2,1); -ones(m2,1)]; 
    case 'nD'
        c     = ones(m,1);
        I0    = randperm(m);
        I     = I0(1:ceil(m2)); 
        c(I)  = -1;
        A     = repmat(c.*rand(m,1),1,n)+ randn(m,n);
end

T   = randperm(m); 
Atr = A(T(1:m2),:); 
ctr = c(T(1:m2)); 
ctr = filp(ctr,r);  

Ate = A(T(m2+1:m),:); 
cte = c(T(1+m2:m));     
clear A c T q
end

function fc = filp(fc,r)
      if r  > 0
         mc = length(fc) ;    
         T0 = randperm(mc);  
         fc(T0(1:ceil(r*mc)))=-fc(T0(1:ceil(r*mc)));
     end
end