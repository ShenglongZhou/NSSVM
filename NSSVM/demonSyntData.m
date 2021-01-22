% demon randomly generated data
clc; close all; clear all;  warning off
 
type        = 1;   % 1 or 2 or 3
Ex          = {'2D', '3D', 'nD'};
m0          = 4e2;  
n0          = 100;
[X,y,tX,ty] = randomData(Ex{type},m0,n0,0);  
[m, n]      = size(X); 
pars.C      = 0.25;
pars.s0     = ceil(n*(log(m/n))^2);
out         = NSSVM(X,y,pars);  

[acc,~,ec]  = accuracy(X,out.w,y); 
[tacc,~,tec]= accuracy(tX,out.w,ty);

fprintf('Training  Time:             %5.3fsec\n',out.time);
fprintf('Training  Size:             %dx%d\n',m,n);
fprintf('Training  Accuracy:         %5.2f%%\n', acc*100) 
fprintf('Testing   Size:             %dx%d\n',size(tX,1),n);
fprintf('Testing   Accuracy:         %5.2f%%\n',tacc*100);
fprintf('Number of Support Vectors:  %d\n',out.sv); 
if isequal(Ex{type},'2D') && m <400
   figure('Renderer', 'painters', 'Position', [800, 200, 650 300])
   axes('Position', [0.12 0.14 0.9 0.8] ); 
   subplot(1,2,1), plot2D(X,y,ec,out.w,'NSSVM',acc);
   xlabel('Training data')
   subplot(1,2,2), plot2D(tX,ty,tec,out.w,'NSSVM',tacc);
   xlabel('Testing data')
end
