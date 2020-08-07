% demon randomly generated data
clc; close all; clear all;  warning off
 
type        = 1; % 1 or 2 or 3
Ex          = {'2D', '3D', 'nD'};
m0          = 400;
n           = 100;
[X,y,tX,ty] = randomData(Ex{type},m0,n,0);  
[m,n ]      = size(X); 

pars.s0     = ceil(log10(m)*(10*(m<=1e4)+100*(m>1e4)));
out         = SNASVM(X,y,pars); 

[acc,~,ec]  = accuracy(X,out.w,y);
[tacc,~,tec]= accuracy(tX,out.w,ty);

fprintf('Training  Time:             %5.3fsec\n',out.time);
fprintf('Training  Size:             %dx%d\n',m,n);
fprintf('Training  Accuracy:         %5.2f%%\n',out.acc*100) 
fprintf('Testing   Size:             %dx%d\n',size(tX,1),n);
fprintf('Testing   Accuracy:         %5.2f%%\n',tacc*100);
fprintf('Number of Support Vectors:  %d\n',out.sv); 
if isequal(Ex{type},'2D') && m <400
   figure(1)
   subplot(1,2,1), plot2D(X,y,ec,out.w,'snasvm',acc);
   xlabel('Training data')
   subplot(1,2,2), plot2D(tX,ty,tec,out.w,'snasvm',tacc);
   xlabel('Testing data')
   saveas(figure(1), 'output\snasvm-2D.eps','epsc');
   saveas(figure(1), 'output\snasvm-2D.fig');
   saveas(figure(1), 'output\snasvm-2D.png');
end
