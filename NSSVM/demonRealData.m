% demon real data data
clc; close all; clear all; warning off
  
load 'dhrb.mat';   load 'dhrbclass.mat'; 
M  = size(X,1);    y(y~=1)= -1;       
X  = normalization(X,2*(max(X(:))>1)); % normalize the data
 
% randomly split the data into training and testing data
m0 = ceil(0.9*M);  mt = M-m0; rng(1);  I  = randperm(M);
Tt = I(1:mt);      tX = X(Tt,:);       ty = y(Tt);   % testing  data 
T  = I(1+mt:end);  X  = X(T,:);        y  = y(T,:);  % training data

out  = NSSVM(X,y); 
tacc = accuracy(tX,out.w,ty);
fprintf('Training  Time:             %5.3fsec\n',out.time);
fprintf('Training  Size:             %dx%d\n',size(X,1),size(X,2));
fprintf('Training  Accuracy:         %5.2f%%\n',out.acc*100);
fprintf('Number of Support Vectors:  %d\n',out.sv); 
fprintf('Testing   Size:             %dx%d\n',size(tX,1),size(tX,2));
fprintf('Testing   Accuracy:         %5.2f%%\n',tacc*100);
