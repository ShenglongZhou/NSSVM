% demon real data data
clc; close all; clear all; warning off
name   = 'dhrb'; 
data   = load(strcat(name,'.mat')); 
class  = load(strcat(name,'class.mat')); 
M      = size(data.X,1); class.y(class.y~=1)= -1;       
data.X = normalization(data.X,2*(max(data.X(:))>1)); % normalize the data
 
% randomly splitting the data into training and testing data
m0 = ceil(0.9*M);  mt = M-m0; rng(1);   I  = randperm(M);
Tt = I(1:mt);      tX = data.X(Tt,:);   ty = class.y(Tt);   % testing  data 
T  = I(1+mt:end);  X  = data.X(T,:);    y  = class.y(T,:);  % training data
clear data class  

[m,n]        = size(X); 
out          = NSSVM(X,y); 
tacc         = accuracy(tX,out.w,ty);
fprintf('Training  Time:             %5.3fsec\n',out.time);
fprintf('Training  Size:             %dx%d\n',m,n);
fprintf('Training  Accuracy:         %5.2f%%\n',out.acc*100);
fprintf('Number of Support Vectors:  %d\n',out.sv); 
fprintf('Testing   Size:             %dx%d\n',size(tX,1),n);
fprintf('Testing   Accuracy:         %5.2f%%\n',tacc*100);
