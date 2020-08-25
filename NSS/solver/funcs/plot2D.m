function  plot2D(Atr,ctr,ec,x,text,acc)

siz = 50;
at1 = Atr(ctr==1,:);
at2 = Atr(ctr==-1,:);
x0  = [-2 2]; 
y0  = 2.5*x0;

scatter(at1(:,1),at1(:,2),siz,'o','m','LineWidth',1.5), hold on
scatter(at2(:,1),at2(:,2),siz,'o','b','LineWidth',1.5), hold on
line(x0,y0,'Color','black','LineStyle',':','LineWidth',1.5)
axis([-2 2 min([at1(:,2);at2(:,2)]) max([at1(:,2);at2(:,2)])])
box on, grid on 

if nargin>2
at1 = Atr(ec==1,:);
at2 = Atr(ec==-1,:);
x0  = [-2 2 ]; 
y   = -x(1)/x(2)*x0-x(3)/x(2);
scatter(at1(:,1),at1(:,2),siz,'+','m','LineWidth',1.5), hold on
scatter(at2(:,1),at2(:,2),siz,'x','b','LineWidth',1.5), hold on
line(x0,y,'Color','g','LineWidth',1.5), box on
legend('Positive','Negative','Bayes',strcat('Positive-',text),...
       strcat('Negative-',text),text,'Location','NorthEast')
title(strcat('Accuracy:',num2str(acc*100,'%7.2f%%'))), grid on   
axis([-2 2 min(Atr(:,2))-.1 max(Atr(:,2))+1.5])
end
end
