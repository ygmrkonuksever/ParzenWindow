clc
clear all
load ('data_33rpz_cv05.mat');
xs=zeros(1800,1);
xtest=zeros(200,1);
buffer1=int16(0);
buffer2=int16(0);
syms x
global c1
global c2

for i=1:1800      %find x for all 1800 example
    for j=1:10
        for k=1:5
            buffer1= buffer1+ int16(trn_2000.images(j,k,i));
        end
        for k=6:10
            buffer2= buffer2+ int16(trn_2000.images(j,k,i));
        end
        buffer1=buffer1-buffer2;
        buffer2=0;
    end
    xs(i,1)=buffer1;
    buffer1=0;
end
xs(:,1)

count1=0;
count2=0;
for i=1:1800
    if(trn_2000.labels(1,i)==1)
        count1=count1+1;
    elseif(trn_2000.labels(1,i)==2)
        count2=count2+1;
    end
end
count1
count2

% x1's in w1 and w2
X1=zeros(count1,1);
X2=zeros(count2,1);
c1=1;
c2=1;

for i=1:1800
    if(trn_2000.labels(1,i)==1)
        X1(c1,1)=xs(i,1);
        c1=c1+1;
    elseif(trn_2000.labels(1,i)==2)
        X2(c2,1)=xs(i,1);
        c2=c2+1;
    end 
end


%Test set
for i=1:200      %find x for 200 test example
    for j=1:10
        for k=1:5
            buffer1= buffer1+ int16(tst.images(j,k,i));
        end
        for k=6:10
            buffer2= buffer2+ int16(tst.images(j,k,i));
        end
        buffer1=buffer1-buffer2;
        buffer2=0;
    end
    xtest(i,1)=buffer1;
    buffer1=0;
end
xtest(:,1)

%%%%%%Parzen window
for i=1:4
    if(i==1)
        sigma=100;
        fprintf("Parzen window for sigma=100");
    elseif(i==2)
        sigma=300;
        fprintf("Parzen window for sigma=300");
    elseif(i==3)
        sigma=1000;
        fprintf("Parzen window for sigma=1000");
    else
        sigma=2000;
        fprintf("Parzen window for sigma=2000");
    end
    for j=1:200
        y = my_parzen(xtest(j,1),X1,sigma);
        fprintf("Parzen result for");
        fprintf(" %d \n %d \n", xtest(j,1), y);
    end
end

%%%%%%%%%%% KNN   Distance metric:City-block
correct=0;
difference=99999;
nnclass=0;

%1-NN
for i=1:200
    for j=1:1179
        if(abs(xtest(i,1)-X1(j,1))<difference)
            difference=abs(xtest(i,1)-X1(j,1));
            nnclass=1;
        end
    end
    for j=1:621
        if(abs(xtest(i,1)-X2(j,1))<difference)
            difference=abs(xtest(i,1)-X2(j,1));
            nnclass=2;
        end
    end
    if(tst.labels(1,i)==1 && nnclass==1)
        correct=correct+1;
    elseif(tst.labels(1,i)==2 && nnclass==2)
        correct=correct+1;
    end
    difference=99999;
    nnclass=0;
end
accuracy=correct/200;
fprintf("1-NN Accuracy is: %d \n", accuracy);
correct=0;
accuracy=0;

%3-NN
difference3=zeros(3,1);
difference3(1,1)=99999;
difference3(2,1)=99999;
difference3(3,1)=99999;
nnclass3=zeros(3,1);
class1count=0;
class2count=0;

for i=1:200
    for j=1:1179
        if(abs(xtest(i,1)-X1(j,1))<difference3(1,1))
            difference3(1,1)=abs(xtest(i,1)-X1(j,1));
            nnclass3(1,1)=1;
        elseif(abs(xtest(i,1)-X1(j,1))<difference3(2,1))
            difference3(2,1)=abs(xtest(i,1)-X1(j,1));
            nnclass3(2,1)=1;
        elseif(abs(xtest(i,1)-X1(j,1))<difference3(3,1))
            difference3(3,1)=abs(xtest(i,1)-X1(j,1));
            nnclass3(3,1)=1;
        end
    end
    for j=1:621
        if(abs(xtest(i,1)-X2(j,1))<difference3(1,1))
            difference3(1,1)=abs(xtest(i,1)-X2(j,1));
            nnclass3(1,1)=2;
        elseif(abs(xtest(i,1)-X2(j,1))<difference3(2,1))
            difference3(2,1)=abs(xtest(i,1)-X2(j,1));
            nnclass3(2,1)=2;
        elseif(abs(xtest(i,1)-X2(j,1))<difference3(3,1))
            difference3(3,1)=abs(xtest(i,1)-X2(j,1));
            nnclass3(3,1)=2;
        end
    end
    for c=1:3
        if(nnclass3(c,1)==1)
            class1count=class1count+1;
        else
            class2count=class2count+1;
        end
    end
    if(class1count>class2count)
        nnclass=1;
    else
        nnclass=2;
    end
    if(tst.labels(1,i)==1 && nnclass==1)
        correct=correct+1;
    elseif(tst.labels(1,i)==2 && nnclass==2)
        correct=correct+1;
    end
    difference3(1,1)=99999;
    difference3(2,1)=99999;
    difference3(3,1)=99999;
    nnclass3=zeros(3,1);
    class1count=0;
    class2count=0;
end
accuracy=correct/200;
fprintf("3-NN Accuracy is: %d \n", accuracy);
correct=0;
accuracy=0;

%5-NN
difference5=zeros(5,1);
difference5(1,1)=99999;
difference5(2,1)=99999;
difference5(3,1)=99999;
difference5(4,1)=99999;
difference5(5,1)=99999;
nnclass5=zeros(5,1);
class1count=0;
class2count=0;

for i=1:200
    for j=1:1179
        if(abs(xtest(i,1)-X1(j,1))<difference5(1,1))
            difference5(1,1)=abs(xtest(i,1)-X1(j,1));
            nnclass5(1,1)=1;
        elseif(abs(xtest(i,1)-X1(j,1))<difference5(2,1))
            difference5(2,1)=abs(xtest(i,1)-X1(j,1));
            nnclass5(2,1)=1;
        elseif(abs(xtest(i,1)-X1(j,1))<difference5(3,1))
            difference5(3,1)=abs(xtest(i,1)-X1(j,1));
            nnclass5(3,1)=1;
        elseif(abs(xtest(i,1)-X1(j,1))<difference5(4,1))
            difference5(4,1)=abs(xtest(i,1)-X1(j,1));
            nnclass5(4,1)=1;
        elseif(abs(xtest(i,1)-X1(j,1))<difference5(5,1))
            difference5(5,1)=abs(xtest(i,1)-X1(j,1));
            nnclass5(5,1)=1;
        end
    end
    for j=1:621
        if(abs(xtest(i,1)-X2(j,1))<difference5(1,1))
            difference5(1,1)=abs(xtest(i,1)-X2(j,1));
            nnclass5(1,1)=2;
        elseif(abs(xtest(i,1)-X2(j,1))<difference5(2,1))
            difference5(2,1)=abs(xtest(i,1)-X2(j,1));
            nnclass5(2,1)=2;
        elseif(abs(xtest(i,1)-X2(j,1))<difference5(3,1))
            difference5(3,1)=abs(xtest(i,1)-X2(j,1));
            nnclass5(3,1)=2;
        elseif(abs(xtest(i,1)-X2(j,1))<difference5(4,1))
            difference5(4,1)=abs(xtest(i,1)-X2(j,1));
            nnclass5(4,1)=2;
        elseif(abs(xtest(i,1)-X2(j,1))<difference5(5,1))
            difference5(5,1)=abs(xtest(i,1)-X2(j,1));
            nnclass5(5,1)=2;
        end
    end
    for c=1:5
        if(nnclass5(c,1)==1)
            class1count=class1count+1;
        else
            class2count=class2count+1;
        end
    end
    if(class1count>class2count)
        nnclass=1;
    else
        nnclass=2;
    end
    if(tst.labels(1,i)==1 && nnclass==1)
        correct=correct+1;
    elseif(tst.labels(1,i)==2 && nnclass==2)
        correct=correct+1;
    end
    difference5(1,1)=99999;
    difference5(2,1)=99999;
    difference5(3,1)=99999;
    difference5(4,1)=99999;
    difference5(5,1)=99999;
    nnclass5=zeros(5,1);
    class1count=0;
    class2count=0;
end
accuracy=correct/200;
fprintf("5-NN Accuracy is: %d", accuracy);
correct=0;
accuracy=0;

%%%%%%%%%% Minimum Squared Error (MSE)
xs_200=zeros(200,1);
for i=1:200      %find x for all 200 example
    for j=1:10
        for k=1:5
            buffer1= buffer1+ int16(trn_200.images(j,k,i));
        end
        for k=6:10
            buffer2= buffer2+ int16(trn_200.images(j,k,i));
        end
        buffer1=buffer1-buffer2;
        buffer2=0;
    end
    xs_200(i,1)=buffer1;
    buffer1=0;
end
xs_200(:,1)

err = mse(xs_200,xtest);
fprintf("Minumum Squared Error(MSE) between test set and trn_200 dataset is: \n %d \n ", err);



function  y = my_parzen(x,X,sigma)
    global c1
    total=0;
    for i=1:c1-1
        a=x-X(i,1);
        total=total+normpdf(a,0,sigma);
    end
    y=total/c1;
end


