clc;clear all; close all;
x_train = loadMNISTImages('train-images-idx3-ubyte');
x_train = x_train';
y_train = loadMNISTLabels('train-labels-idx1-ubyte');

x_test = loadMNISTImages('t10k-images-idx3-ubyte');
x_test = x_test';
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
x_test = x_test / 255.0 * 2 - 1;
x_train = x_train / 255.0 * 2 - 1;


rng(1);
t = templateSVM('KernelFunction','RBF',...
    'KernelScale','auto','BoxConstraint',2.8);
muhtime = tic;
Md1 = fitcecoc(x_train,y_train,'Learners',t);
toc(muhtime);
Md1.ClassNames
CodingMat = Md1.CodingMatrix;

% CVMd1 = crossval(Md1);
% toc(muhtime);
% oosLoss = kfoldLoss(CVMd1)
% toc(muhtime);

predicted = predict(Md1,x_test);
toc(muhtime);

[C,order] = confusionmat(y_test, predicted)
results = sum(predicted == y_test)/length(y_test)