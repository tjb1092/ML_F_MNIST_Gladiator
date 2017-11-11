clc;clear all; close all;
x_train = loadMNISTImages('train-images-idx3-ubyte');
x_train = x_train';
y_train = loadMNISTLabels('train-labels-idx1-ubyte');

x_test = loadMNISTImages('t10k-images-idx3-ubyte');
x_test = x_test';
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
%Standardizing the pixel data
x_test = x_test / 255.0 * 2 - 1;
x_train = x_train / 255.0 * 2 - 1;

muhtime = tic;

KNNModel = fitcknn(x_train,y_train,'NumNeighbors',4);
rloss = resubLoss(KNNModel)
CVMdl = crossval(KNNModel);
kloss = kfoldLoss(CVMdl);

predicted = predict(KNNModel,x_test);
toc(muhtime);

[C,order] = confusionmat(y_test, predicted)
results = sum(predicted == y_test)/length(y_test)