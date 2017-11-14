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
timing = tic;

SVMModel = fitcecoc(x_train,y_train,...
    'Learners',templateSVM('KernelFunction','rbf',...
    'KernelScale','auto','BoxConstraint',2.8));

toc(timing);
SVMModel.ClassNames
CodingMat = SVMModel.CodingMatrix;
% 
% CVMdl = crossval(SVMModel);
% oosLoss = kfoldLoss(CVMdl)

predicted = predict(SVMModel,x_test);
toc(timing);

[C,order] = confusionmat(y_test, predicted)
results = sum(predicted == y_test)/length(y_test)