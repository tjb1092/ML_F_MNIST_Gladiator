clc;clear all; close all;
x_train = loadMNISTImages('train-images-idx3-ubyte');
x_train = x_train';
y_train = loadMNISTLabels('train-labels-idx1-ubyte');

x_test = loadMNISTImages('t10k-images-idx3-ubyte');
x_test = x_test';
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
rng(1);
t = templateSVM('Standardize',1,'KernelFunction','RBF',...
    'KernelScale','auto','BoxConstraint',2.8);

Md1 = fitcecoc(x_train,y_train,'Learners',t);

Md1.ClassNames
CodingMat = Md1.CodingMatrix;

CVMd1 = crossval(Md1);
oosLoss = kfoldLoss(CVMd1)

predicted = predict(Md1,x_test);


[C,order] = confusionmat(y_test, predicted)
results = sum(predicted == y_test)/length(y_test);