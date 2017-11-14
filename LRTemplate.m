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

% c = cvpartition(length(x_train),'KFold',10);
% sigma = optimizableVariable('sigma',[1e-5,1e5],'Transform','log');
% box = optimizableVariable('box',[1e-5,1e5],'Transform','log');
% rng(1);
% 
% minfn = @(z)kfoldLoss(fitcecoc(x_train,y_train,...
%     'Learners',templateSVM('KernelFunction','RBF',...
%     'KernelScale',z.sigma,'BoxConstraint',z.box)));
% OptResults = bayesopt(minfn,[sigma,box],'IsObjectiveDeterministic',true,...
%     'AcquisitionFunctionName','expected-improvement-plus')
muhtime = tic;
% 
% z(1) = OptResults.XAtMinObjective.sigma;
% z(2) = OptResults.XAtMinObjective.box;
%y = num2str(y_train);

%for k = 1:length(y)
 %  yt{k} = y(k);
%end


LogisticReg = mnrfit(x_train,y_train);%ordinal(y_train,{'0','1','2','3','4','5','6','7','8','9'},[],[0,1,2,3,4,5,6,7,8,9]),'model','ordinal');

predicted = predict(LogisticReg,x_test);
toc(muhtime);

[C,order] = confusionmat(y_test, predicted)
results = sum(predicted == y_test)/length(y_test)