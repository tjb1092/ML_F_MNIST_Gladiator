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

% 
% z(1) = OptResults.XAtMinObjective.sigma;
% z(2) = OptResults.XAtMinObjective.box;
%y = num2str(y_train);

%for k = 1:length(y)
 %  yt{k} = y(k);
%end

y_train = y_train + 1;
y_test = y_test + 1;
x_train = x_train + 1;

timer = tic;
[B,dev,stats] = mnrfit(x_train,y_train);
toc(timer);

timer = tic;
Posterior = mnrval(B,X);
[val, predicted] = max(Posterior, [], 2);
toc(timer);

[C,order] = confusionmat(y_test, predicted)
results = sum(predicted == y_test)/length(y_test)