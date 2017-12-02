clc;clear all; close all;
addpath(genpath('drtoolbox'))
[x_train, image_train] = loadMNISTImages('train-images-idx3-ubyte');
x_train = x_train';
y_train = loadMNISTLabels('train-labels-idx1-ubyte');

[x_test, image_test] = loadMNISTImages('t10k-images-idx3-ubyte');
x_test = x_test';
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');

%Standardizing the pixel data
x_test = x_test / 255.0 * 2 - 1;
x_train = x_train / 255.0 * 2 - 1;

visualizeData(x_test,y_test, image_test);

Kernel(1).type = 'rbf';
Kernel(1).Scale = {'auto',1e-3,1e-2,1e-1,1,10};
Kernel(1).BoxConstraint = [1e-1, 1, 10];

Kernel(2).type = 'linear';
Kernel(2).Scale = {'auto',1e-3,1e-2,1e-1,1,10};
Kernel(2).BoxConstraint = [1e-1, 1, 10];

for i = 1:length(Kernel)
    for j = 1:length(Scale)
        for k = 1:length(BoxConstraint)
            
        end
    end
end


timing = tic;

%Define the SVM function that will be used to train on the data.
t = templateSVM('KernelFunction','rbf',...
    'KernelScale','auto','BoxConstraint',2.8, 'Verbose', 1);

% Create an error correcting output coding model to allow for multi-class
% SVM classification
SVMModel = fitcecoc(x_train,y_train,...
    'Learners',t);

%Display training time.
toc(timing);

%Predict on the test set.
predicted = predict(SVMModel,x_test);

%Display inference time.
toc(timing);

CVSVMModel = crossval(SVMModel);
[~,scorePred] = kfoldPredict(CVSVMModel);
outlierRate = mean(scorePred<0)  % Represents the fraction considered outliers.


%Visualize the classification results.
[C,order] = confusionmat(y_test, predicted)
results = sum(predicted == y_test)/length(y_test)