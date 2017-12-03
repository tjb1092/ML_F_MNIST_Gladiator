function [ Accuracy ] = ECOC_Classifier( x_train, y_train, x_test, y_test )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

Kernel(1).type = 'rbf';
Kernel(1).Scale = {'auto',1e-3,1e-2,1e-1,1,10};
Kernel(1).BoxConstraint = [1e-1, 1, 10];

Kernel(2).type = 'linear';
Kernel(2).Scale = {'auto',1e-3,1e-2,1e-1,1,10};
Kernel(2).BoxConstraint = [1e-1, 1, 10];

for i = 1:length(Kernel)
    for j = 1:length(Kernel(i).Scale)
        for k = 1:length(Kernel(i).BoxConstraint)
            
        end
    end
end


timing = tic;

%Bayesian Optimization only works on 2016b onward.
% rng default
% Mdl = fitcecoc(X,Y,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus'))

%Define the SVM function that will be used to train on the data.
t = templateSVM('KernelFunction','rbf',...
    'KernelScale','auto','BoxConstraint',2.8, 'Verbose', 1);

% Create an error correcting output coding model to allow for multi-class
% SVM classification
SVMModel = fitcecoc(x_train,y_train,'Learners', t);
%Display training time.
toc(timing);

%Predict on the test set.
predicted = predict(SVMModel,x_test);

%Display inference time.
toc(timing);

% CVSVMModel = crossval(SVMModel);
% [~,scorePred] = kfoldPredict(CVSVMModel);
% outlierRate = mean(scorePred<0)  % Represents the fraction considered outliers.


%Visualize the classification results.
[C,order] = confusionmat(y_test, predicted)
Accuracy = sum(predicted == y_test)/length(y_test);

end

