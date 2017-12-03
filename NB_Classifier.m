function [ Accuracy ] = NB_Classifier( x_train, y_train, x_test, y_test )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%Predict on the test set.
for i = 0:9
    index = var(x_train(y_train(:,1)==i,:))==0; % Find all features w/ zero varience for each label.

    x_train(:,index) = []; %Delete cols from training and test sets. 
    x_test(:,index) = []; %Delete cols from training and test sets. 
end

timing = tic;
% Create an error correcting output coding model to allow for multi-class
% Naive Bayes classification
NB_Model = fitcnb(x_train,y_train);
%Display training time.
toc(timing);

predicted = predict(NB_Model,x_test);

%Display inference time.
toc(timing);

%Visualize the classification results.
[C,order] = confusionmat(y_test, predicted)
plotconfusion(y_test, predicted)
Accuracy = sum(predicted == y_test)/length(y_test);

end

