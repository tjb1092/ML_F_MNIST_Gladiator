function [ Accuracy, NB_Model ] = NB_Classifier( x_train, y_train, x_test, y_test )
% Performs the optimization and training of the Naive Bayes Classifier.
% Stores the generated model for future use as a .mat file. 
% Attemped to add Kernel smoothing, but it made the algorithm 
% prohibitively expensive to train and cross-validate (upwards of 15hr per
% parameter test with little gain in accuracy compared to the simple NB). 

% Pre-processing:
for i = 0:9
    index = var(x_train(y_train(:,1)==i,:))==0; % Find all features w/ zero varience for each label.

    x_train(:,index) = []; %Delete cols from training and test sets. 
    x_test(:,index) = []; %Delete cols from training and test sets. 
end

timing = tic;
% Naive Bayes classification
disp('Training Naive Bayes Model');
NB_Model = fitcnb(x_train,y_train);

%Save model for future use.
save('NBModel.mat','NB_Model');
%Display training time.
toc(timing);

disp('Cross-Validating Naive Bayes Model');
CVMdl = crossval(NB_Model);  %10-fold crossval
toc(timing);
disp('Display k-fold out-of-sample Loss');
oosLoss = kfoldLoss(CVMdl); %Compute score
fprintf('OOS Loss Score: %0.4f\n',oosLoss);
toc(timing);   
   
disp('Predicting Test Labels Naive Bayes Model');
predicted = predict(NB_Model,x_test);

%Display inference time.
toc(timing);

%Visualize the classification results.
[C,order] = confusionmat(y_test, predicted);
fprintf('Label:\t\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n', order.') %Write col header
fprintf('%i | \t\t\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n', order, C.')% Write Rows
Accuracy = sum(predicted == y_test)/length(y_test);
fprintf('\nClassification Accuracy: %0.4f\n',Accuracy);

end

