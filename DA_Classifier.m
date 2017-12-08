function [ DAModel, GridSearch ] = DA_Classifier( x_train, y_train, x_test, y_test, mode )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%Didn't have time to come up with a more elegant solution here. These are
%the results of sequentially optimizing the hyper-parameters. From general
%to emperically observed "best".
if mode == 0
    GridSearch.Gamma = [0, 0.1, 0.5, 0.9];
    GridSearch.Delta = [0, 0.1, 0.5, 1];
    GridSearch.Score = [];
elseif mode == 1
    GridSearch.Gamma = [0,0.01, 0.05, 0.1];
    GridSearch.Delta = 0;
    GridSearch.Score = [];
else
    GridSearch.Gamma = 0;
    GridSearch.Delta = 0;
    GridSearch.Score = [];
end

if mode == 0 || mode == 1
    %Grid-Search to find optimal parameter.
    for i = 1:length(GridSearch)
        for j = 1:length(GridSearch(i).Gamma)
            for k = 1:length(GridSearch(i).Delta)
                %Display which parameters are being tested. 
                fprintf('\nGamma: %0.2f\n',GridSearch(i).Gamma(j));
                fprintf('Delta: %0.2f\n',GridSearch(i).Delta(k));
                
                timing = tic;
                
                disp('Training DA Model');
                % Create an error correcting output coding model to allow for multi-class
                % SVM classification
                DAModel = fitcdiscr(x_train,y_train,'Gamma',...
                    GridSearch.Gamma(j), 'Delta',GridSearch(i).Delta(k));
                toc(timing);
                if mode == 0
                    % To combat long cross-val times, we did an intial 
                    % examination to see which distance metric would be most worth 
                    % optimizing.
                    disp('Predicting Using DA Model');
                    predicted = predict(DAModel,x_test);
                    toc(timing);
                    Accuracy = sum(predicted == y_test)/length(y_test);
                    fprintf('Accuracy: %0.4f\n',Accuracy);
                    GridSearch.Score(end+1) = Accuracy;

                else               
                    %Then, we cross-validated the number of neighbors to find
                    %the best parameter.
                    disp('Cross-Validating DA Model');
                    CVMdl = crossval(DAModel);  %10-fold crossval
                    toc(timing);
                    disp('Computing Out-Of-Sample Loss Score');
                    oosLoss = kfoldLoss(CVMdl); %Compute score
                    fprintf('OOS Loss Score: %0.4f\n',oosLoss);
                    GridSearch.Score(end+1) = oosLoss;
                end
                
                disp('Total Time');
                toc(timing);
                
            end
        end
    end
else
    %Find final, optimized results.
    timing = tic;

    disp('Training DA Model');
    DAModel = fitcdiscr(x_train,y_train,'Gamma',...
                    GridSearch.Gamma, 'Delta',GridSearch.Delta);
    %Display training time.
    toc(timing);

    %Save model for future use.
    save('DAModel.mat','DAModel');

    disp('Predicting Using DA Model');
    %Predict on the test set.
    predicted = predict(DAModel,x_test);

    %Display inference time.
    toc(timing);

    %Visualize the classification results.
    [C,order] = confusionmat(y_test, predicted);
    %Print confusion matrix
    fprintf('Label:\t\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n', order.') %Write col header
    fprintf('            -------------------------------------\n'); %hline
    fprintf('%i | \t\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n', [order,C].')% Write Rows
    
    Accuracy = sum(predicted == y_test)/length(y_test);
    GridSearch.Score(end+1) = Accuracy;
    fprintf('\nClassification Accuracy: %0.4f\n',Accuracy);
end

end