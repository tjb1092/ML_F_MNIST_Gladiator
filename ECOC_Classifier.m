function [ SVMModel, GridSearch ] = ECOC_Classifier( x_train, y_train, x_test, y_test, mode )
% Performs the optimization and training a multi-class Support Vector Machine.
% Stores the generated model for future use as a .mat file. 
% Looked at the rbf and linear kernel using the 'auto' scaling option.
% Specific scaling parameters were chosen, but these were found to be
% significantly more computationally complex (upwards of 10x longer to 
% train with cross-validations taking upwards of 12hr per parameter set)  
% and yielded worse, or equivalent classification accuracies. Grid-search
% was therefore performed on the BoxConstraint (i.e. cost) parameter.

if mode == 1
    GridSearch(1).type = 'rbf';
    GridSearch(1).Scale = {'auto'};
    GridSearch(1).BoxConstraint = [1e-1, 1, 10];
    GridSearch(1).Score = [];
    GridSearch(2).type = 'linear';
    GridSearch(2).Scale = {'auto'};
    GridSearch(2).BoxConstraint = [1e-1, 1, 10];
    GridSearch(2).Score = [];
else
    GridSearch.type = 'rbf';
    GridSearch.Scale = 'auto';
    GridSearch.BoxConstraint = 10;
    GridSearch.Score = [];
end
if mode == 1
    %Grid-Search to find optimal parameter.
    for i = 1:length(GridSearch)
        for j = 1:length(GridSearch(i).Scale)
            for k = 1:length(GridSearch(i).BoxConstraint)
                
                fprintf('Kernel Type: %s\n', GridSearch(i).type);
                fprintf('Kernel Scale: %s\n',GridSearch(i).Scale);
                fprintf('Kernel Box Constraint: %0.2f\n',GridSearch(i).BoxConstraint(k));
                timing = tic;
                disp('Training SVM Model');
                t = templateSVM('KernelFunction',GridSearch(i).type,...
        'KernelScale',GridSearch(i).Scale{j},'BoxConstraint',GridSearch(i).BoxConstraint(k));

                % Create an error correcting output coding model to allow for multi-class
                % SVM classification
                SVMModel = fitcecoc(x_train,y_train,'Learners', t);
                toc(timing);

                disp('Cross-Validating SVM Model');
                CVMdl = crossval(SVMModel);  %10-fold crossval
                toc(timing);
   
                disp('Computing OOS Loss');
                oosLoss = kfoldLoss(CVMdl); %Compute score
                fprintf('OOS Loss Score: %0.4f\n',oosLoss);
                GridSearch.Score(end+1) = oosLoss;
                toc(timing);
            end
        end
    end
else
    timing = tic;

    disp('Training SVM Model');
    %Define the SVM function that will be used to train on the data.
    t = templateSVM('KernelFunction',GridSearch.type,...
        'KernelScale',GridSearch.Scale,'BoxConstraint',GridSearch.BoxConstraint);

    % Create an error correcting output coding model to allow for multi-class
    % SVM classification
    SVMModel = fitcecoc(x_train,y_train,'Learners', t);
    %Display training time.

    %Save model for future use.
    save('SVMModel.mat','SVMModel');
    toc(timing);
    disp('Predicting Test Labels with SVM Model');
    %Predict on the test set.
    predicted = predict(SVMModel,x_test);

    %Display inference time.
    toc(timing);

    %Visualize the classification results.
    [C,order] = confusionmat(y_test, predicted);
    %print confusion matrix
    fprintf('Label:\t\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n', order.') %Write col header
    fprintf('            -------------------------------------\n'); %hline
    fprintf('%i | \t\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\t%i\n', [order,C].')% Write Rows

    Accuracy = sum(predicted == y_test)/length(y_test);
    GridSearch.Score(end+1) = Accuracy;
    fprintf('\nClassification Accuracy: %0.4f\n',Accuracy);
end
end

