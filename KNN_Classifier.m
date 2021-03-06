function [ KNNModel, GridSearch ] = KNN_Classifier( x_train, y_train, x_test, y_test, mode )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if mode == 0
    GridSearch.NSMethod = 'kdtree';
    GridSearch.Distance = {'chebychev','euclidean', 'cityblock', 'minkowski'};
    GridSearch.k = [1, 10, 100];
    GridSearch.Score = [];
elseif mode == 1
    GridSearch.NSMethod = 'kdtree';
    GridSearch.Distance = {'cityblock'};
    GridSearch.k = [5, 10, 20, 50];
    GridSearch.Score = [];
else
    GridSearch.NSMethod = 'kdtree';
    GridSearch.Distance = {'cityblock'};
    GridSearch.k = 5;
    GridSearch.Score = [];
end

if mode == 0 || mode == 1
    %Grid-Search to find optimal parameter.
    for i = 1:length(GridSearch)
        for j = 1:length(GridSearch(i).Distance)
            for k = 1:length(GridSearch(i).k)
                %Display which parameters are being tested. 
                fprintf('\nDistance: %s\n',GridSearch(i).Distance{j});
                fprintf('k: %i\n',GridSearch(i).k(k));
                
                timing = tic;
                
                disp('Training KNN Model');
                % Create an error correcting output coding model to allow for multi-class
                % SVM classification
                KNNModel = fitcknn(x_train,y_train,'NSMethod',...
                    GridSearch.NSMethod, 'Distance',GridSearch(i).Distance{j},...
                    'NumNeighbors',GridSearch(i).k(k));
                toc(timing);
                if mode == 0
                    % To combat long cross-val times, we did an intial 
                    % examination to see which distance metric would be most worth 
                    % optimizing.
                    disp('Predicting Using KNN Model');
                    predicted = predict(KNNModel,x_test);
                    toc(timing);
                    Accuracy = sum(predicted == y_test)/length(y_test);
                    fprintf('Accuracy: %0.4f\n',Accuracy);
                    GridSearch.Score(end+1) = Accuracy;

                else               
                    %Then, we cross-validated the number of neighbors to find
                    %the best parameter.
                    disp('Cross-Validating KNN Model');
                    CVMdl = crossval(KNNModel);  %10-fold crossval
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
    
    disp('Training KNN Model');
    KNNModel = fitcknn(x_train,y_train,'NSMethod',GridSearch.NSMethod,...
        'Distance',GridSearch.Distance,'NumNeighbors',GridSearch.k);    
    
    %Save model for future use.
    save('KNNModel.mat','KNNModel');
    
    %Display training time.
    toc(timing);
    
	disp('Predicting Test Labels with KNN Model');
    %Predict on the test set.
    predicted = predict(KNNModel,x_test);

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
