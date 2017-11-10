x_train = loadMNISTImages('train-images-idx3-ubyte');
x_train = x_train';
y_train = loadMNISTLabels('train-labels-idx1-ubyte');

x_test = loadMNISTImages('t10k-images-idx3-ubyte');
x_test = x_test';
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');
x_test = x_test / 255.0 * 2 - 1;
x_train = x_train / 255.0 * 2 - 1;
pairwise = nchoosek(0:9,2);
svmModel = cell(size(pairwise,1),1);
predTest = zeros(size(x_test, 1), numel(svmModel));


timer= tic;
for k = 1:numel(svmModel)
    idx = any(bsxfun(@eq,y_train,pairwise(k,:)),2);
    
    svmModel{k} = fitcsvm(x_train(idx,:),y_train(idx),'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto','BoxConstraint',2.8);
    
    predTest(:,k) = predict(svmModel{k},x_test);
    k;
    
end
toc(timer)
predictedy = mode(predTest,2);
[C,order] = confusionmat(y_test, predictedy)
results = sum(predictedy == y_test)/length(y_test)