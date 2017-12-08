function [ output_args ] = visualizeData( X, y, images)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

output_args = 1;
[~,~,d] = size(images);
index = randi(d,16,16); %pick a few images to plot
grid = [];
for j = 1:16
    for k = 1:16
        if(k ~= 1)
           %don't do this for the first iteration.
            temp = [temp, images(:,:,index(j,k))];
        else
            temp = images(:,:,index(j,1));
        end
    end
    if j ~= 1
       grid = [grid ; temp];
    else
       grid = temp;
    end
end

imshow(grid);
title('Sample images from the fashion-MNIST Dataset');
clear grid; % Clears variable so that grid can be shown on plots

timing = tic;
%Perform PCA on the dataset to vizualize the clustering of the data.
display('Performing PCA');
[mapped_data, mapping] = compute_mapping(X,'PCA',2);
toc(timing);
figure, gscatter(mapped_data(:,1),mapped_data(:,2),y,[],[],10);
title('PCA');
legend('T-shirt/Top','Trousers','Pullover','Dress','Coat','Sandal','Shirt',...
    'Sneaker','Bag','Ankle Boot');
xlabel('X_1'); ylabel('X_2');
grid on;

timing = tic;
display('Performing T-SNE');
%Perform t-sne on the dataset to vizualize the clustering of the data.
[mapped_data, mapping] = compute_mapping(X,'tSNE',2);
toc(timing);
figure, gscatter(mapped_data(:,1),mapped_data(:,2),y,[],[],[],10);
title('T-SNE');
legend('T-shirt/Top','Trousers','Pullover','Dress','Coat','Sandal','Shirt',...
    'Sneaker','Bag','Ankle Boot');
xlabel('X_1'); ylabel('X_2');
grid on;
end

