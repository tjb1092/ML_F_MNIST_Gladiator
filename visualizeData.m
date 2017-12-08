function [ output_args ] = visualizeData( X, y, images)
% Performs several methods of data visualization. First, the images are
% stitched together in a grid to display a sample of our data set. Then, we
% utilized Laurens van der Maaten's Dimensionality-Reduction toolbox to
% perform PCA and t-SNE on our dataset.

images = mat2gray(images);  % Convert to normalized images.
output_args = 1;
[~,~,d] = size(images);
index = randi(d,16,16); %pick a few images to plot
im_grid = [];
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
       im_grid = [im_grid ; temp];
    else
       im_grid = temp;
    end
end

imshow(im_grid);
title('Sample images from the Fashion-MNIST Dataset');
clear grid; % Clears variable so that grid can be shown on plots

timing = tic;
%Perform PCA on the dataset to vizualize the clustering of the data.
display('Performing PCA');
[mapped_data, mapping] = compute_mapping(X,'PCA',2);
toc(timing);
figure, gscatter(mapped_data(:,1),mapped_data(:,2),y,[],[],10);
title('PCA');
legend('T-shirt/Top (0)','Trousers (1)','Pullover (2)','Dress (3)',...
    'Coat (4)','Sandal (5)','Shirt (6)','Sneaker (7)','Bag (8)','Ankle Boot (9)');
xlabel('X_1'); ylabel('X_2');
grid on;

timing = tic;
display('Performing t-SNE');
%Perform t-sne on the dataset to vizualize the clustering of the data.
[mapped_data, mapping] = compute_mapping(X,'tSNE',2);
toc(timing);
figure, gscatter(mapped_data(:,1),mapped_data(:,2),y,[],[],[],40);
title('t-SNE');
legend('T-shirt/Top (0)','Trousers (1)','Pullover (2)','Dress (3)',...
    'Coat (4)','Sandal (5)','Shirt (6)','Sneaker (7)','Bag (8)','Ankle Boot (9)');
xlabel('X_1'); ylabel('X_2');
grid on;
end

