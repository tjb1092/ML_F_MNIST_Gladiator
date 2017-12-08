function [X,images] = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

%Open data file
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

%Confirm valid file type
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

%Define some length values.
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

%Read images from file
images = fread(fp, inf, 'unsigned char');
% Reshape to #pixels x #examples
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

norm_images = mat2gray(images);  % Convert to normalized images.
X = reshape(norm_images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
X = double(X); %/ 255;

end
