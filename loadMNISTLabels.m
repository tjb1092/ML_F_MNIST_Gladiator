function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

%Open Data File
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

%Confirm File Type
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

%Define length
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

%Read off the labels from the file.
labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end
