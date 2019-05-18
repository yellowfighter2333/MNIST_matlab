function images = loadMNISTImages(filename)

%loadMNIST returns a 28*28*[number of MNIST images] matrix containing

fp = fopen(filename,'rb');
assert(fp ~= -1,['could not open',filename,'']);

magic = fread(fp,1,'int32',0,'ieee-be');
assert(magic == 2051,['Bad magic number in ',filename,'']);

numImages = fread(fp,1,'int32',0,'ieee-be');
numRows = fread(fp,1,'int32',0,'ieee-be');
numCols = fread(fp,1,'int32',0,'ieee-be');

images = fread(fp,inf,'unsigned char');
images = reshape(images,numCols,numRows,numImages);
images = permute(images,[2 1 3]);

fclose(fp);


images = double(images)/255;

end
