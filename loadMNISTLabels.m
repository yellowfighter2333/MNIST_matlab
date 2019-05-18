function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]*1 matrix containing
%the labels for MNIST images

fp = fopen(filename,'rb');
assert(fp ~= -1,['could not open',filename,'']);

magic = fread(fp,1,'int32',0,'ieee-be');
assert(magic == 2049,['Bad number in',filename,'']);

numLabels = fread(fp,1,'int32',0,'ieee-be');

labels = fread(fp,inf,'unsigned char');

assert(size(labels,1) == numLabels,'Mismatch in label conunt');
labels = permute(labels,[2 1]);
fclose(fp);

end