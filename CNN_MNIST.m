clear
train_images = loadMNISTImages('train-images-idx3-ubyte');
train_labels = loadMNISTLabels('train-labels-idx1-ubyte');
test_images = loadMNISTImages('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
for i = 1:60000
    if train_labels(:,i) == 0
        train_labels(:,i) = 10;
    end
end
for p = 1:10000
    if test_labels(:,p) == 0
        test_labels(:,p) = 10;
    end
end
%load the data from the fout original files

W1 = randn(9,9,20);
W3 = (2*rand(100,2000)-1)/20;
W4 = (2*rand(10,100)-1)/10;

epochs = 30;
for k = 1:epochs
    [W1,W3,W4] = DeepReLU_Conv(train_images,train_labels,W1,W3,W4);
    [acc,ErrorNumber] = validateTwoLayerPerceotion(test_images,test_labels,W1,W3,W4);
    fprintf('epochs:%d accuracy:%f Errornumber:%d\n',k,acc,ErrorNumber)
end




