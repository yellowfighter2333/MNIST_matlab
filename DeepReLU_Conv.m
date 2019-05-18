function [W1,W3,W4] = DeepReLU_Conv(train_images,train_labels,W1,W3,W4)
%the function includes both the method backprop and the forward direction
%calculation use once this function means finish one one big epoch
    alpha = 0.01;
    N = size(train_labels,2);
    acc = zeros(10000,1);
    error = zeros(10000,1);
    test_images = loadMNISTImages('t10k-images-idx3-ubyte');
    test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    for k = 1:N
        X = train_images(:,:,k);
        d = int_to_hotcode(train_labels(:,k));
        V1 = Conv2(X,W1);
        Y1 = ReLU(V1);
        Y2 = Pool(Y1);
        y2 = reshape(Y2,[],1);
        v3 = W3*y2;
        y3 = ReLU(v3);
        v = W4*y3;
        y = Softmax(v);
        
        e = d-y;
        delta = e;
        dW4 = alpha*delta*y3';

        
        
        e3 = W4'*delta;
        delta3 = (v3>0).*e3;
        dW3 = alpha*delta3*y2';
        
        e2 = W3'*delta3;
        E2 = reshape(e2,size(Y2));
        E1 = zeros(size(Y1));E2_4 = E2/4;
        E1(1:2:end,1:2:end,:) = E2_4;
        E1(1:2:end,2:2:end,:) = E2_4;
        E1(2:2:end,1:2:end,:) = E2_4;
        E1(2:2:end,2:2:end,:) = E2_4;
        delta1 = (V1>0).*E1;
        dW1 = alpha*Conv2(X,delta1); 
        
        W4 = W4 + dW4;
        W3 = W3 + dW3;
        W1 = W1 + dW1;
        [acc(k,:),error(k,:)] = validateTwoLayerPerceotion(test_images,test_labels,W1,W3,W4);
    end
        plot(acc)
end

