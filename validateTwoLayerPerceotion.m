function [acc,classificationErrors] = validateTwoLayerPerceotion(TestValue,TestLabels,W1,W3,W4)

    N = size(TestLabels,2);
    d_comp = zeros(1,N);
    for k = 1:N
        X = TestValue(:,:,k);
        V1 = Conv2(X,W1);
        Y1 = ReLU(V1);
        Y2 = Pool(Y1);
        y2 = reshape(Y2,[],1);
        v3 = W3*y2;y3 = ReLU(v3);
        v = W4*y3;y = Softmax(v);
        [~,i] = max(y);
        d_comp(k) = i;
    end
    TrueNumber = sum(d_comp==TestLabels);
    classificationErrors = N - TrueNumber;
    acc = TrueNumber/N;
end
