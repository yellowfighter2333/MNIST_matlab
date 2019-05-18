function V = Conv2(X,W1)
% the function ouput the result after using filter group
    N = size(W1,3);
    for k = 1:N
        V(:,:,k) = conv2(X,W1(:,:,k),'valid');
    end
end
