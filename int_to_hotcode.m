function vec = int_to_hotcode(X)
%transfer int type to hotcode
        vec = zeros(10,1);
        vec(X,1) = 1;

end