function Y = Pool(x)
    Y = (x(1:2:end,1:2:end,:)+x(2:2:end,1:2:end,:)+x(1:2:end,2:2:end,:)+x(2:2:end,2:2:end,:))/4;
end
        
        
