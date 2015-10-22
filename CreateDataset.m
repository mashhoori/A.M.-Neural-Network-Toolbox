
function dataset = CreateDataset(data, label, numClass, batchSize, loadData, twoD)

    numIns = size(data, 1);
    numBatches = numIns/batchSize;    
%     
    if(~isempty(label) && numIns ~= length(label))
        error('');
    end
    
    if(~loadData)
        mn = mean(data, 1);
        stdVar = std(data);
        save mn mn
        save stdVar stdVar
    else
        mn = load('mn');
        mn = mn.mn;
        
        stdVar = load('stdVar');
        stdVar = stdVar.stdVar;
    end
    
    data = bsxfun(@minus, data, mn); 
    data = bsxfun(@rdivide, data, max(stdVar,eps));

    for i=1:numBatches
        dataset(i).input = single(data((i-1)*batchSize+1:i*batchSize, :));
        
        if(twoD)
            dataset(i).input = reshape(dataset(i).input, size(dataset(i).input, 1), 28,28);        
        end
        
        if(~isempty(label) && numClass > 0)
            dataset(i).label = label((i-1)*batchSize+1:i*batchSize);
            dataset(i).output = dummyvar(dataset(i).label+1);
            if(size(dataset(i).output, 2) < numClass)
                dataset(i).output = [dataset(i).output, zeros(size(dataset(i).output, 1), numClass - size(dataset(i).output, 2))];
            end
        elseif(numClass == 0)
            dataset(i).output = dataset(i).input;%single(label((i-1)*batchSize+1:i*batchSize, :));
        end
        
        dataset(i).numInstance = batchSize;
    end

end