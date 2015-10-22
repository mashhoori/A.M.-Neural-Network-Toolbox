function result = PredictNN(net, testDataset, fileName)    
    result = [];
    for i=1:numel(testDataset)
        testMat = testDataset(i).data;
        output = net.CalculateOutput(testMat);

        [~, I] = max(output, [], 2);
        result = [result; I(:)];        
    end
    
    result = result - 1;
    
    csvwrite(fileName, result);
end
    