function [error, output, pred] = EvaluateNN(nets, validDataset, method)
    
    numClassifiers = numel(nets);

    switch(method)                       
        case 'accuracy'
            numError = 0;
            numIns = 0;
            
            for i=1:numel(validDataset)
                validMat = validDataset(i).input;
                
                result = nets{1}.CalculateOutput(validMat);               
                for nc = 2:numClassifiers
                    result = result + net.CalculateOutput(validMat);                                  
                end
                
                [~, I] = max(result, [], 2);
                I = I(:);
                numError = numError + sum(((I-1) ~= validDataset(i).label(:)));        
                numIns = numIns + validDataset(i).numInstance;
            end
            error = numError/numIns;
            fprintf(['Error is ', num2str(error), '\n']);
            
            
            
        case 'AUCFuck'
            target = [];             
            
            preds = cell(1, numClassifiers);
            for nc = 1:numClassifiers
                preds{nc} = [];
            end
            
            for i=1:numel(validDataset)
                validMat = validDataset(i).input;
                target = [target; validDataset(i).output];                 
                
                for nc = 1:numClassifiers  
                    result = nets{nc}.CalculateOutput(validMat);
                    preds{nc} = [preds{nc}; result];                                   
                end
            end
            
            pred = zeros(size(preds{1}));
            for nc = 1:numClassifiers
                pred = pred + preds{nc};
            end
            pred = pred / numClassifiers;            
            
            Evaluate(target, pred, 1:6);        
        
            
        case 'distance'
            totalError = zeros(size(validDataset(1).output, 1), 1);
            numIns = 0;
            
            for i=1:numel(validDataset)
                validMat = validDataset(i).input;
                
                result = nets{1}.CalculateOutput(validMat);               
                for nc = 2:numClassifiers
                    result = result + net.CalculateOutput(validMat);                                  
                end
                
                result = result / numClassifiers;
                error = (result - validDataset(i).output) .^ 2;
                error = sum(error, 2);
                totalError = totalError + error;                
                numIns = numIns + validDataset(i).numInstance;
            end
            error = sqrt(totalError/numIns);
            fprintf(['Error is ', num2str(error), '\n']);            
            
            
        case 'AUC'
            target = [];             
            
            preds = cell(1, numClassifiers);
            for nc = 1:numClassifiers
                preds{nc} = [];
            end
            
            for i=1:numel(validDataset)
                validMat = validDataset(i).input;
                target = [target; validDataset(i).label(:)];                 
                
                for nc = 1:numClassifiers  
                    result = nets{nc}.CalculateOutput(validMat);
                    preds{nc} = [preds{nc}; result];                                   
                end
            end
            
            pred = zeros(size(preds{nc}, 1), 1);
            for nc = 1:numClassifiers
                pred = pred + preds{nc}(:, 2);
            end
            pred = pred / numClassifiers;
            
            [~, ~, ~, auc] = perfcurve(target, pred, 1);                 
            fprintf(['AUC is ', num2str(auc), '\n']);
    
    end
    
    
end

