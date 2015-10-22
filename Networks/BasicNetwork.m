
classdef BasicNetwork < handle
        
    properties 
        numOfInputs;        
        numOfOutputs;
        numOfLayers;               
        layers = BasicLayer.empty();        
        costType;
    end

    methods        
        function SetActivationFunction(this, func) 
            for i = 2:numel(this.layers)
                this.layers(i).activationFunction = func;
            end
        end          
        
        function SetInTrainingFlag(this, flag)
            for i = 1:numel(this.layers)
                this.layers(i).inTraining = flag;
            end
        end

        function res = CalculateOutput(this, input)  

            this.layers(1).input = input;
            for i = 1:numel(this.layers)
                this.layers(i).CalculateOutput();
            end

            res = this.layers(end).output;
        end

        function UpdateWeights(this)  
            for i = 2: numel(this.layers)
                this.layers(i).UpdateWeights();
            end
        end

        function res = GetNumberOfParameters(this) 
            res = 0;
            for i = 2: numel(this.layers)
                res = res + this.layers(i).GetNumberOfParameters();
            end            
        end       

        function TrainWithMomentum(this, dataset, valid, trainSize, maxIter)  
            if (trainSize == 0)
                trainSize = dataset.NumInstance;
            end

            momentum = 0.9;
            learningRate = 0.002;           
            reg = 0.00;
                       
            for iter = 1:maxIter         
                tic;
                cnt = 0;
                costValue = 0;
                
                this.SetInTrainingFlag(1);
                for bc = 1: numel(dataset)                   
                    
                    batch = dataset(bc);                    
                    cnt = cnt + batch.numInstance;                     
                    
                    [cV, errorMat] = GetCostandErrorforDataset(this, batch);
                    costValue = costValue + cV;
                    
                    this.layers(end).trainingInfo.error = errorMat;
                    for i = this.numOfLayers:-1:2
                        this.layers(i).ComputeGradientAndBackpropagateErrorSignal();
                    end
                  
                    for cc=2:this.numOfLayers
                        this.layers(cc).trainingInfo.deltaWeight = -learningRate * (this.layers(cc).trainingInfo.gradient - reg * this.layers(cc).weights) + momentum * this.layers(cc).trainingInfo.deltaWeight_previous;
                    end 

                    this.UpdateWeights();                   

                    if (mod(cnt, 20000) == 0)
                        fprintf([num2str(cnt), '.  ']);
                    end
                    
                    if (cnt >= trainSize)
                        break;
                    end                   
                                        
                end            

                 fprintf(['Iteration ', num2str(iter), ': ', num2str(costValue/ cnt),'   Elapsed Time: ', num2str(toc), '   \n']);                     

                this.SetInTrainingFlag(0);
                
                if(~isempty(valid))
                    EvaluateNN({this}, valid, 'accuracy');
                end
                
                %this.WriteWeightsToFile("Net4", iter.ToString());                
            end             
        end        
      
        function [costValue, errorMat] = GetCostandErrorforDataset(this, dataset)        
            
            costValue = 0;
            outputMatrix = this.CalculateOutput(dataset.input);
            targets = dataset.output;
            errorMat = outputMatrix - targets;

            switch (this.costType)            
                case 'CrossEntropy'
                    outputMatrix(outputMatrix == 0) = 1e-20;
                    costValue = sum(sum(-1 * targets .* log(outputMatrix)));                      
                    
                case 'EuclideanDistance'
                    costValue = sum(sum(errorMat.*errorMat));  
                    
                case 'AUC'
                    [~, ~, ~, auc] = perfcurve(dataset.label, outputMatrix(:, 2), 1);
                    costValue = auc;                    
            end
        end
    end            
end
 

