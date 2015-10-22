classdef HybridNetwork < handle    
    
    properties
        nets
        numInputJoin = 0;
        numOfNets
        numOfOutputs
        numOfExtraLayers
        extraLayersShape
        joinLayerIndices        
        indexMatrix;
        layers = BasicLayer.empty();
        costType
    end
    
    methods
        
        function net = HybridNetwork(nets, joinLayerIndices, extraLayersShape, softMax)
            net.nets = nets;
            net.extraLayersShape = extraLayersShape;
            net.numOfNets = numel(nets);
            net.joinLayerIndices = joinLayerIndices;
            net.numOfExtraLayers = 1 + numel(extraLayersShape);
            net.numOfOutputs = extraLayersShape(end);   
            net.indexMatrix = zeros(net.numOfNets, 2);
            
            if(softMax)
                net.costType = 'CrossEntropy';
            else
                net.costType = 'EuclideanDistance';
            end
            
            net.InitNetwork();
        end
        
        function InitNetwork(this)
           
            for i=1:this.numOfNets  
                n = prod(this.nets{i}.layers(this.joinLayerIndices(i)).shape);
                this.numInputJoin = this.numInputJoin + n;
                
                if(i == 1)
                    this.indexMatrix(i, 1) = 1;
                else
                    this.indexMatrix(i, 1) = this.indexMatrix(i-1, 2) + 1;
                end
                
                this.indexMatrix(i, 2) = this.indexMatrix(i, 1) + n -1 ;                
            end                        
            
            index = 1;
            inputLayer = InputLayer(this.numInputJoin);
%           inputLayer.dropoutProb = 0.2;
            inputLayer.index = index;
            this.layers(index) = inputLayer;
            index = index + 1;
            
            for i = 1:this.numOfExtraLayers - 2
                hidden = BasicLayer(this.extraLayersShape(i), 'hidden');
                hidden.index = index;
                hidden.activationFunction = AF_HyperbolicTangant;
%               hidden.dropoutProb = 0.5;
                this.layers(index) = hidden;
                hidden.previousLayer = this.layers(index - 1);
                index = index + 1;
            end
            
            outputLayer = BasicLayer(this.numOfOutputs, 'Output');
            if(strcmp(this.costType,'CrossEntropy'))
                outputLayer = SoftMaxLayer(this.numOfOutputs);
            end
            
            outputLayer.index = index;
            this.layers(index) = outputLayer;
            outputLayer.previousLayer = this.layers(index - 1);
            
            for i = 2:this.numOfExtraLayers
                this.layers(i).InitRandomWeights();
            end
            
        end
        
        function SetActivationFunction(this, func) 
            for i = 2:numel(this.layers)
                this.layers(i).activationFunction = func;
            end
        end          
        
        function SetInTrainingFlag(this, flag)
            
            for i=1:this.numOfNets
                this.nets{i}.SetInTrainingFlag(flag);
            end            
            
            for i = 1:numel(this.layers)
                this.layers(i).inTraining = flag;
            end
        end

        function res = CalculateOutput(this, input)  
            
            for i=1:this.numOfNets
                this.nets{i}.layers(1).input = input{i};
                
                for j = 1:this.joinLayerIndices(i)
                    this.nets{i}.layers(j).CalculateOutput();
                end
            end
            
            shp = size(input{1});
            numIns = shp(1);
            a = zeros( numIns, this.numInputJoin);            
            for i=1:this.numOfNets                
                a(:, this.indexMatrix(i,1):this.indexMatrix(i,2)) = this.nets{i}.layers(this.joinLayerIndices(i)).output;
            end   
            
            this.layers(1).input = a;
            for i = 1:numel(this.layers)
                this.layers(i).CalculateOutput();
            end
                      
            res = this.layers(end).output;
        end

        function UpdateWeights(this)  
            for i = 2: numel(this.layers)
                this.layers(i).UpdateWeights();
            end
            
            for i=1:this.numOfNets
                for j = 2:this.joinLayerIndices(i)
                    this.nets{i}.layers(j).UpdateWeights();
                end
            end
        end

        function res = GetNumberOfParameters(this)
             res = 0;
%            for i = 2: numel(this.layers)
%                  res = res + this.layers(i).GetNumberOfParameters();
%            end            
        end       

        function TrainWithMomentum(this, dataset, valid, trainSize, maxIter)  
            if (trainSize == 0)
                trainSize = dataset.NumInstance;
            end

            momentum = 0.9;
            learningRate = 0.01;            
            reg = 0.00;
                       
            for iter = 1:maxIter         
                tic;
                cnt = 0;
                costValue = 0;
                
                this.SetInTrainingFlag(1);
                for bc = 1:numel(dataset)
                    
                    batch = dataset(bc);
                    cnt = cnt + batch.numInstance;                     
                    
                    [cV, errorMat] = GetCostandErrorforDataset(this, batch);
                    costValue = costValue + cV;
                    
                    this.layers(end).trainingInfo.error = errorMat;
                    
                    
                    for i = this.numOfExtraLayers:-1:1
                        this.layers(i).ComputeGradientAndBackpropagateErrorSignal();
                    end
                    for i=1:this.numOfNets
                        this.nets{i}.layers(this.joinLayerIndices(i)).trainingInfo.error = this.layers(1).trainingInfo.error(:, this.indexMatrix(i,1):this.indexMatrix(i,2));
                        
                        for j = this.joinLayerIndices(i):-1:2
                            this.nets{i}.layers(j).ComputeGradientAndBackpropagateErrorSignal();
                        end
                    end
                  
                    for cc=2:this.numOfExtraLayers
                        this.layers(cc).trainingInfo.deltaWeight = -learningRate * (this.layers(cc).trainingInfo.gradient - reg * this.layers(cc).weights) + momentum * this.layers(cc).trainingInfo.deltaWeight_previous;
                    end 
                    for i= 1:this.numOfNets                        
                        for j = 2:this.joinLayerIndices(i)
                            this.nets{i}.layers(j).trainingInfo.deltaWeight = -learningRate * (this.nets{i}.layers(j).trainingInfo.gradient - reg * this.nets{i}.layers(j).weights) + momentum * this.nets{i}.layers(j).trainingInfo.deltaWeight_previous;
                        end
                    end                  

                    this.UpdateWeights();                   

                    if (mod(cnt, 5000) == 0)
                        fprintf([num2str(cnt), '.  ']);
                    end
                    
                    if (cnt >= trainSize)
                        break;
                    end                   
                                        
                end            

                fprintf(['Iteration ', num2str(iter), ': ', num2str(costValue/ cnt),'   Elapsed Time: ', num2str(toc), '   \n']);                     
                this.SetInTrainingFlag(0);
                
                if(~isempty(valid))
                    EvaluateNN({this}, valid, 'AUC');                
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