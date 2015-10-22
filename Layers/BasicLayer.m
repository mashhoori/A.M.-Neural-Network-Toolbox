
classdef BasicLayer < ILayer
    
    properties
         w
    end
    
    methods
        function layer = BasicLayer(neuronCount, type, activationFunction)
            layer.neuronCount = neuronCount;
            layer.type = type;
            
            if(nargin < 3)
                layer.activationFunction = AF_HyperbolicTangant();
            else
                layer.activationFunction = activationFunction;
            end
            
            layer.shape = [neuronCount 1 1];
        end
        
        function UpdateInternalRepresentation(this)
           this.w = reshape(this.weights, this.previousLayer.neuronCount + 1, this.neuronCount) ;
        end
        
        function res = GetNumberOfParameters(this)
            res = (this.previousLayer.neuronCount + 1) * this.neuronCount;            
        end
        
        function InitRandomWeights(this)
            numParam = this.GetNumberOfParameters();
            this.trainingInfo = GetTrainingInfo(numParam);            
           % this.weights = (rand(numParam, 1)*2-1)*0.2;            
            this.weights = rand(numParam, 1)*0.01;
%             this.weights(1:this.previousLayer.neuronCount+1:numParam) = 0.5;
            this.UpdateInternalRepresentation();
        end
                        
        function CalculateOutput(this)
            %batchInput = reshape(this.previousLayer.output, [size(this.previousLayer.output,1) prod(this.previousLayer.shape)]);                        
            batchInput = reshape(this.previousLayer.output, size(this.previousLayer.output,1), []);                        
            batchInput = [ones(size(batchInput,1),1) batchInput];            
            netInput = batchInput * this.w;
            this.output = this.activationFunction.Calculate(netInput);           
            
            if(this.dropoutProb > 0 && this.inTraining)
                this.dropoutList = rand(size(this.output)) > this.dropoutProb;
                this.output = this.output .* this.dropoutList;
            elseif (this.dropoutProb > 0 && ~this.inTraining)
                this.output = this.output * (1-this.dropoutProb);
            end
        end
        
        function ComputeGradientAndBackpropagateErrorSignal(this)
            
            if(this.dropoutProb > 0)
                this.trainingInfo.error = this.trainingInfo.error .* this.dropoutList;
            end

            numIns = size(this.output, 1);
            derivative = this.activationFunction.CalculateDerivative(this.output);
            tmp = this.trainingInfo.error .* derivative;
            
            input = reshape(this.previousLayer.output, numIns, []);
            input = [ones(size(this.previousLayer.output,1),1) input];

            this.trainingInfo.gradient = (input' * tmp / numIns);            
            this.trainingInfo.gradient = this.trainingInfo.gradient(:);
            this.previousLayer.trainingInfo.error = tmp * this.w(2:end, :)'; 

            %this.previousLayer.trainingInfo.error = squeeze(reshape(this.previousLayer.trainingInfo.error, [numIns, this.previousLayer.shape]));
            this.previousLayer.trainingInfo.error = reshape(this.previousLayer.trainingInfo.error, [numIns, this.previousLayer.shape]);
            
        end
    end
    
end