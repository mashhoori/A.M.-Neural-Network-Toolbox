classdef MeanPooling < ILayer

    properties
        poolingDim;        
    end

    methods
        function layer = MeanPooling(poolingDim, activationFunction)
            
            layer.poolingDim = poolingDim;
            if(nargin < 2)
                layer.activationFunction = AF_Linear();
            else
                layer.activationFunction = activationFunction;
            end
            
            layer.type = 'pooling';
            
        end
        
        function InitRandomWeights(this)

            if (isempty(this.previousLayer))
                error('Previous layer have to be defined!');
            end
            
            this.shape = [0 0 0];
            this.shape(1) = this.previousLayer.shape(1) / this.poolingDim(1);
            this.shape(2) = this.previousLayer.shape(2) / this.poolingDim(2);
            this.shape(3) = this.previousLayer.shape(3);

            this.neuronCount = prod(this.shape);

            numParam = this.GetNumberOfParameters();
            this.trainingInfo = GetTrainingInfo(numParam);
            this.weights = 0; 
            this.UpdateInternalRepresentation();
            
        end
    
        function res = GetNumberOfParameters(this)
            res = 1;    
        end
        
        function CalculateOutput(this)            
            this.output = convn(this.previousLayer.output, ones([1 this.poolingDim 1])/prod(this.poolingDim), 'valid');
            this.output = this.output(:, 1:this.poolingDim(1):end, 1:this.poolingDim(2):end, :);            
        end
        
        function ComputeGradientAndBackpropagateErrorSignal(this)            
            this.previousLayer.trainingInfo.error = myKron(this.trainingInfo.error, [1 this.poolingDim 1])/prod(this.poolingDim);            
        end
        
        function UpdateInternalRepresentation(this)
           
        end
    end
    
end


function B = myKron(A, S)

B = repmat(A, S);

s1 = zeros(S(2), size(A, 2));
for i=1:S(2)
    s1(i, :) = [1:size(A, 2)] + (i-1)*size(A, 2);
end

s2 = zeros(S(3), size(A, 3));
for i=1:S(3)
    s2(i, :) = [1:size(A, 3)] + (i-1)*size(A, 3);
end

s1 = s1(:);
s2 = s2(:);

B = B(:, s1, s2, :);

end