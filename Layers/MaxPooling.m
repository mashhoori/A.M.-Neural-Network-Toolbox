classdef MaxPooling < ILayer

    properties
        poolingDim; 
        maxIndices;
    end

    methods
        function layer = MaxPooling(poolingDim, activationFunction)
            
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
            
            
            for i=1:length(this.poolingDim)
                this.shape(i) = this.previousLayer.shape(i) / this.poolingDim(i);                
            end
            
            for i=length(this.poolingDim)+1:length(this.previousLayer.shape)
                this.shape(i) = this.previousLayer.shape(i);
            end

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
            
            input = this.previousLayer.output;            
            a = size(input);                             
            
            this.output = zeros([a(1), this.shape]); 
            
            preShape = this.previousLayer.shape;
            this.maxIndices = zeros([a(1), preShape]);            
           
            preMapNrnCnt = preShape(1) * preShape(2);
            s1 = preMapNrnCnt/this.poolingDim(1);
            
            permInd = reshape(1:s1, this.shape(1), [])';
            permInd = permInd(:); 
            
            permInd2 = reshape(1:s1, s1/this.shape(1), [])';
            permInd2 = permInd2(:);
            
            permInd3 = reshape(1:prod(this.shape(1:2)), this.shape(1), [])';
            permInd3 = permInd3(:);
                     
            for k=1:a(4)

                preInp = input(:,:,:,k);
                preInp = reshape(preInp, a(1), this.poolingDim(1), []);
                preInp = preInp(:,:, permInd);
                preInp = reshape(preInp, a(1),prod(this.poolingDim), []);
                [preInp, ind] = max(preInp, [],2);
                preInp = preInp(:,:, permInd3);

                indMat = zeros(a(1), prod(this.poolingDim), preMapNrnCnt/prod(this.poolingDim));
                [r1, r2] = ndgrid(1:size(indMat, 1), 1:size(indMat, 3));
                indMat(sub2ind(size(indMat), r1(:), ind(:), r2(:))) = 1;                

                indMat = reshape(indMat, a(1), this.poolingDim(1), []);

                indMat = indMat(:,:, permInd2);
                indMat = reshape(indMat, a(1), this.previousLayer.shape(1), []);
                this.maxIndices(:,:,:,k) = reshape(indMat, a(1), this.previousLayer.shape(1), []);
                this.output(:,:,:,k) = reshape(preInp, a(1), this.shape(1), []);

            end           
                      
        end
        
        function ComputeGradientAndBackpropagateErrorSignal(this)            
            this.previousLayer.trainingInfo.error = myKron(this.trainingInfo.error, [1 this.poolingDim 1])/prod(this.poolingDim);            
            this.previousLayer.trainingInfo.error = this.previousLayer.trainingInfo.error .* this.maxIndices;
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