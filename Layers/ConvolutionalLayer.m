classdef ConvolutionalLayer < ILayer
    
    properties
        kernels;
        biases;
        kernelShape;
        numParametersForKernel
        numKernel
    end
    
    methods
        
        function layer = ConvolutionalLayer(kernelShape, numKernel, activationFunction)
            layer.type = 'convolutional';
            layer.kernelShape = kernelShape;
            layer.numKernel = numKernel;
            
            if(nargin < 3)
                layer.activationFunction = AF_Sigmoid();
            else
                layer.activationFunction = activationFunction;
            end
        end
        
        function InitRandomWeights(this)
            
            if (isempty(this.previousLayer))
                error('Previous layer have to be defined');
            end
            
            for i=1:length(this.kernelShape)                
                if(this.kernelShape(i) ~= 0)                    
                    this.shape(i) = this.previousLayer.shape(i) - this.kernelShape(i) + 1;                
                else                    
                    if( numel(this.previousLayer.shape) < i)                    
                        this.kernelShape(i) = 1;
                    else
                        this.kernelShape(i) = this.previousLayer.shape(i);
                    end
                end
            end
            this.shape(end+1) = this.numKernel;            
          
            this.neuronCount = prod(this.shape);            
            this.numParametersForKernel = prod(this.kernelShape);
            
            numParam = this.GetNumberOfParameters();
            this.trainingInfo = GetTrainingInfo(numParam);            
            %this.weights = (rand(numParam, 1)*2-1)*0.2;   
            this.weights = rand(numParam, 1)*0.05;
            this.UpdateInternalRepresentation();
        end
        
        function res = GetNumberOfParameters(this)
            res = (this.numParametersForKernel + 1) * this.numKernel;
        end
        
        function UpdateInternalRepresentation(this)
            
            this.biases = zeros(1, this.numKernel);
            this.kernels = cell(1, this.numKernel);
            
            for i=1:this.numKernel
                bIndex = (i-1)*(this.numParametersForKernel+1)+1;
                this.biases(i) =  this.weights(bIndex);                
                this.kernels{i} = reshape(this.weights(bIndex+1:bIndex+this.numParametersForKernel), this.kernelShape);
            end
        end
        
        function CalculateOutput(this)
            input = this.previousLayer.output;
            
            inpShape = size(input);
            numIns = inpShape(1);
            this.output = zeros([numIns, this.shape]);
            
            for i=1:this.numKernel
                
                if (strcmp(this.previousLayer.type, 'input'))
                    this.output(:, :, :, i) = convn(input, reshape(rot90(this.kernels{i}, 2), [1 size(this.kernels{i})]), 'valid');
                else
                    krnl  = this.kernels{i}(this.kernelShape(1):-1:1, this.kernelShape(2):-1:1, this.kernelShape(3):-1:1);
                    krnl = reshape(krnl, [1 size(krnl)]);
                    this.output(:, :, :, i) = convn(input, krnl, 'valid');
                end
                this.output(:, :, :, i) = this.output(:, :, :, i) + this.biases(i);
            end
            
            this.output = this.activationFunction.Calculate(this.output);           
            
            if(this.dropoutProb > 0 && this.inTraining)
                this.dropoutList = rand(numIns, this.numKernel) > this.dropoutProb;                
                [r, c] = find(1 - this.dropoutList);
                for i=1:numel(r)
                    this.output(r(i),:,:, c(i)) = this.output(r(i),:,:, c(i)) * 0;
                end
            elseif (this.dropoutProb > 0 && ~this.inTraining)
                this.output = this.output * (1-this.dropoutProb);
            end
            
        end
        
        function ComputeGradientAndBackpropagateErrorSignal(this)
            
            a = size(this.output);
            numIns = a(1);
            
            if(this.dropoutProb > 0)                
                [r, c] = find(1 - this.dropoutList);
                for i=1:numel(r)                    
                    this.trainingInfo.error(r(i),:,:, c(i)) = this.trainingInfo.error(r(i),:,:, c(i)) * 0;
                end                                                                
            end            
           
            derivative = this.activationFunction.CalculateDerivative(this.output);
            tmp = this.trainingInfo.error .* derivative;
            
            input = this.previousLayer.output;
            preLyrShape = this.previousLayer.shape;
            
            bGrad = sum(sum(mean(tmp, 1), 2), 3);
            bGrad = bGrad(:);
            
            q  = tmp(numIns:-1:1, this.shape(1):-1:1, this.shape(2):-1:1, :);                
            
            for i=1:this.numKernel
                this.trainingInfo.gradient((i-1)*(this.numParametersForKernel+1)+1) = bGrad(i);
                kGrad = zeros(size(this.kernels{i}));                
                                
                if (strcmp(this.previousLayer.type, 'input'))
                    kGrad = convn(input, q(:,:,:,i), 'valid');                    
                else
                    for j=1:preLyrShape(3)                        
                        kGrad(:,:,j) = convn(input(:, :, :, j), q(:,:,:,i), 'valid');
                    end
                end
                                                
                kGrad = kGrad / numIns;
                this.trainingInfo.gradient((i-1)*(this.numParametersForKernel+1)+2:i*(this.numParametersForKernel+1) ) = kGrad(:);
            end
            
            if (~strcmp(this.previousLayer.type, 'input'))
                this.previousLayer.trainingInfo.error = zeros(size(this.previousLayer.output));
                
                for i=1:preLyrShape(3)
                    for j=1:this.numKernel                        
                        this.previousLayer.trainingInfo.error(:, :, :, i) = this.previousLayer.trainingInfo.error(:, :, :, i) + convn(tmp(:,:,:,j), reshape(this.kernels{j}(:,:,i), [1, this.kernelShape(1), this.kernelShape(2)] ));                        
                    end
                end
                
            end
        end 
        
    end
end
