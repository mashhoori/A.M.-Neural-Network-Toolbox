classdef ChannelConvolutionalLayer < ILayer
    
    properties
        kernels;
        biases;
        kernelShape;
        numParametersForKernel
        numKernelPerChannel
        numKernel;
        channelDim
    end
    
    methods
        
        function layer = ChannelConvolutionalLayer(kernelShape, channelDim, numKernelPerChannel, activationFunction)
            layer.type = 'convolutional';
            
            layer.kernelShape = kernelShape;      
            layer.channelDim  = channelDim;
            layer.numKernelPerChannel = numKernelPerChannel;
            
            if(nargin < 4)
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
            
            
            
            this.shape(1) = this.previousLayer.shape(1) - this.kernelShape(1) + 1;
            this.shape(2) = this.numKernelPerChannel;%this.previousLayer.shape(2) - this.kernelShape(2) + 1;
            this.numKernel = this.numKernelPerChannel * this.previousLayer.shape(3);            
            this.shape(3) = this.previousLayer.shape(3);           
            
            this.kernelShape = [this.kernelShape 1];              
            this.neuronCount = prod(this.shape);
            
            this.numParametersForKernel = prod(this.kernelShape);
            
            numParam = this.GetNumberOfParameters();
            this.trainingInfo = GetTrainingInfo(numParam);
            
            this.weights = (rand(numParam, 1)*2-1)*0.2;
           % this.weights = rand(numParam, 1)*0.01;
            
            this.UpdateInternalRepresentation();
        end
        
        function res = GetNumberOfParameters(this)
            res = (this.numParametersForKernel + 1) * this.numKernelPerChannel  * this.shape(3);             
        end
        
        function UpdateInternalRepresentation(this)
            
            this.biases = zeros(this.shape(3), this.numKernelPerChannel);
            this.kernels = cell(this.shape(3), this.numKernelPerChannel);            
            
            for i=1:this.numKernel
                bIndex = (i-1)*(this.numParametersForKernel+1)+1;
                
                chN = ceil(i/ this.numKernelPerChannel);
                kN  = mod(i, this.numKernelPerChannel);
                if(kN == 0)
                    kN = this.numKernelPerChannel;
                end
                
                this.biases( kN, chN) =  this.weights(bIndex);
                this.kernels{ kN, chN} = reshape(this.weights(bIndex+1:bIndex+this.numParametersForKernel), this.kernelShape(1), this.kernelShape(2), this.kernelShape(3));
            end
            
        end
        
        function CalculateOutput(this)
            input = this.previousLayer.output;
            
            a = size(input);
            numIns = a(1);
            %this.output = zeros(numIns, this.shape(1), this.shape(2), this.shape(3));
            this.output = zeros(numIns, this.shape(1), this.numKernelPerChannel, this.shape(3));
                        
            for nC=1:this.shape(3) 
                for nK = 1:this.numKernelPerChannel
                
                    krnl  = this.kernels{nK, nC}(this.kernelShape(1):-1:1, this.kernelShape(2):-1:1, this.kernelShape(3):-1:1);                   
                    this.output(:, :, nK, nC) = convn(input(:, :, 1, nC), reshape(krnl, [1 size(krnl)]), 'valid');  
                    this.output(:, :, nK, nC) = this.output(:, :, nK, nC) + this.biases(nK, nC);  
                end
            end
            this.output = this.activationFunction.Calculate(this.output);   
            
            if(this.dropoutProb > 0 && this.inTraining)
                this.dropoutList = rand(numIns, this.numKernel) > this.dropoutProb;                
                [r, c] = find(1 - this.dropoutList);
                
                for i=1:numel(r) 
                    
                    chN = ceil(c(i)/ this.numKernelPerChannel);
                    kN  = mod(c(i), this.numKernelPerChannel);
                    if(kN == 0)
                        kN = this.numKernelPerChannel;
                    end                    
                    this.output(r(i),:,kN, chN) = this.output(r(i),:,kN, chN) * 0;
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
                    
                    chN = ceil(c(i)/ this.numKernelPerChannel);
                    kN  = mod(c(i), this.numKernelPerChannel);
                    if(kN == 0)
                        kN = this.numKernelPerChannel;
                    end                    
                    
                    this.trainingInfo.error(r(i),:,kN, chN) = this.trainingInfo.error(r(i),:,kN, chN) * 0;
                end                                                                
            end
           
            derivative = this.activationFunction.CalculateDerivative(this.output);
            tmp = this.trainingInfo.error .* derivative;
            
            input = this.previousLayer.output;
            preLyrShape = this.previousLayer.shape;
            
            bGrad = squeeze(sum(mean(tmp, 1), 2));
%             bGrad = bGrad(:);
            
            q  = tmp(numIns:-1:1, this.shape(1):-1:1, :, :);                
            
            for i=1:this.numKernel
                
                chN = ceil(i/ this.numKernelPerChannel);
                kN  = mod(i, this.numKernelPerChannel);
                if(kN == 0)
                    kN = this.numKernelPerChannel;
                end
                
                this.trainingInfo.gradient((i-1)*(this.numParametersForKernel+1)+1) = bGrad(kN, chN); 
                kGrad = convn(input(:, :, 1, chN), q(:,:,kN,chN), 'valid');                                               
                                                
                kGrad = kGrad / numIns;
                this.trainingInfo.gradient((i-1)*(this.numParametersForKernel+1)+2:i*(this.numParametersForKernel+1) ) = kGrad(:);
            end
            
            if (~strcmp(this.previousLayer.type, 'input'))
                this.previousLayer.trainingInfo.error = zeros(size(this.previousLayer.output));
                
                for i=1:preLyrShape(3)
                    for j=1:this.numKernel  
                        
                        chN = ceil(j/ this.numKernelPerChannel);
                        kN  = mod(j, this.numKernelPerChannel);
                        if(kN == 0)
                            kN = this.numKernelPerChannel;
                        end
                        
                        this.previousLayer.trainingInfo.error(:, :, 1, i) = this.previousLayer.trainingInfo.error(:, :, 1, i) + convn(tmp(:,:,kN,chN), reshape(this.kernels{kN,chN}, [1, this.kernelShape(1)] ));                        
                    end
                end
                
            end
        end 
        
        
        
    end
end
