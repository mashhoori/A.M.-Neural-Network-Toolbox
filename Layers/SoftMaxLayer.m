

classdef SoftMaxLayer < BasicLayer
   
    methods
        function layer = SoftMaxLayer(neuronCount)
            layer = layer@BasicLayer(neuronCount, 'output', AF_Linear());
        end
        
        function CalculateOutput(this)                    
            batchInput = reshape(this.previousLayer.output, size(this.previousLayer.output,1), []);
            batchInput = [ones(size(batchInput,1),1) batchInput];
            netInput = batchInput * this.w;
            
            maxValues = max(netInput, [], 2);
            netInput = bsxfun(@minus, netInput, maxValues);
            netInput = exp(netInput);
            this.output = bsxfun(@rdivide, netInput, sum(netInput, 2));           
        end 
    end
    
end