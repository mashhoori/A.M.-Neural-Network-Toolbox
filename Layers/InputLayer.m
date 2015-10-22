
classdef InputLayer < ILayer
    
    properties
        input
    end
        
    
    methods
        function layer = InputLayer(shape)
            layer.neuronCount = prod(shape);
            layer.type = 'input';            
            
            layer.shape = shape;
        end
        
        function this = set.input(this, value)
            iS = size(value);
            
            if(length(iS)-1 ~= length(this.shape))
                error('')
            elseif (~all(iS(2:end) == this.shape))
                error('');                
            end
                
            this.input = value;
        end
        
        function UpdateInternalRepresentation(this)
           
        end
        
        function res = GetNumberOfParameters(this)
            res = 1;
        end
        
        function InitRandomWeights(this)
            
        end
                        
        function CalculateOutput(this)
            this.output = this.input;       
            
            if(this.dropoutProb > 0 && this.inTraining)
                this.dropoutList = rand(size(this.output)) > this.dropoutProb;
                this.output = this.output .* this.dropoutList;
            elseif (this.dropoutProb > 0 && ~this.inTraining)
                this.output = this.output * (1-this.dropoutProb);
            end
            
        end
        
        function ComputeGradientAndBackpropagateErrorSignal(this)                    
            
        end
    end
    
end