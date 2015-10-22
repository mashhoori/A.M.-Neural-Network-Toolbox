
classdef IActivationFunction < handle & matlab.mixin.Heterogeneous
    
    methods (Abstract)
        Calculate(obj, input)
        CalculateDerivative(obj, input)        
    end
    
    methods (Static, Sealed, Access = protected)
        function default_object = getDefaultScalarElement
            default_object = AF_Sigmoid;
        end
    end
    
end