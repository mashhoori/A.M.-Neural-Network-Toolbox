classdef AF_HyperbolicTangant < IActivationFunction
   
    methods
        function res = Calculate(obj, input)            
            res = tanh(input);            
        end
        
        function res = CalculateDerivative(obj, input)
            res = (1 - input.^2);           
        end
    end
    
end