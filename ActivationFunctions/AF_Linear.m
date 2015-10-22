
classdef AF_Linear < IActivationFunction
   
    methods
        function res = Calculate(obj, input)
            res = input;            
        end
        
        function res = CalculateDerivative(obj, input)
            res = ones(size(input));
        end
    end
    
end