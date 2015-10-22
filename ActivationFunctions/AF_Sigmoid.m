
classdef AF_Sigmoid < IActivationFunction
   
    methods
        function res = Calculate(obj, input)
            %res = ones(size(input))/2;
            res = (1+exp(-input)) .^ -1;
        end
        
        function res = CalculateDerivative(obj, input)
            res = input .* (1 - input);
        end
    end
    
end