
classdef AF_RectifiedLinear < IActivationFunction
   
    methods
        function res = Calculate(obj, input)                        
            res = input;
            res(res < 0) = 0;            
        end
        
        function res = CalculateDerivative(obj, input)           
            
            res = zeros(size(input));
            res(input > 0) = 1;
            
        end
    end
    
end