
classdef ILayer < handle & matlab.mixin.Heterogeneous
   
    properties
       dropoutProb = 0;
       dropoutList = [];
       shape
       type
       neuronCount
       index
       previousLayer
       activationFunction
       weights
       output
       trainingInfo 
       inTraining
    end
    
    methods(Abstract)
        InitRandomWeights(obj);
        UpdateInternalRepresentation(obj);
        %UpdateWeights(obj);
        CalculateOutput(obj);
        ComputeGradientAndBackpropagateErrorSignal(obj);
        GetNumberOfParameters(obj);
    end
    
    methods
        function UpdateWeights(this)
            this.weights = this.weights + this.trainingInfo.deltaWeight;
            this.UpdateInternalRepresentation();
            
            this.trainingInfo.deltaWeight_previous = this.trainingInfo.deltaWeight;
            this.trainingInfo.gradient_previous = this.trainingInfo.gradient;           

            numParam = this.GetNumberOfParameters();

            this.trainingInfo.gradient = zeros(numParam, 1);
            this.trainingInfo.deltaWeight = zeros(numParam, 1);            
        end
    end
    
    
end