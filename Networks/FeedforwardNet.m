
classdef FeedforwardNet < BasicNetwork
    
    methods
        function net = FeedforwardNet(shape, softMax)
            net.numOfInputs = shape(1);
            net.numOfOutputs = shape(end);
            net.numOfLayers = length(shape);
            
            if(softMax)
                net.costType = 'CrossEntropy';
            else
                net.costType = 'EuclideanDistance';
            end
            
            net.InitNetwork(shape);
        end
    end
    
    methods (Access = private)
        function InitNetwork(this, shape)
            
            index = 1;
            inputLayer = InputLayer(this.numOfInputs);
%              inputLayer.dropoutProb = 0.2;
            inputLayer.index = index;
            this.layers(index) = inputLayer;
            index = index + 1;
            
            for i = 2:this.numOfLayers - 1
                hidden = BasicLayer(shape(i), 'hidden');
                hidden.index = index;
                hidden.activationFunction = AF_HyperbolicTangant;
%                 hidden.dropoutProb = 0.5;
                this.layers(index) = hidden;
                hidden.previousLayer = this.layers(index - 1);
                index = index + 1;
            end
            
            outputLayer = BasicLayer(this.numOfOutputs, 'Output');
            if(strcmp(this.costType,'CrossEntropy'))
                outputLayer = SoftMaxLayer(this.numOfOutputs);
            end
            
            outputLayer.index = index;
            this.layers(index) = outputLayer;
            outputLayer.previousLayer = this.layers(index - 1);
            
            for i = 1:this.numOfLayers
                this.layers(i).InitRandomWeights();
            end
            
        end
    end
end