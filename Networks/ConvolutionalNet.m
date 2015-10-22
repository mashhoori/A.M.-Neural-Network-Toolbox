
classdef ConvolutionalNet < BasicNetwork
    
    methods
        function net = ConvolutionalNet(layersType, shape, softMax)
            net.numOfInputs = prod(shape{1});
            net.numOfOutputs = shape{end};
            net.numOfLayers = numel(shape);            
            
            if(softMax)
                net.costType = 'CrossEntropy';
%                 net.costType = 'AUC';
            else
                net.costType = 'EuclideanDistance';
            end
            
            net.InitNetwork(layersType, shape);
        end
    end
    
    methods (Access = private)
                
        function InitNetwork(this, layersType, shape)
                        
            for index= 1:numel(layersType)
                
                switch layersType{index}
                    case 'i'
                        layer = InputLayer(shape{index});                        
                        
                    case 'c'
                        layer = ConvolutionalLayer(shape{index}(1:end-1), shape{index}(end));            
                        
                    case 's'
                        %layer = MeanPooling(shape{index}); 
                        layer = MaxPooling(shape{index}); 
                        
                    case 'h'
                        layer = BasicLayer(shape{index}, 'hidden');
                        
                    case 'o'
                        layer = BasicLayer(this.numOfOutputs, 'Output');
                        if(strcmp(this.costType,'CrossEntropy'))
                            layer = SoftMaxLayer(this.numOfOutputs);
                        end  
                end
                
                this.layers(index) = layer;
                layer.index = index;
                if(index > 1)
                    layer.previousLayer = this.layers(index-1);
                end                
            end            
                   
%             convLayer2 = ChannelConvolutionalLayer([21, 1], 3);
%             convLayer2.index = index;
%             convLayer2.dropoutProb = 0.5;
%             convLayer2.previousLayer = convLayer;%poolLayer
%             this.layers(index) = convLayer2;
%             index = index + 1;
%             
            for i = 2:this.numOfLayers
                this.layers(i).InitRandomWeights();
            end
            
        end
    end
end