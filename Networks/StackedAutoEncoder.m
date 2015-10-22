
classdef StackedAutoEncoder < handle
    
    properties 
        encoders = FeedforwardNet.empty();
        numberOfAutoEncoders;
        shape
        codingNet;
        decodingNet;
    end
    methods
        function net = StackedAutoEncoder(shape)
            net.shape = shape;
            net.numberOfAutoEncoders = length(shape)-1;                             
            net.InitNetwork(shape);
        end
    end
    
    methods (Access = private)
        function InitNetwork(this, shape)            
            for i=1:this.numberOfAutoEncoders
                this.encoders(i) = FeedforwardNet([shape(i), shape(i+1), shape(i)], false);
            end
        end
    end
    
    methods
         function Train(this, dataset, trainSize, maxIter)  
             
             dt = dataset;
             
             for i=1:this.numberOfAutoEncoders
                 this.encoders(i).TrainWithMomentum(dt, [], trainSize, maxIter);
                 
                 for bc=1:numel(dataset)
                     this.encoders(i).CalculateOutput(dt(bc).input);
                     dt(bc).input = this.encoders(i).layers(2).output;
                     dt(bc).output = dt(bc).input;
                 end
                 
             end   
                   
         end
         
         function CreateInternalNetworks(this)
             this.codingNet = FeedforwardNet(this.shape, false);
             for i=2:numel(this.shape)
                 this.codingNet.layers(i).weights = this.encoders(i-1).layers(2).weights;
                 this.codingNet.layers(i).UpdateInternalRepresentation();
             end
             
             this.decodingNet = FeedforwardNet(this.shape(end:-1:1), false);
             for i=2:numel(this.shape)
                 this.decodingNet.layers(i).weights = this.encoders(end-i+2).layers(3).weights;
                 this.decodingNet.layers(i).UpdateInternalRepresentation();
             end
             
         end         
    end
end