
function info = GetTrainingInfo(numParam)
    numParameters = numParam;
    gradient = zeros(numParameters, 1);
    gradient_previous = zeros(numParameters, 1);        

    deltaWeight = zeros(numParameters, 1);        
    deltaWeight_previous = zeros(numParameters, 1);        

    conjugateDirection= [];
    conjugateDirection_previous= [];

    gain = [];
    error =[];
    
    info = struct('numParameters', numParameters, 'gradient', gradient, 'gradient_previous', gradient_previous, 'deltaWeight', deltaWeight, 'deltaWeight_previous', deltaWeight_previous, 'error', error);
        
end
