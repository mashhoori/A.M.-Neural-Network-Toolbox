clear
addpath(genpath('.\'));
rng('default');

trainData = load('train.mat');
trainLabel = load('labeltrain.mat');
% trainData = load('trainfull.mat');
% trainLabel = load('labeltrainfull.mat');
validData = load('valid.mat');
validLabel = load('labelvalid.mat');
% testData = load('test.mat');
% testLabel = load('labeltest.mat');

trainDataset = CreateDataset(trainData.train, trainLabel.labeltrain, 10, 100, 0, 1);
validDataset = CreateDataset(validData.valid, validLabel.labelvalid, 10, 100, 1, 1);

clear trainData trainLabel validData validLabel 

net = ConvolutionalNet({'i', 'c', 's', 'c', 's', 'h', 'o'}, {[28, 28], [5 5 0 6], [2 2], [5 5 0 10], [2 2], 80, 10}, true);
net.layers(2).activationFunction = AF_RectifiedLinear;
net.layers(4).activationFunction = AF_RectifiedLinear;
net.layers(6).activationFunction = AF_RectifiedLinear;

net.TrainWithMomentum(trainDataset, validDataset, 12000, 100);


%testData = load('test.mat');
% testDataset = CreateDataset(testData.test, [], 10, 100, 1, 1);
% clear testData
% PredictNN(net, validDataset, 'result.csv');