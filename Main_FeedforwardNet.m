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

trainDataset = CreateDataset(trainData.train, trainLabel.labeltrain, 10, 100, 0, 0);
validDataset = CreateDataset(validData.valid, validLabel.labelvalid, 10, 100, 1, 0);

clear trainData trainLabel validData validLabel 

net = FeedforwardNet([784 200 10], true);
net.layers(2).activationFunction = AF_RectifiedLinear;
net.TrainWithMomentum(trainDataset, validDataset, 12000, 100);

%testData = load('test.mat');
% testDataset = CreateDataset(testData.test, [], 10, 100, 1, 0);
% clear testData
% PredictNN(net, validDataset, 'result.csv');