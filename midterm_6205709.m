clear
clc
close all
checkpointPath='checkpointPath';
mkdir(checkpointPath);
%%  Digit dataset
folder_location='C:\Users\apoorva\Desktop\Bme433\Totalcrop';
imds = imageDatastore(folder_location,'IncludeSubfolders',true,'LabelSource','foldernames', 'FileExtensions',{'.jpg'});
tabulate(imds.Labels);
%% Each image information
img = readimage(imds,1);
size(img);
% look at each label distribution
labelCount = countEachLabel(imds);
aa=size(labelCount);
outputSize=aa(1);
%% Separate training
[imds_training,imds_validation] = splitEachLabel(imds,0.9,'randomize');
% Change the size of the image to 224x224x1
outputSize=[80 80 3];
imageAugmenter = imageDataAugmenter( ...
                 'RandRotation',[-20,20], ...
                 'RandXTranslation',[-3 3], ...
                 'RandYTranslation',[-3 3]);
augimdstrain = augmentedImageDatastore(outputSize,imds_training);%,'DataAugmentation',imageAugmenter,'ColorPreprocessing','rgb2gray');
augimdsvalidation = augmentedImageDatastore(outputSize,imds_validation);%,'DataAugmentation',imageAugmenter,'ColorPreprocessing','rgb2gray');

%A=imdsTrain.readimage(1);
inputSize=outputSize;
outputSize=2;
%% Layer
Layers= [   imageInputLayer(inputSize);... 
    % 1st CNN
            convolution2dLayer(10,8,'Padding','same');...   % convolution operator [3x3]; there are 8 filters in this layer
            batchNormalizationLayer;...    % batch normalization layer applies the z-score transformation z=(x-mean)/std to the output from the previous layer
            reluLayer;... % drop out the pixels that are less than 0 rectified linear unit (reLu)
            maxPooling2dLayer(2,'Stride',2);...% Pooling: pooling window size [2x2]
            % maxPooling2dLayer|averagePooling2dLayer|globalAveragePooling2dLayer|MaxUnpooling2dLayer
            % 2nd CNN
            convolution2dLayer(10,8,'Padding','same');...   % convolution operator [3x3]; there are 8 filters in this layer
            batchNormalizationLayer;...    % batch normalization layer applies the z-score transformation z=(x-mean)/std to the output from the previous layer
            reluLayer;... % drop out the pixels that are less than 0 rectified linear unit (reLu)
            maxPooling2dLayer(2,'Stride',2);...% Pooling: pooling window size [2x2]
            % maxPooling2dLayer|averagePooling2dLayer|globalAveragePooling2dLayer|MaxUnpooling2dLayer
            % 3 CNN
            convolution2dLayer(10,8,'Padding','same');...   % convolution operator [3x3]; there are 8 filters in this layer
            batchNormalizationLayer;...    % batch normalization layer applies the z-score transformation z=(x-mean)/std to the output from the previous layer
            reluLayer;... % drop out the pixels that are less than 0 rectified linear unit (reLu)
            maxPooling2dLayer(2,'Stride',2);...% Pooling: pooling window size [2x2]
            % maxPooling2dLayer|averagePooling2dLayer|globalAveragePooling2dLayer|MaxUnpooling2dLayer
    % Ouput
     fullyConnectedLayer(outputSize);...%(1) fullyconnected layer
            softmaxLayer;...                   %(2) SoftmaxLayer to identify the highest possibility output
            classificationLayer;];             %(3) classification layer% Analyze the network
            
options = trainingOptions(  'sgdm', ...                         % sgdm or adam does not affect the training, but the speed
                            'InitialLearnRate',0.01, ...        % set the initial learning rate
                            'MaxEpochs',100, ...                 % how many times that you want the network to learn from the whole set of training data
                            'MiniBatchSize',128,...
                            'Shuffle','every-epoch', ...        % Shuffle the images around at the begining of every iteration (Epoch)
                            'ValidationData',augimdsvalidation, ... % Specify validation data image data store
                            'ValidationFrequency',2, ...       % Validate every 30 training
                            'Verbose',true, ...                 % display code generation process true of false
                            'Plots','training-progress',...       % plot training process?
                            'CheckpointPath',checkpointPath);   % save check point so that if something go wrong you can retrain the network from the last checkpoint not from the start
 %% Train Data                      
net = trainNetwork(augimdstrain,Layers,options);
save('Trained_network.mat','net');