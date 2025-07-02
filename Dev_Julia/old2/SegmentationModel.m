%% DeepGlobe Land Cover Segmentation Pipeline

%parpool('local');  % Starts parallel workers (only needed once per session)


% Set base directory (where "archive" folder lives)
baseDir = fileparts(mfilename('fullpath'));  % folder of this .m file
dataDir = fullfile(baseDir, 'archive');

% Define class labels and RGB colors
classNames = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"];
labelColors = [
    0 255 255;    % Urban
    255 255 0;    % Agriculture
    255 0 255;    % Rangeland
    0 255 0;      % Forest
    0 0 255;      % Water
    255 255 255;  % Barren
    0 0 0         % Unknown
];

inputSize = [512 512 3];
numClasses = numel(classNames);

%% STEP 1: Preprocess training masks into indexed labels
disp('>> Preprocess training masks into indexed labels');
%preprocessMasks(fullfile(dataDir, 'train'), labelColors, fullfile(dataDir, 'train_labels'));

%% STEP 2: Create Datastores from training set
disp('>> Create Datastores from training set');
imds = imageDatastore(fullfile(dataDir, 'train'), 'FileExtensions','.jpg');
pxds = pixelLabelDatastore(fullfile(dataDir, 'train_labels'), classNames, 1:numClasses);

%% STEP 3: Split training into train/val (80/20)
disp('>> Split training into train/val (80/20)');
rng(1);  % Reproducibility
numImages = numel(imds.Files);
idx = randperm(numImages);
numTrain = round(0.8 * numImages);

imdsTrain = subset(imds, idx(1:numTrain));
imdsVal   = subset(imds, idx(numTrain+1:end));
pxdsTrain = subset(pxds, idx(1:numTrain));
pxdsVal   = subset(pxds, idx(numTrain+1:end));

%% STEP 4: Combine and Resize
disp('>> Combine and Resize');
augmenter = imageDataAugmenter('RandXReflection', true);
pximdsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'DataAugmentation', augmenter, 'OutputSize', inputSize);
pximdsVal = pixelLabelImageDatastore(imdsVal, pxdsVal, ...
    'OutputSize', inputSize);

%% STEP 5: Define Network (DeepLabv3+ with ResNet-18)
lgraph = deeplabv3plusLayers(inputSize, numClasses, 'resnet18');

%% STEP 6: Train Model
disp('>> Starting training...');
tic;

options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'gpu', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',pximdsVal, ...
    'ValidationFrequency',50, ...
    'VerboseFrequency',10, ...
    'Plots','training-progress');

disp(">> Training started...");
tic;
net = trainNetwork(pximdsTrain, lgraph, options);
trainingTime = toc;
fprintf(">> Training finished in %.2f seconds (%.2f minutes)\n", trainingTime, trainingTime/60);

%% STEP 7: Save the model
save(fullfile(dataDir, 'trainedNet.mat'), 'net');
disp('>> Model saved to archive/trainedNet.mat');

%% STEP 8: Evaluate on validation split
pxdsResults = semanticseg(imdsVal, net, 'MiniBatchSize', 4, 'OutputType', 'uint8');
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsVal);
disp(metrics.DataSetMetrics);

%% STEP 9: Predict on an unlabeled image (optional)
testImage = fullfile(dataDir, 'valid', 'C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Dev_Keno\Theresienwiese\2024_05.png');  % Example test image
if isfile(testImage)
    segmentImage(net, testImage, classNames, fullfile(dataDir, 'test_output_123.png'));
end

%% -------- Functions --------

function preprocessMasks(maskFolder, labelColors, outputFolder)
    if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end
    files = dir(fullfile(maskFolder, '*_mask.png'));
    for i = 1:numel(files)
        maskRGB = imread(fullfile(maskFolder, files(i).name));
        label = convertRGBMaskToLabel(maskRGB, labelColors);
        imwrite(label, fullfile(outputFolder, files(i).name));
    end
end

function label = convertRGBMaskToLabel(maskRGB, labelColors)
    maskRGB = uint8(maskRGB >= 128) * 255;
    label = zeros(size(maskRGB, 1), size(maskRGB, 2), 'uint8');
    for idx = 1:size(labelColors,1)
        match = all(maskRGB == reshape(labelColors(idx,:), 1, 1, 3), 3);
        label(match) = idx;
    end
end

function segmentImage(net, imagePath, classNames, outputPath)
    if ~isfile(imagePath)
        warning('Test image not found: %s', imagePath);
        return;
    end
    I = imread(imagePath);
    inputSize = net.Layers(1).InputSize(1:2);
    Iresized = imresize(I, inputSize);
    predictedMask = semanticseg(Iresized, net);
    overlay = labeloverlay(Iresized, predictedMask, 'Colormap', jet(numel(classNames)));
    figure; imshow(overlay); title('Predicted Land Cover');
    if nargin == 4 && ~isempty(outputPath)
        imwrite(overlay, outputPath);
        fprintf('Saved prediction to: %s\n', outputPath);
    end
end
