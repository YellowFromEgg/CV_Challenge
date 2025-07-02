function Test()
    % Define class names and colors (match class_dict.csv)
    classNames = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"];
    classColors = [
        0 255 255;     % Urban - Cyan
        255 255 0;     % Agriculture - Yellow
        255 0 255;     % Rangeland - Magenta
        0 255 0;       % Forest - Green
        0 0 255;       % Water - Blue
        255 255 255;   % Barren - White
        0 0 0          % Unknown - Black
    ] / 255;

    % Load trained network
    baseDir = fileparts(mfilename('fullpath'));
    netPath = fullfile(baseDir, 'archive', 'trainedNet.mat');
    if ~isfile(netPath)
        error('Model file not found: %s', netPath);
    end
    loadedData = load(netPath);
    net = loadedData.net;

    % Folder with test images
    testDir = fullfile(baseDir, '..', 'Datasets', 'Columbia Glacier');
    %testDir = fullfile(baseDir, 'archive', 'valid')
    outDir = fullfile(testDir, 'Segmented');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % Get image files
    imageFiles = dir(fullfile(testDir, '*.jpg'));
    if isempty(imageFiles)
        error('No images found in folder: %s', testDir);
    end

    % Segment and save
    for k = 1:length(imageFiles)
        imagePath = fullfile(testDir, imageFiles(k).name);
        outputPath = fullfile(outDir, imageFiles(k).name);
        segmentAndSave(net, imagePath, outputPath, classColors, classNames);
    end

    % Show color legend
    showClassColorLegend(classNames, classColors);
end

function segmentAndSave(net, imagePath, outputPath, classColors, classNames)
    I = imread(imagePath);
    inputSize = net.Layers(1).InputSize(1:2);
    Iresized = imresize(I, inputSize);

    % Predict
    predictedMask = semanticseg(Iresized, net, 'ExecutionEnvironment', 'auto');

    % Overlay
    overlay = labeloverlay(Iresized, predictedMask, 'Colormap', classColors, 'Transparency', 0.4);

    % Save
    imwrite(overlay, outputPath);
    fprintf('Saved: %s\n', outputPath);
end

function showClassColorLegend(classNames, classColors)
    figure('Name', 'Class Color Legend');
    for i = 1:length(classNames)
        patch([0 1 1 0], [i-1 i-1 i i], classColors(i, :), 'EdgeColor', 'none');
        hold on;
    end
    set(gca, 'YTick', 0.5:1:length(classNames)-0.5, 'YTickLabel', classNames, 'YDir', 'reverse');
    axis([0 1 0 length(classNames)]);
    title('Segmentation Classes and Colors');
end
