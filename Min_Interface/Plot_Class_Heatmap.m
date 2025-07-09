function Plot_Class_Heatmap(segmentedOverlapMasks, imageFiles)
% Visualizes class presence as RGB heatmap with discrete colorbar for class legend.
% Crops to strict overlap region and uses colorbar for proper categorical legend.

    if isempty(segmentedOverlapMasks) || isempty(imageFiles)
        error('Segmented masks and image file names must not be empty.');
    end

    % --- Class definitions ---
    fullCMap = [
        0.8 0.8 0.8;      % 0 = Unclassified
        0.2 0.55 0.5;     % 1 = Water/Forest
        0.6 0.4 0.2;      % 2 = Land
        1 0 0;            % 3 = Urban/Agriculture
        1 1 1;            % 4 = Snow
        0 1 1             % 5 = River/Road
    ];
    classLabels = {'Unclassified', 'Water/Forest', 'Land', ...
                   'Urban/Agriculture', 'Snow', 'River/Road'};

    numClasses = numel(classLabels);
    numImages = numel(segmentedOverlapMasks);
    imgSize = size(segmentedOverlapMasks{1});

    % --- Strict overlap mask ---
    commonMask = segmentedOverlapMasks{1} > 0;
    for i = 2:numImages
        commonMask = commonMask & (segmentedOverlapMasks{i} > 0);
    end

    % --- Count class frequencies ---
    classCounts = zeros([imgSize, numClasses]);
    for i = 1:numImages
        seg = segmentedOverlapMasks{i};
        for cls = 0:(numClasses - 1)
            classCounts(:, :, cls+1) = classCounts(:, :, cls+1) + (seg == cls);
        end
    end

    % --- Normalize and blend to RGB ---
    totalCounts = sum(classCounts, 3);
    totalCounts(totalCounts == 0) = 1;
    rgbImage = zeros([imgSize, 3]);

    for cls = 0:(numClasses - 1)
        color = reshape(fullCMap(cls+1, :), 1, 1, 3);
        weight = classCounts(:, :, cls+1) ./ totalCounts;
        for c = 1:3
            rgbImage(:, :, c) = rgbImage(:, :, c) + weight .* color(:, :, c);
        end
    end

    % --- Mask non-overlap (black out) ---
    for c = 1:3
        rgbImage(:, :, c) = rgbImage(:, :, c) .* commonMask;
    end

    % --- Display heatmap and colorbar-based legend ---
    numImages = numel(segmentedOverlapMasks);
    cols = ceil(sqrt(numImages));
    rows = ceil(numImages / cols);

    figure('Name', 'Class Heatmap with Colorbar Legend', ...
           'NumberTitle', 'off', ...
           'Position', [100, 100, min(1800, cols*300), min(1200, rows*250)]);

    % Heatmap
    imshow(rgbImage);
    title('Class Presence Heatmap Over Time (Strict Overlap)');
    axis image;

    % Colorbar legend (categorical ticks)
    colormap(fullCMap); % Apply full categorical colormap
    colorbar('Ticks', 0:numel(classLabels)-1, ...
                     'TickLabels', classLabels, ...
                     'TickLength', 0, 'Direction', 'reverse');

end
