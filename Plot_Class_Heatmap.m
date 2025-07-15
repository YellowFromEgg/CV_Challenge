function Plot_Class_Heatmap(segmentedOverlapMasks, imageFiles, axesHandle, selectedClasses)
% PLOT_CLASS_HEATMAP Visualizes selected class presence across multiple transformed images.
% The result is a categorical heatmap showing frequency-based blending of class colors.
%
% Inputs:
%   segmentedOverlapMasks - cell array of transformed segmentation label maps
%   imageFiles            - cell array of corresponding image filenames (for context)
%   axesHandle            - handle to axes for plotting
%   selectedClasses       - list of class indices (0-based) to include in the heatmap

    % If no classes selected, clear plot and exit
    if isempty(selectedClasses)
        cla(axesHandle);  % Clear axes
        title(axesHandle, 'Keine Klassen ausgewählt');  % Show message
        return;
    end

    % Safety check for required inputs
    if isempty(segmentedOverlapMasks) || isempty(imageFiles)
        error('Segmented masks and image file names must not be empty.');
    end

    % --- Class definitions (color + labels) ---
    fullCMap = [
        0.8 0.8 0.8;      % 0 = Unbekannt (Unknown)
        0.2 0.55 0.5;     % 1 = Wasser/Wald (Water/Forest)
        0.6 0.4 0.2;      % 2 = Land
        1 0 0;            % 3 = Stadt/Landwirtschaft (Urban/Agriculture)
        1 1 1;            % 4 = Schnee (Snow)
        0 1 1             % 5 = Fluss/Straße (River/Road)
    ];
    classLabels = {'Unbekannt', 'Wasser/Wald', 'Land', ...
                   'Stadt/Landwirtschaft', 'Schnee', 'Fluss/Straße'};

    numClasses = numel(classLabels);
    numImages = numel(segmentedOverlapMasks);
    imgSize = size(segmentedOverlapMasks{1});  % Assume all are same size

    % --- Create strict overlap mask (where all images overlap) ---
    commonMask = segmentedOverlapMasks{1} > 0;
    for i = 2:numImages
        commonMask = commonMask & (segmentedOverlapMasks{i} > 0);
    end

    % --- Count how often each class appears at each pixel ---
    classCounts = zeros([imgSize, numClasses]);  % 3D matrix: H × W × Classes
    for i = 1:numImages
        seg = segmentedOverlapMasks{i};
        for cls = 0:(numClasses - 1)
            classCounts(:, :, cls+1) = classCounts(:, :, cls+1) + (seg == cls);
        end
    end

    % --- Convert class counts to RGB blend image ---
    totalCounts = sum(classCounts, 3);  % Total labels per pixel
    totalCounts(totalCounts == 0) = 1;  % Prevent division by zero

    rgbImage = zeros([imgSize, 3]);  % Initialize RGB image

    % Blend colors only for selected classes
    for cls = selectedClasses
        color = reshape(fullCMap(cls+1, :), 1, 1, 3);  % Get RGB color
        weight = classCounts(:, :, cls+1) ./ totalCounts;  % Class frequency ratio
        for c = 1:3
            rgbImage(:, :, c) = rgbImage(:, :, c) + weight .* color(:, :, c);
        end
    end

    % --- Apply overlap mask to avoid visual artifacts outside common area ---
    for c = 1:3
        rgbImage(:, :, c) = rgbImage(:, :, c) .* commonMask;
    end

    % --- Display heatmap ---
    imshow(rgbImage, 'Parent', axesHandle);
    title(axesHandle, 'Segmentart über den kompletten Zeitraum');  % Title in German
    axis(axesHandle, 'image');

    % === Draw Custom Legend (manually rendered, not using MATLAB's legend) ===
    hold(axesHandle, 'on');

    % Layout parameters for manual legend
    legendX = 20;              % Legend top-left X
    legendY = 55;              % Legend top-left Y
    boxSize = 20;              % Size of color box
    spacing = 40;              % Space between legend entries
    textOffset = 3;            % Distance from color box to text
    fontSize = 10;             % Text font size

    % Estimate size for background rectangle
    numLegendEntries = numel(classLabels);
    labelLengths = cellfun(@(s) strlength(s), classLabels);
    maxLabelLength = max(labelLengths);
    charWidthPx = 7;  % Approximate width per character in pixels
    legendWidth = 2.0 * (boxSize + textOffset + maxLabelLength * charWidthPx + 10);
    legendHeight = 1.2 * (spacing * numLegendEntries);

    % Draw semi-transparent background box behind legend
    patch(axesHandle, ...
        [0 1 1 0] * legendWidth + legendX - 5, ...
        [0 0 1 1] * legendHeight + legendY - spacing, ...
        [1 1 1], ...
        'EdgeColor', [0.2 0.2 0.2], ...
        'FaceAlpha', 0.85);

    % Draw each legend entry (box + text)
    for cls = 0:(numLegendEntries - 1)
        color = fullCMap(cls + 1, :);
        label = classLabels{cls + 1};

        % Draw color box
        patch(axesHandle, ...
            [0 1 1 0] * boxSize + legendX, ...
            [0 0 1 1] * boxSize + legendY + cls * spacing, ...
            color, 'EdgeColor', 'none');

        % Draw label text
        text(axesHandle, ...
            legendX + boxSize + textOffset, ...
            legendY + cls * spacing + boxSize / 2, ...
            label, ...
            'Color', 'k', ...
            'FontSize', fontSize, ...
            'VerticalAlignment', 'middle');
    end

    hold(axesHandle, 'off');

    % Note: Optional colorbar could be used instead of manual legend
    % colormap(axesHandle, fullCMap(selectedClasses + 1, :));
    % colorbar(...) — intentionally omitted to preserve custom legend layout
end
