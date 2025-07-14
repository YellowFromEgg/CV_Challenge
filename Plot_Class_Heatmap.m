function Plot_Class_Heatmap(segmentedOverlapMasks, imageFiles, axesHandle, selectedClasses)
% Visualizes class presence as RGB heatmap with discrete colorbar for class legend.
% Crops to strict overlap region and uses colorbar for proper categorical legend.
    if isempty(selectedClasses)
    cla(axesHandle);  % Achse leeren
    title(axesHandle, 'Keine Klassen ausgewählt');
    return;
    end
    if isempty(segmentedOverlapMasks) || isempty(imageFiles)
        error('Segmented masks and image file names must not be empty.');
    end

    % --- Class definitions ---
    fullCMap = [
        0.8 0.8 0.8;      % 0 = Unbekannt
        0.2 0.55 0.5;     % 1 = Wasser/Wald
        0.6 0.4 0.2;      % 2 = Land
        1 0 0;            % 3 = Stadt/Landwirtschaft
        1 1 1;            % 4 = Schnee
        0 1 1             % 5 = Fluss/Straße
    ];
    classLabels = {'Unbekannt', 'Wasser/Wald', 'Land', ...
                   'Stadt/Landwirtschaft', 'Schnee', 'Fluss/Straße'};

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

    for cls = selectedClasses
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
    % numImages = numel(segmentedOverlapMasks);
    % cols = ceil(sqrt(numImages));
    % rows = ceil(numImages / cols);



    % Heatmap
    imshow(rgbImage, 'Parent', axesHandle);
    title(axesHandle, 'Class Presence Heatmap Over Time (Strict Overlap)');
    axis(axesHandle, 'image');
    % Manuelle Legende anzeigen
    % === Legende zeichnen (zeigt IMMER alle Klassen) ===
    hold(axesHandle, 'on');
    
    legendX = 20;              % Start X-Position
    legendY = 55;              % Start Y-Position
    boxSize = 20;              % Größe der Farbfelder
    spacing = 40;              % Abstand zwischen den Einträgen
    textOffset = 3;            % Abstand von Box zu Text
    fontSize = 10;
    
    % Alle Klassen anzeigen
    numLegendEntries = numel(classLabels);
    labelLengths = cellfun(@(s) strlength(s), classLabels);
    maxLabelLength = max(labelLengths);
    
    % Abschätzung der Textbreite in Pixeln
    charWidthPx = 7;  % Standardwert für FontSize ~10
    legendWidth = 2.0*(boxSize + textOffset + maxLabelLength * charWidthPx + 10);
    legendHeight =1.2 *(spacing * numLegendEntries);
    
    % Hintergrundbox hinter Legende zeichnen
    patch(axesHandle, ...
        [0 1 1 0]*legendWidth + legendX - 5, ...
        [0 0 1 1]*legendHeight + legendY - spacing, ...
        [1 1 1], ...
        'EdgeColor', [0.2 0.2 0.2], ...
        'FaceAlpha', 0.85);
    
    % Alle Einträge zeichnen
    for cls = 0:(numLegendEntries - 1)
        color = fullCMap(cls + 1, :);
        label = classLabels{cls + 1};
    
        % Farbfeld
        patch(axesHandle, ...
            [0 1 1 0]*boxSize + legendX, ...
            [0 0 1 1]*boxSize + legendY + cls * spacing, ...
            color, 'EdgeColor', 'none');
    
        % Text
        text(axesHandle, ...
            legendX + boxSize + textOffset, ...
            legendY + cls * spacing + boxSize/2, ...
            label, ...
            'Color', 'k', ...
            'FontSize', fontSize, ...
            'VerticalAlignment', 'middle');
    end
    
    hold(axesHandle, 'off');



    % Colorbar im App-Fenster anzeigen
    %colormap(axesHandle, fullCMap(selectedClasses + 1, :));
    % % colorbar(axesHandle, ...
    % %     'Ticks', selectedClasses, ...
    % %     'TickLabels', classLabels(selectedClasses + 1), ...
    % %     'TickLength', 0, ...
    % %     'Direction', 'reverse');;

end
