function Plot_Class_Percentages_Over_Time(segmentedOverlapMasks, imageFiles)
% Plots class percentages over time in the consistent overlap region only.

    if isempty(segmentedOverlapMasks) || isempty(imageFiles)
        error('Segmented masks and image file names must not be empty.');
    end

    % Class labels (excluding Unclassified = 0)
    classLabels = {'Wasser/Wald', 'Land', 'Stadt/Landwirtschaft', ...
                   'Schnee', 'Fluss/Straße'};

    fullCMap = [
        0.8 0.8 0.8;      % Unclassified (0)
        0.2 0.55 0.5;     % Water/Forest (1)
        0.6 0.4 0.2;      % Land (2)
        1 0 0;            % Urban/Agriculture (3)
        1 1 1;            % Snow (4)
        0 1 1             % River/Road (5)
    ];
    cmap = fullCMap(2:end, :);  % exclude Unclassified

    validClassIndices = 1:5;  % Ignore class 0
    numClasses = numel(validClassIndices);
    numImages  = numel(segmentedOverlapMasks);

    % Extract dates
    dates = NaT(1, numImages);
    for i = 1:numImages
        tokens = regexp(imageFiles{i}, '(\d{1,2})_(\d{4})', 'tokens', 'once');
        if isempty(tokens)
            error('Invalid filename format: %s', imageFiles{i});
        end
        month = str2double(tokens{1});
        year  = str2double(tokens{2});
        dates(i) = datetime(year, month, 1);
    end

    % Sort by date
    [dates, sortIdx] = sort(dates);
    segmentedOverlapMasks = segmentedOverlapMasks(sortIdx);

    % Create masks for all segmentations
    validMasks = cell(1, numImages);
    for i = 1:numImages
        seg = segmentedOverlapMasks{i};
        if isempty(seg)
            validMasks{i} = false(size(segmentedOverlapMasks{1}));  % fallback
        else
            validMasks{i} = seg > 0;
        end
    end
    
    % Compute pixel-wise intersection (logical AND) across all masks
    commonMask = validMasks{1};
    for i = 2:numImages
        commonMask = commonMask & validMasks{i};
    end% Create masks for all segmentations
    validMasks = cell(1, numImages);
    for i = 1:numImages
        seg = segmentedOverlapMasks{i};
        if isempty(seg)
            validMasks{i} = false(size(segmentedOverlapMasks{1}));  % fallback
        else
            validMasks{i} = seg > 0;
        end
    end

% Compute pixel-wise intersection (logical AND) across all masks
commonMask = validMasks{1};
for i = 2:numImages
    commonMask = commonMask & validMasks{i};
end

    % Compute percentages
    classPercents = zeros(numImages, numClasses);
    for i = 1:numImages
        seg = segmentedOverlapMasks{i};
        if isempty(seg)
            continue;
        end
        roi = seg(commonMask);
        if isempty(roi)
            continue;
        end
        total = numel(roi);
        for j = 1:numClasses
            classVal = validClassIndices(j);
            classPercents(i, j) = sum(roi == classVal) / total * 100;
        end
    end

    % Plot
    figure('Name', 'Segmentanteile über die Zeit (gemeinsamer sichtbarer Bereich)', ...
           'NumberTitle', 'off', ...
           'Position', [200, 200, 1000, 500]);
    
    h = area(dates, classPercents);
    
    % Apply correct colors in REVERSE order due to stacking
    for i = 1:numel(h)
        h(i).FaceColor = cmap(i, :);
    end
    
    legend(classLabels, 'Location', 'eastoutside');
    title('Segmentanteile über die Zeit (gemeinsamer sichtbarer Bereich)');
    xlabel('Jahr');
    ylabel('Prozent (%)');
    ylim([0 100]);
    xticks(dates);
    xtickformat('yyyy');
    grid on;
end