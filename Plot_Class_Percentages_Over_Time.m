function Plot_Class_Percentages_Over_Time(segmentedOverlapMasks, imageFiles, axesHandle)
    if isempty(segmentedOverlapMasks) || isempty(imageFiles)
        error('Segmented masks and image file names must not be empty.');
    end

    % Class labels & Farben (ohne Unclassified)
    classLabels = {'Water/Forest', 'Land', 'Urban/Agriculture', ...
                   'Snow', 'River/Road'};
    fullCMap = [
        0.8 0.8 0.8;      % Unclassified (0)
        0.2 0.55 0.5;     % Water/Forest (1)
        0.6 0.4 0.2;      % Land (2)
        1 0 0;            % Urban/Agriculture (3)
        1 1 1;            % Snow (4)
        0 1 1             % River/Road (5)
    ];
    cmap = fullCMap(2:end, :);
    validClassIndices = 1:5;

    numClasses = numel(validClassIndices);
    numImages  = numel(segmentedOverlapMasks);

    % Datumswerte aus Dateinamen
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

    % Nach Datum sortieren
    [dates, sortIdx] = sort(dates);
    segmentedOverlapMasks = segmentedOverlapMasks(sortIdx);

    % Gemeinsame Maske berechnen
    validMasks = cellfun(@(seg) seg > 0, segmentedOverlapMasks, 'UniformOutput', false);
    commonMask = validMasks{1};
    for i = 2:numImages
        commonMask = commonMask & validMasks{i};
    end

    % Prozentanteile pro Klasse berechnen
    classPercents = zeros(numImages, numClasses);
    for i = 1:numImages
        seg = segmentedOverlapMasks{i};
        roi = seg(commonMask);
        total = numel(roi);
        for j = 1:numClasses
            classVal = validClassIndices(j);
            classPercents(i, j) = sum(roi == classVal) / total * 100;
        end
    end

    % Zeichnen in UIAxes
        cla(axesHandle);
        set(axesHandle, ...
            'Color', 'w', ...             % Hintergrundfarbe: Weiß
            'XColor', 'k', ...            % X-Achsenfarbe: Schwarz
            'YColor', 'k', ...            % Y-Achsenfarbe: Schwarz
            'LineWidth', 1.2, ...
            'FontSize', 10, ...
            'FontWeight', 'normal', ...
            'Box', 'on', ...
            'TickDir', 'out', ...
            'TickLength', [0.015 0.025]);  % Länge der Ticks
        yticks(axesHandle, 0:10:100);  % Y-Ticks alle 10 %
        hold(axesHandle, 'on');

    
        h = area(axesHandle, dates, classPercents);
    
        for i = 1:numel(h)
            h(i).FaceColor = cmap(i, :);
        end
    
        title(axesHandle, 'Class Distribution Over Time (Common Visible Area)', 'FontWeight','bold');
        xlabel(axesHandle, 'Date');
        ylabel(axesHandle, 'Percentage (%)');
        ylim(axesHandle, [0 100]);
        yticks(axesHandle, 0:10:100);                     % ✅ 10er-Ticks von 0–100
        xtickformat(axesHandle, 'MMM-yyyy');              % Datumsformat
        xtickangle(axesHandle, 45);                       % bessere Lesbarkeit
        legend(axesHandle, classLabels, 'Location', 'eastoutside');
        axis(axesHandle, 'normal');
        grid(axesHandle, 'on');
        hold(axesHandle, 'off');




   end
