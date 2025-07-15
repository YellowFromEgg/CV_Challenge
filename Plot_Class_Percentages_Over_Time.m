function Plot_Class_Percentages_Over_Time(segmentedOverlapMasks, imageFiles)
% PLOT_CLASS_PERCENTAGES_OVER_TIME Plots how much of each class is present
% over time, restricted to the consistently overlapping region across all images.
%
% Inputs:
%   segmentedOverlapMasks - cell array of segmented label maps (aligned)
%   imageFiles            - corresponding filenames containing date info
%
% Output:
%   Creates a stacked area chart showing class proportions over time.

    % --- Input validation ---
    if isempty(segmentedOverlapMasks) || isempty(imageFiles)
        error('Segmented masks and image file names must not be empty.');
    end

    % --- Class labels and color map ---
    classLabels = {'Wasser/Wald', 'Land', 'Stadt/Landwirtschaft', ...
                   'Schnee', 'Fluss/Straße'};  % Class 1–5 (excluding Unclassified)

    fullCMap = [
        0.8 0.8 0.8;      % 0 = Unclassified
        0.2 0.55 0.5;     % 1 = Water/Forest
        0.6 0.4 0.2;      % 2 = Land
        1 0 0;            % 3 = Urban/Agriculture
        1 1 1;            % 4 = Snow
        0 1 1             % 5 = River/Road
    ];
    cmap = fullCMap(2:end, :);  % Only use classes 1 to 5

    validClassIndices = 1:5;  % Ignore class 0
    numClasses = numel(validClassIndices);
    numImages  = numel(segmentedOverlapMasks);

    % --- Parse and extract timestamps from filenames ---
    dates = NaT(1, numImages);  % Preallocate datetime array
    for i = 1:numImages
        tokens = regexp(imageFiles{i}, '(\d{1,2})_(\d{4})', 'tokens', 'once');
        if isempty(tokens)
            error('Invalid filename format: %s', imageFiles{i});
        end
        month = str2double(tokens{1});
        year  = str2double(tokens{2});
        dates(i) = datetime(year, month, 1);  % Set to first day of the month
    end

    % --- Sort files and masks by date ---
    [dates, sortIdx] = sort(dates);  % Chronological order
    segmentedOverlapMasks = segmentedOverlapMasks(sortIdx);

    % --- Create binary masks for each segmentation (where labels exist) ---
    validMasks = cell(1, numImages);
    for i = 1:numImages
        seg = segmentedOverlapMasks{i};
        if isempty(seg)
            validMasks{i} = false(size(segmentedOverlapMasks{1}));  % fallback empty mask
        else
            validMasks{i} = seg > 0;  % Valid (non-unclassified) region
        end
    end

    % --- Compute common overlapping region across all segmentations ---
    commonMask = validMasks{1};
    for i = 2:numImages
        commonMask = commonMask & validMasks{i};  % Logical AND across all masks
    end

    % --- Compute class percentages within common region ---
    classPercents = zeros(numImages, numClasses);  % Rows = time, Cols = class %
    for i = 1:numImages
        seg = segmentedOverlapMasks{i};
        if isempty(seg)
            continue;
        end

        roi = seg(commonMask);  % Only consider overlapping region
        if isempty(roi)
            continue;
        end

        total = numel(roi);  % Total pixels in common region
        for j = 1:numClasses
            classVal = validClassIndices(j);
            classPercents(i, j) = sum(roi == classVal) / total * 100;  % % of this class
        end
    end

    % --- Plot stacked area chart ---
    figure('Name', 'Segmentanteile über die Zeit (gemeinsamer sichtbarer Bereich)', ...
           'NumberTitle', 'off', ...
           'Position', [200, 200, 1000, 500]);

    h = area(dates, classPercents);  % Stacked area plot

    % Apply class-specific colors (in reverse order for correct stack appearance)
    for i = 1:numel(h)
        h(i).FaceColor = cmap(i, :);
    end

    % --- Plot formatting ---
    legend(classLabels, 'Location', 'eastoutside');
    title('Segmentanteile über die Zeit (gemeinsamer sichtbarer Bereich)');
    xlabel('Jahr');
    ylabel('Prozent (%)');
    ylim([0 100]);
    xticks(dates);
    xtickformat('yyyy');  % Show only years on X-axis
    grid on;
end
