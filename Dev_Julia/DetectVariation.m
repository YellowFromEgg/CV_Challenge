imgPaths = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2010.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2002.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2004.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1990.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2017.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Wiesn\3_2020.jpg"
};

for i = 1:length(imgPaths)
    I = imread(imgPaths{i});
    [cityMask, finalMask] = detect_city_by_variation(I);
end

function [cityMask, finalMask] = detect_city_by_variation2(I)
    % STEP 1: Preprocessing
    gray = rgb2gray(I);
    gray = im2double(gray);
    stdMap = stdfilt(gray, true(15));
    stdMap = mat2gray(stdMap);
    baseMask = stdMap > 0.15;

    % STEP 2: Clean and extract connected regions
    cityMask = bwareaopen(baseMask, 100);
    cityMask = imclose(cityMask, strel('disk', 5));
    cityMask = imfill(cityMask, 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    minArea = 500;
    keepIdx = find([props.Area] >= minArea);
    filteredMask = ismember(labelmatrix(CC), keepIdx);

    % STEP 3: Color-based classification
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);
    finalMask = zeros(size(filteredMask)); % 0 = background

    for i = 1:length(keepIdx)
        idx = keepIdx(i);
        pix = props(idx).PixelIdxList;

        h = H(pix); s = S(pix); v = V(pix);

        % Stats
        greenish = h > 0.2 & h < 0.45 & s > 0.25;
        reddish  = (h < 0.05 | h > 0.95) & s > 0.3;
        greenRatio = sum(greenish) / numel(pix);
        redRatio   = sum(reddish) / numel(pix);
        meanSat = mean(s);

        % --- Classification ---
        if redRatio > 0.2
            classID = 1;  % city (red rooftops)
        elseif greenRatio > 0.8 && meanSat > 0.35
            classID = 2;  % forest
        else
            classID = 1;  % fallback to city
        end

        finalMask(pix) = classID;
    end

    % STEP 4: Plot results
    cmap = [0 0 0;        % 0 - background
            1 0 0;        % 1 - city (red)
            0.1 0.7 0.1]; % 2 - forest (green)

    figure('Name', 'City Detection: Color-Based Classification', 'Position', [100 100 1600 500]);
    subplot(1,4,1); imshow(I); title('Original Image');
    subplot(1,4,2); imshow(stdMap, []); title('Local Std Map');
    subplot(1,4,3); imshow(filteredMask); title('Filtered High-Variation Regions');
    subplot(1,4,4); imshow(label2rgb(finalMask, cmap)); title('Classified: City (R), Forest (G)');
end

function [cityMask, finalMask] = detect_city_by_variation_good(I)
    % STEP 1: Preprocessing
    gray = rgb2gray(I);
    gray = im2double(gray);
    stdMap = stdfilt(gray, true(15));
    stdMap = mat2gray(stdMap);
    baseMask = stdMap > 0.15;

    % STEP 2: Clean and extract connected regions
    cityMask = bwareaopen(baseMask, 100);
    cityMask = imclose(cityMask, strel('disk', 5));
    cityMask = imfill(cityMask, 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    minArea = 500;
    keepIdx = find([props.Area] >= minArea);
    filteredMask = ismember(labelmatrix(CC), keepIdx);

    % STEP 3: Color-space classification
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    finalMask = zeros(size(filteredMask)); % 0 = background

    for i = 1:length(keepIdx)
        idx = keepIdx(i);
        pix = props(idx).PixelIdxList;

        % Extract HSV values
        h = H(pix); s = S(pix); v = V(pix);

        % --- RED ROOFTOP CHECK ---
        isRed = (h < 0.05 | h > 0.95) & s > 0.3;
        redRatio = sum(isRed);

        % Compute color stats
        meanHue = mean(H(pix));      % 0 to 1
        meanSat = mean(S(pix));
        meanVal = mean(V(pix));

        classID = 1;  % default to city
        % --- CLASSIFICATION BY COLOR ---
        if meanHue > 0.2 && meanHue < 0.45 && meanSat > 0.25
            classID = 2;  % forest (green hue)
        elseif meanSat < 0.2 && meanVal >= 0.2 && meanVal <= 0.8
            classID = 1;  % city (grayish)
        else
            classID = 1;  % city
        end

        finalMask(pix) = classID;
    end

    % STEP 4: Plot result
    cmap = [0 0 0;        % 0 - background (black)
            1 0 0;        % 1 - city (red)
            0.1 0.7 0.1]; % 2 - forest (green)

    figure('Name', 'City Detection: Color-Based Classification', 'Position', [100 100 1600 500]);
    subplot(1,4,1); imshow(I); title('Original Image');
    subplot(1,4,2); imshow(stdMap, []); title('Local Std Map');
    subplot(1,4,3); imshow(filteredMask); title('Filtered High-Variation Regions');
    subplot(1,4,4); imshow(label2rgb(finalMask, cmap)); title('Classified: City (R), Forest (G), Land (Y)');
end

function [cityMask, finalMask] = detect_city_by_variation3(I)
    % STEP 1: Preprocessing
    gray = rgb2gray(I);
    gray = im2double(gray);
    stdMap = stdfilt(gray, true(15));
    stdMap = mat2gray(stdMap);
    baseMask = stdMap > 0.15;

    % STEP 2: Clean and extract connected regions
    cityMask = bwareaopen(baseMask, 100);
    cityMask = imclose(cityMask, strel('disk', 5));
    cityMask = imfill(cityMask, 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    minArea = 500;
    keepIdx = find([props.Area] >= minArea);
    filteredMask = ismember(labelmatrix(CC), keepIdx);

    % STEP 3: Color classification
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    % Initialize final mask: low-saturation = 0 (white)
    finalMask = zeros(size(filteredMask));

    % Assign all high-variation pixels a base label 1 (black)
    finalMask(filteredMask) = 1;

    for i = 1:length(keepIdx)
        idx = keepIdx(i);
        pix = props(idx).PixelIdxList;

        h = H(pix); s = S(pix); v = V(pix);

        greenish = h > 0.2 & h < 0.45 & s > 0.25;
        reddish  = (h < 0.05 | h > 0.95) & s > 0.3;
        greenRatio = sum(greenish) / numel(pix);
        redRatio   = sum(reddish) / numel(pix);
        meanSat = mean(s);

        % Classify only high-saturation regions (s > 0.2 mean)
        if meanSat > 0.2
            if redRatio > 0.2
                classID = 2; % city (red)
            elseif greenRatio > 0.8 && meanSat > 0.35
                classID = 3; % forest (green)
            else
                classID = 2; % fallback to city
            end
            finalMask(pix) = classID;
        end
    end

    % STEP 4: Plot results
    cmap = [1 1 1;        % 0 - low saturation (white)
            0 0 0;        % 1 - high variation (black)
            1 0 0;        % 2 - city (red)
            0.1 0.7 0.1]; % 3 - forest (green)

    figure('Name', 'City Detection: Label 0=white, 1=black, 2=red, 3=green', 'Position', [100 100 1600 500]);
    subplot(1,4,1); imshow(I); title('Original Image');
    subplot(1,4,2); imshow(stdMap, []); title('Local Std Map');
    subplot(1,4,3); imshow(filteredMask); title('Filtered High-Variation Regions');
    subplot(1,4,4); imshow(label2rgb(finalMask, cmap)); title('Classified: 0-W, 1-B, 2-City(R), 3-Forest(G)');
end

function [cityMask, finalMask] = detect_city_by_variation5(I)
    % STEP 1: Preprocessing
    gray = rgb2gray(I);
    gray = im2double(gray);
    stdMap = stdfilt(gray, true(15));
    stdMap = mat2gray(stdMap);
    baseMask = stdMap > 0.15;

    % STEP 2: Clean and extract connected regions
    cityMask = bwareaopen(baseMask, 100);
    cityMask = imclose(cityMask, strel('disk', 5));
    cityMask = imfill(cityMask, 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    minArea = 500;
    keepIdx = find([props.Area] >= minArea);
    filteredMask = ismember(labelmatrix(CC), keepIdx);

    % STEP 3: Color-based classification
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);
    finalMask = zeros(size(filteredMask)); % 0 = background (low-saturation)

    for i = 1:length(keepIdx)
        idx = keepIdx(i);
        pix = props(idx).PixelIdxList;

        h = H(pix); s = S(pix); v = V(pix);

        % Stats
        greenish = h > 0.2 & h < 0.45 & s > 0.25;
        reddish  = (h < 0.05 | h > 0.95) & s > 0.3;
        greenRatio = sum(greenish) / numel(pix);
        redRatio   = sum(reddish) / numel(pix);
        meanSat = mean(s);

        % Classification
        if redRatio > 0.2
            classID = 2;  % city (red rooftops)
        elseif greenRatio > 0.8 && meanSat > 0.35
            classID = 3;  % forest
        else
            classID = 1;  % high-saturation, uncategorized (black)
        end

        finalMask(pix) = classID;
    end

    % STEP 4: Plot results
    cmap = [1 1 1;        % 0 - white (low saturation background)
            0 0 0;        % 1 - black (uncategorized high variation)
            1 0 0;        % 2 - red (city)
            0.1 0.7 0.1]; % 3 - green (forest)

    disp('Labels present in finalMask:');
    disp(unique(finalMask));

    figure('Name', 'City Detection with Label Map', 'Position', [100 100 1600 500]);
    subplot(1,4,1); imshow(I); title('Original Image');
    subplot(1,4,2); imshow(stdMap, []); title('Local Std Map');
    subplot(1,4,3); imshow(filteredMask); title('Filtered High-Variation Regions');

    subplot(1,4,4);
    imshow(label2rgb(finalMask, cmap));
    title('Classified: 0-W, 1-B, 2-City(R), 3-Forest(G)');

    % Add custom colorbar
    colormap(gca, cmap);
    cb = colorbar('Ticks', 0.375:1:3.375, ...
                  'TickLabels', {'0:White (bg)', '1:Black (uncat)', '2:City', '3:Forest'}, ...
                  'TickLength', 0);
    caxis([0 3]);
end

function [cityMask, labelMap] = detect_city_by_variation(I)
    % STEP 1: Preprocessing
    gray = rgb2gray(I);
    gray = im2double(gray);
    stdMap = stdfilt(gray, true(15));
    stdMap = mat2gray(stdMap);
    baseMask = stdMap > 0.15;

    % STEP 2: Clean and extract high-variation regions
    cityMask = bwareaopen(baseMask, 100);
    cityMask = imclose(cityMask, strel('disk', 5));
    cityMask = imfill(cityMask, 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    minArea = 500;
    keepIdx = find([props.Area] >= minArea);
    filteredMask = ismember(labelmatrix(CC), keepIdx);

    % STEP 3: HSV-based classification
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    labelMap = zeros(size(filteredMask)); % 0 = background

    for i = 1:length(keepIdx)
        idx = keepIdx(i);
        pix = props(idx).PixelIdxList;

        h = H(pix); s = S(pix); v = V(pix);

        % Classify based on HSV and internal variation
        isRedRoof = (h < 0.05 | h > 0.95) & s > 0.3;
        isGrayish = s < 0.25 & v > 0.3 & v < 0.8;
        isTanLike = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.4;
        greenish = h > 0.2 & h < 0.45 & s > 0.25;
        
        RedSum = sum(isRedRoof);
        RedRatio = sum(isRedRoof)/numel(pix);
        GrayRatio = sum(isGrayish) / numel(pix);
        TanRatio = sum(isTanLike) / numel(pix);
        GreenRatio = sum(greenish) / numel(pix);
        MeanSat = mean(s);
        MeanVal = mean(v);
        VarVal = std(double(v));
        
        % CLASSIFY
        if (RedSum > 28)
            label = 3; % City
        elseif GreenRatio > 0.42
            label = 3; %forest
        %elseif GrayRatio > 0.2 && GreenRatio < 0.1 && MeanVal < 0.75
            %label = 3; % City (gray urban)
        %elseif GreenRatio > 0.4 && GrayRatio < 0.1 && VarVal < 0.02
            %label = 1; % Forest
        elseif TanRatio > 0.1 && GreenRatio < 0.4 && MeanVal > 0.5
            label = 3; % Land
        elseif MeanSat < 0.1 && VarVal < 0.02
            label = 0;
        else
            label = 4; % Unclassified but high-variation
        end

        
        labelMap(pix) = label;
    end

    % STEP 4: Plot result
    cmap = [
    1 1 1;         % 0 - Background (white)
    0 1 0;         % 1 - Forest (green)
    0 0 0;         % 2 - Unclassified (black)
    1 0 0;         % 3 - City (red)
    0.8 0.8 0.4    % 4 - Land (yellowish-green)
    ];
    
    labels = {'Background','Forest','Unclassified','City','Land'};

    figure('Name', 'City Detection: Label Map', 'Position', [100 100 1600 500]);
    subplot(1,4,1); imshow(I); title('Original Image');
    subplot(1,4,2); imshow(stdMap, []); title('Local Std Map');
    subplot(1,4,3); imshow(filteredMask); title('Filtered High-Variation Regions');

    subplot(1,4,4);
    imagesc(labelMap); axis image off;
    title('Classified: 0-W, 1-G, 2-B, 3-R');
    colormap(gca, cmap); caxis([0 4]);
    colorbar('Ticks', 0:4, 'TickLabels', labels);
end
