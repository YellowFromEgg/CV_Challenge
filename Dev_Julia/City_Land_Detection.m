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
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2003.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2005.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2010.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2015.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2020.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2017.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Wiesn\3_2020.jpg"
};

for i = 1:length(imgPaths)
    I = imread(imgPaths{i});
    [finalMask] = detect_city_by_variation(I);
end


function [labelMap] = detect_city_by_variation(I)
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

%% old
function [labelMap] = detect_city_by_variation2(I)
    % STEP 1: Preprocessing
    gray = im2double(rgb2gray(I));
    stdMap = mat2gray(stdfilt(gray, true(15)));
    baseMask = stdMap > 0.15;

    % STEP 2: Clean and extract high-variation regions
    cityMask = imfill(imclose(bwareaopen(baseMask, 100), strel('disk', 5)), 'holes');

    CC = bwconncomp(cityMask);
    stats = regionprops(CC, 'PixelIdxList', 'Area');
    areaArray = [stats.Area];
    keepIdx = find(areaArray >= 500);

    labelMatrix = labelmatrix(CC);
    filteredMask = false(size(labelMatrix));
    for k = 1:length(keepIdx)
        filteredMask(CC.PixelIdxList{keepIdx(k)}) = true;
    end

    % STEP 3: HSV-based classification
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    labelMap = zeros(size(filteredMask), 'uint8');

    for i = 1:length(keepIdx)
        idx = keepIdx(i);
        pix = stats(idx).PixelIdxList;

        h = H(pix); s = S(pix); v = V(pix);

        isRedRoof = (h < 0.05 | h > 0.95) & s > 0.3;
        isGrayish = s < 0.25 & v > 0.3 & v < 0.8;
        isTanLike = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.4;
        greenish = h > 0.2 & h < 0.45 & s > 0.25;

        RedSum = sum(isRedRoof);
        RedRatio = RedSum / numel(pix);
        GrayRatio = sum(isGrayish) / numel(pix);
        TanRatio = sum(isTanLike) / numel(pix);
        GreenRatio = sum(greenish) / numel(pix);
        MeanSat = mean(s);
        MeanVal = mean(v);
        VarVal = std(v);

        % CLASSIFY
        if RedSum > 28
            label = 3; % City
        elseif GreenRatio > 0.42
            label = 3; 
        elseif TanRatio > 0.1 && GreenRatio < 0.4 && MeanVal > 0.5
            label = 3; 
        elseif MeanSat < 0.1 && VarVal < 0.02
            label = 0; % Background
        else
            label = 2; % Land
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
