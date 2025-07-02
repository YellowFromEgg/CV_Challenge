profile on
imgPaths1 = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2002.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2004.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg"
};
imgPaths1 = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1990.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1995.jpg"
};
imgPaths1 = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2015.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2017.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\5_2017.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\6_2018.jpg"
};
imgPaths = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1990.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2005.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2010.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2015.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg"
};
imgPaths = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2015_07.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2015_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2016_07.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2017_04.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2018_04.jpg"

    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2019_03.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2019_06.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2020_03.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2021_06.jpg"

    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Wiesn\3_2020.jpg"
};

for i = 1:length(imgPaths)
    segment_environment(imgPaths{i});
end
profile viewer

function segment_environment_old(imgPath)
    % Read input image
    I = imread(imgPath);

    % Get individual segmentation maps
    labelSnowWater = detection_Snow_Water(imgPath);     % 0,1,4
    labelCityLand  = detect_city_land(I);               % 0,2,3
    labelRivers    = detectRivers(imgPath);             % 0,5
    %%
    figure('Name', 'CityLand', 'Position', [100 100 1400 600]);
    subplot(1,2,1); imshow(I); title('Original Image');
    subplot(1,2,2); imagesc(finalLabelMap); axis image off;
    colormap(gca, cmap); caxis([0 5]);
    colorbar('Ticks', 0:5, 'TickLabels', classNames);
    title('Segmented Map');
    %%
    % Initialize final label map with water (1) and unclassified (0)
    finalLabelMap = labelSnowWater;

    % --- Overwrite with Land (2) ---
    finalLabelMap(labelCityLand == 2) = 2;

    % --- Overwrite with City (3) ---
    finalLabelMap(labelCityLand == 3) = 3;

    % --- Overwrite with Snow (4) ---
    finalLabelMap(labelSnowWater == 4) = 4;

    % --- Overwrite with River (5) ---
    finalLabelMap(labelRivers == 5) = 5;

    % --- Visualization ---
    cmap = [
        0.8 0.8 0.8;  % 0 - Unclassified (gray)
        0 0 1;        % 1 - Water (blue)
        0.6 0.4 0.2;  % 2 - Land (brown)
        1 0 0;        % 3 - City (red)
        1 1 1;        % 4 - Snow (white)
        0 1 1         % 5 - River (cyan)
    ];

    classNames = {'Unclassified', 'Water', 'Land', 'City', 'Snow', 'River'};

    figure('Name', 'Final Segmentation Result', 'Position', [100 100 1400 600]);
    subplot(1,2,1); imshow(I); title('Original Image');
    subplot(1,2,2); imagesc(finalLabelMap); axis image off;
    colormap(gca, cmap); caxis([0 5]);
    colorbar('Ticks', 0:5, 'TickLabels', classNames);
    title('Segmented Map');
end

function segment_environment(imgPath)
    % Read input image
    I = imread(imgPath);

    %% Run detection modules
    %detects Snow, Water and Shadow based on smoothness; shadows (class 8) will be reclassified later
    labelSnowWater = detection_Snow_Water(imgPath);     % 0,1,4
    %detects city and land based on edge densities
    labelCityLand  = detect_city_land(I);               % 0,2,3,7 (7 = forest inside city)
    %detects rivers/roads based on long thin edges
    labelRivers    = detectRivers(imgPath);             % 0,5

    %%
    cmap = [
        0.8 0.8 0.8;  % 0 - Unclassified (gray) [shouldn't appear]
        0 0 1;        % 1 - Water (blue)
        0.6 0.4 0.2;  % 2 - Land (brown)
        1 0 0;        % 3 - City (red)
        1 1 1;        % 4 - Snow (white)
        0 1 1;        % 5 - River (cyan)
        1 0.9 0.5;    % 6 - Sand (light yellow)
        0 0.6 0       % 7 - Forest (green)
        0 1 0         % 8 - Shadow
    ];
    classNames = {'Unclassified','Water','Land','Urban/Agriculture','Snow','River/Road','Sand','Forest','Shadow'};
    figure('Name', 'CityLand', 'Position', [100 100 1400 600]);
    subplot(1,2,1); imshow(I); title('Original Image');
    subplot(1,2,2); imagesc(labelSnowWater); axis image off;
    colormap(gca, cmap); caxis([0 8]);
    colorbar('Ticks', 0:8, 'TickLabels', classNames);
    title('Segmented Map');
    %%

    % --- Step 2: Combine layers in correct overwrite order
    finalLabelMap = labelSnowWater;             % start with water/snow (1,4)

    finalLabelMap(labelCityLand == 2) = 2;      % Land
    finalLabelMap(labelCityLand == 3) = 3;      % City
    finalLabelMap(labelCityLand == 7) = 7;      % Forest inside City
    finalLabelMap(labelSnowWater == 4) = 4; % Snow
    finalLabelMap(labelRivers == 5) = 5;    % River

    % Fill unclassified using color-based classification
    finalLabelMap = classify_unclassified_by_color(finalLabelMap, I);

    % Reassign shadows to neighboring dominant class
    finalLabelMap = reassign_shadow_regions(finalLabelMap);
    % Final colormap
    cmap = [
        0.8 0.8 0.8;  % 0 - Unclassified (gray) [shouldn't appear]
        0 0 1;        % 1 - Water (blue)
        0.6 0.4 0.2;  % 2 - Land (brown)
        1 0 0;        % 3 - City (red)
        1 1 1;        % 4 - Snow (white)
        0 1 1;        % 5 - River (cyan)
        1 0.9 0.5;    % 6 - Sand (light yellow)
        0 0.6 0      % 7 - Forest (green)
    ];
    classNames = {'Unclassified','Water','Land','Urban/Agriculture','Snow','River/Road','Sand','Forest','Shadow'};

    % Visualization
    figure('Name', 'Final Segmentation Result', 'Position', [100 100 1400 600]);
    subplot(1,2,1); imshow(I); title('Original Image');
    subplot(1,2,2); imagesc(finalLabelMap); axis image off;
    colormap(gca, cmap); caxis([0 7]);
    colorbar('Ticks', 0:7, 'TickLabels', classNames);
    title('Segmented Map');
end

%% Detection Functions

%% detect_city_land -> returns label map containing 0,2,3
function [labelMap] = detect_city_land2(I)
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
        isTanLike = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.4;
        greenish = h > 0.2 & h < 0.45 & s > 0.25;
        
        RedSum = sum(isRedRoof);
        TanRatio = sum(isTanLike) / numel(pix);
        GreenRatio = sum(greenish) / numel(pix);
        MeanSat = mean(s);
        MeanVal = mean(v);
        VarVal = std(double(v));
        
        % CLASSIFY
        if (RedSum > 28)
            label = 3; % City
        elseif GreenRatio > 0.42
            label = 3; 
        elseif TanRatio > 0.1 && GreenRatio < 0.4 && MeanVal > 0.5
            label = 3; 
        elseif MeanSat < 0.1 && VarVal < 0.02
            label = 0;
        else
            label = 2; 
        end

        
        labelMap(pix) = label;
    end
end
function [labelMap] = detect_city_land3(I)
    % --- Step 1: Preprocessing ---
    gray = im2double(rgb2gray(I));
    stdMap = mat2gray(stdfilt(gray, true(15)));
    baseMask = stdMap > 0.15;

    % --- Step 2: Morphological cleaning and filtering ---
    cityMask = imfill(imclose(bwareaopen(baseMask, 100), strel('disk', 5)), 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    keepIdx = find([props.Area] >= 500);

    % --- HSV & RGB preparation ---
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);
    I_norm = im2double(I);
    R = I_norm(:,:,1); G = I_norm(:,:,2); B = I_norm(:,:,3);

    labelMap = zeros(size(cityMask), 'uint8');  % 0 = background

    % --- Step 3: Classify high-variation regions ---
    for i = 1:length(keepIdx)
        idx = keepIdx(i);
        pix = props(idx).PixelIdxList;

        h = H(pix); s = S(pix); v = V(pix);

        isRedRoof = (h < 0.05 | h > 0.95) & s > 0.3;
        isTanLike = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.4;
        greenish  = h > 0.2 & h < 0.45 & s > 0.25;

        RedSum     = sum(isRedRoof);
        TanRatio   = sum(isTanLike) / numel(pix);
        GreenRatio = sum(greenish) / numel(pix);
        MeanSat    = mean(s);
        MeanVal    = mean(v);
        VarVal     = std(v);

        % Default label
        label = 2; % Land

        if RedSum > 28 || GreenRatio > 0.42 || ...
           (TanRatio > 0.1 && GreenRatio < 0.4 && MeanVal > 0.5)
            label = 3; % City
        elseif MeanSat < 0.1 && VarVal < 0.02
            label = 0; % Background
        end

        % Apply initial label
        labelMap(pix) = label;

        % --- Forest detection inside urban regions ---
        if label == 3
            % Apply forest color check only within this region
            regionR = R(pix);
            regionG = G(pix);
            regionB = B(pix);
            regionH = H(pix);
            regionS = S(pix);
            regionV = V(pix);

            isGreenHSV = (regionH > 0.2) & (regionH < 0.45) & ...
                         regionS > 0.2 & regionV > 0.2;
            isGreenRGB = regionG > regionR + 0.05 & ...
                         regionG > regionB + 0.05 & ...
                         regionG > 0.3;

            forestPixels = isGreenHSV & isGreenRGB;

            if any(forestPixels)
                forestIdx = pix(forestPixels);
                labelMap(forestIdx) = 7; % Forest inside city
            end
        end
    end
end
% new -> 90% city detection
function [labelMap] = detect_city_land(I)
    % --- Step 1: Preprocessing ---
    gray = im2double(rgb2gray(I));
    stdMap = mat2gray(stdfilt(gray, true(15)));
    baseMask = stdMap > 0.15;

    % --- Step 2: Morphological cleaning and filtering ---
    cityMask = imfill(imclose(bwareaopen(baseMask, 100), strel('disk', 5)), 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    keepIdx = find([props.Area] >= 500);

    % --- HSV & RGB preparation (once) ---
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);
    I_norm = im2double(I);
    R = I_norm(:,:,1); G = I_norm(:,:,2); B = I_norm(:,:,3);

    labelMap = zeros(size(cityMask), 'uint8');  % 0 = background

    % --- Step 3: Classify regions ---
    for i = 1:length(keepIdx)
        pix = props(keepIdx(i)).PixelIdxList;

        h = H(pix); s = S(pix); v = V(pix);

        % Precompute only once
        numPix = numel(pix);

        isRedRoof = (h < 0.05 | h > 0.95) & s > 0.3;
        isTanLike = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.4;
        greenish  = h > 0.2 & h < 0.45 & s > 0.25;

        RedSum     = sum(isRedRoof);
        TanRatio   = sum(isTanLike) / numPix;
        GreenRatio = sum(greenish) / numPix;
        MeanSat    = mean(s);
        MeanVal    = mean(v);
        VarVal     = std(v);

        label = 2; % Land
        if RedSum > 28 || GreenRatio > 0.42 || ...
           (TanRatio > 0.1 && GreenRatio < 0.4 && MeanVal > 0.5)
            label = 3; % City
        elseif MeanSat < 0.1 && VarVal < 0.02
            label = 0; % Background
        end

        labelMap(pix) = label;

        % --- Global override if >90% is city ---
        cityRatio = sum(labelMap(:) == 3) / numel(labelMap);
        if cityRatio > 0.9
            labelMap(labelMap > 0) = 3;  % All non-background becomes city
        end

        % Forest detection only if city
        if label == 3
            regionR = R(pix); regionG = G(pix); regionB = B(pix);
            regionH = h; regionS = s; regionV = v;

            isGreenHSV = (regionH > 0.2) & (regionH < 0.45) & ...
                         regionS > 0.2 & regionV > 0.2;
            isGreenRGB = regionG > regionR + 0.05 & ...
                         regionG > regionB + 0.05 & ...
                         regionG > 0.3;

            forestPixels = isGreenHSV & isGreenRGB;
            if any(forestPixels)
                labelMap(pix(forestPixels)) = 7;
            end
        end
    end
end

%% detectRivers -> returns label map containing 0,5
function finalRiverMask = detectRivers(imgPath)
    rgbImg = imread(imgPath);
    grayImg = rgb2gray(rgbImg);
    mask = true(size(grayImg));

    % Normalize channels
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;

    % --- Step 1: Color/Shape-based River Candidates ---
    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.7) & (G > 0.7) & (B > 0.7);
    riverCandidates = (greenish | whiteish) & mask;
    riverCandidates = bwareaopen(riverCandidates, 100); % remove small shapes

    riverMaskColorShape = riverCandidates;
    riverMaskColorShape = imclose(riverMaskColorShape, strel('disk', 5));

    % --- Filter for Long and Thin Shapes Only ---
    CC = bwconncomp(riverMaskColorShape);
    stats = regionprops(CC, 'PixelIdxList', 'MajorAxisLength', 'MinorAxisLength', 'Area', 'Solidity');
    
    filteredMask = false(size(riverMaskColorShape));
    
    for i = 1:CC.NumObjects
        major = stats(i).MajorAxisLength;
        minor = stats(i).MinorAxisLength;
        area = stats(i).Area;
        solidity = stats(i).Solidity;
    
        if minor == 0
            continue;
        end
    
        aspectRatio = major / minor;
    
        % Filtering criteria:
        % - Must be elongated (aspect ratio > 3)
        % - Must not be too massive (area < threshold)
        % - Must not be too compact (solidity < 0.95)
        if aspectRatio > 3 && area < 30000 && solidity < 0.95
            filteredMask(CC.PixelIdxList{i}) = true;
        end
    end
    
    riverMaskColorShape = filteredMask;

    % --- Step 2: Sobel Edge Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 100);
    BW_Sobel = imclose(BW_Sobel, strel('disk', 5));

    CC_shape = bwconncomp(BW_Sobel);
    stats_shape = regionprops(CC_shape, 'PixelIdxList', 'MajorAxisLength', 'MinorAxisLength');
    
    filteredMask = false(size(riverMaskColorShape));
    
    for i = 1:CC_shape.NumObjects
        major = stats_shape(i).MajorAxisLength;
        minor = stats_shape(i).MinorAxisLength;
        
        if minor == 0
            continue;
        end
    
        aspectRatio = major / minor;
        
        % Keep shapes with high aspect ratio (e.g., > 3)
        if aspectRatio > 2
            filteredMask(CC_shape.PixelIdxList{i}) = true;
        end
    end
    
    BW_Sobel = filteredMask;

    % --- Step 3: Combine Candidates ---
    combinedCandidates = riverMaskColorShape | BW_Sobel;

    % Instead of generic imclose:
    % connectedEdges = imclose(combinedCandidates, strel('disk', 5));
    
    % Use selective bridging
    maxBridgeDistance = 10;  % how far apart to try connecting
    bridgeRadius = 2;        % thickness of bridges
    connectedEdges = connectRegionsByEdgeProximity(combinedCandidates, maxBridgeDistance, bridgeRadius);

    % --- Step 5: Connected Components and Filtering ---
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');

    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = zeros(size(grayImg));

    for i = 1:CC.NumObjects
        if stats(i).Area < 100
            continue;
        end

        pixIdx = stats(i).PixelIdxList;
        [yCoords, xCoords] = ind2sub(size(mask), pixIdx);
        coords = [xCoords, yCoords];

        if numel(pixIdx) < 2
            continue;
        end

        % Efficient max-distance estimation via convex hull
        if numel(pixIdx) > 300
            try
                K = convhull(coords(:,1), coords(:,2));
                boundaryCoords = coords(K, :);
                maxDist = max(pdist(boundaryCoords));
            catch
                maxDist = max(pdist(coords));
            end
        else
            maxDist = max(pdist(coords));
        end

        if maxDist >= minDist
            finalRiverMask(pixIdx) = 5;
        end
    end
end

%% detection_Snow_Water -> returns label map containing 0,1,4 
function labelMap = detection_Snow_Water1(imgPath)
    % Read image
    img = imread(imgPath);
    grayImg = rgb2gray(img);

    % Normalize RGB once
    imgNorm = im2double(img);
    Rn = imgNorm(:,:,1);
    Gn = imgNorm(:,:,2);
    Bn = imgNorm(:,:,3);

    % --- Step 1: Snow Detection (Region-based) ---
    edgeDensity = conv2(double(edge(grayImg, 'Canny')), ones(15) / 225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList', 'Area', 'Eccentricity');

    % Remove regions not smooth/large enough
    for k = 1:numel(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(stats(k).PixelIdxList) = false;
        end
    end

    % Recompute connected components once
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');

    labelMap = zeros(size(grayImg), 'uint8');

    snowFound = false;
    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;

        rMean = mean(Rn(idx));
        gMean = mean(Gn(idx));
        bMean = mean(Bn(idx));

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            labelMap(idx) = 4; % Snow
            snowFound = true;
        elseif bMean > rMean && bMean > gMean
            labelMap(idx) = 1; % Water
        end
    end

    % --- Step 2: Pixel-wise Water Detection ---
    blueDominant = (Bn > 0.35) & (Bn > Rn + 0.08) & (Bn > Gn + 0.05);
    cyanLike = (Bn > 0.4) & (Gn > 0.4) & (Rn < 0.4) & (abs(Bn - Gn) < 0.15);
    pixelWaterCandidates = blueDominant | cyanLike;

    % Area filtering
    L = bwlabel(pixelWaterCandidates);
    statsWater = regionprops(L, 'Area');
    smallAreas = find([statsWater.Area] < 3000);

    for i = 1:numel(smallAreas)
        pixelWaterCandidates(L == smallAreas(i)) = false;
    end

    labelMap(pixelWaterCandidates) = 1;

    % --- Optional Snow Fill ---
    if snowFound
        whiteish = (Rn > 0.7) & (Gn > 0.7) & (Bn > 0.7);
        labelMap(whiteish) = 4;
    end
end
% new -> with shadow detection
function labelMap = detection_Snow_Water(imgPath)
    % Read image
    img = imread(imgPath);
    grayImg = rgb2gray(img);

    % Normalize RGB once
    imgNorm = im2double(img);
    Rn = imgNorm(:,:,1);
    Gn = imgNorm(:,:,2);
    Bn = imgNorm(:,:,3);

    % Compute HSV
    hsvImg = rgb2hsv(imgNorm);
    V = hsvImg(:,:,3);  % Brightness channel

    % --- Step 1: Snow Detection (Region-based) ---
    edgeDensity = conv2(double(edge(grayImg, 'Canny')), ones(15) / 225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList', 'Area', 'Eccentricity');

    % Remove regions not smooth/large enough
    for k = 1:numel(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(stats(k).PixelIdxList) = false;
        end
    end

    % Recompute connected components
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');

    labelMap = zeros(size(grayImg), 'uint8');

    snowFound = false;
    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;

        rMean = mean(Rn(idx));
        gMean = mean(Gn(idx));
        bMean = mean(Bn(idx));
        vMean = mean(V(idx));
        vStd = std(V(idx));

        % --- Shadow detection (low brightness) ---
        if vMean < 0.15 && vStd < 0.05
            labelMap(idx) = 8; % Shadow
            continue;
        end

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            labelMap(idx) = 4; % Snow
            snowFound = true;
        elseif bMean > rMean && bMean > gMean
            labelMap(idx) = 1; % Water
        end
    end

    % --- Step 2: Pixel-wise Water Detection ---
    blueDominant = (Bn > 0.35) & (Bn > Rn + 0.08) & (Bn > Gn + 0.05);
    cyanLike = (Bn > 0.4) & (Gn > 0.4) & (Rn < 0.4) & (abs(Bn - Gn) < 0.15);
    pixelWaterCandidates = blueDominant | cyanLike;

    % Area filtering
    L = bwlabel(pixelWaterCandidates);
    statsWater = regionprops(L, 'Area');
    smallAreas = find([statsWater.Area] < 3000);

    for i = 1:numel(smallAreas)
        pixelWaterCandidates(L == smallAreas(i)) = false;
    end

    labelMap(pixelWaterCandidates) = 1;

    % --- Optional Snow Fill ---
    if snowFound
        whiteish = (Rn > 0.7) & (Gn > 0.7) & (Bn > 0.7);
        labelMap(whiteish) = 4;
    end
end

%% detection unclassified
function labelMap = classify_unclassified_by_color_old(labelMap, rgbImage)
    % Define target color means in RGB (normalized)
    COLOR_THRESHOLDS = struct( ...
        'Water',  [0.2, 0.45, 0.5], ... % Slightly brighter cyan/blue tone [0.18, 0.35, 0.5], ...  % Deeper blue-green %'Water',  [0.2, 0.4, 0.6], ...
        'City',   [0.4, 0.4, 0.4], ...
        'Sand',   [0.85, 0.75, 0.55], ...
        'Forest', [0.18, 0.35, 0.2] ... % Dark forest green (less overlap with teal) [0.13, 0.55, 0.13] ...  % Less blue; more distinct from water 'Forest', [0.25, 0.45, 0.25], [0.2, 0.5, 0.2] ...
    );

    % Connected regions in unclassified areas
    unclassifiedMask = (labelMap == 0);
    CC = bwconncomp(unclassifiedMask);
    stats = regionprops(CC, 'PixelIdxList');

    rgbDouble = im2double(rgbImage);
    R = rgbDouble(:,:,1);
    G = rgbDouble(:,:,2);
    B = rgbDouble(:,:,3);

    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;

        rMean = mean(R(idx));
        gMean = mean(G(idx));
        bMean = mean(B(idx));

        colorVec = [rMean, gMean, bMean];

        % Compute distances to each color prototype
        dist = @(target) norm(colorVec - target);

        dWater  = dist(COLOR_THRESHOLDS.Water);
        dCity   = dist(COLOR_THRESHOLDS.City);
        dSand   = dist(COLOR_THRESHOLDS.Sand);
        dForest = dist(COLOR_THRESHOLDS.Forest);

        [~, minIdx] = min([dWater, dCity, dSand, dForest]);
        labelList = [1, 3, 6, 7];
        newLabel = labelList(minIdx);

        labelMap(idx) = newLabel;
    end
end


function labelMap = classify_unclassified_by_color(labelMap, rgbImage)
    % Define class color prototypes (normalized RGB)
    COLOR_THRESHOLDS = struct( ...
        'Water',  [0.33, 0.38, 0.42], ...   % Blue-green water
        'City',   [0.4, 0.4, 0.4], ...
        'Sand',   [0.85, 0.75, 0.55], ...
        'Forest', [0.25, 0.40, 0.26] ...   % Dark green forest
    );

    % Connected components of unclassified regions
    unclassifiedMask = (labelMap == 0);
    CC = bwconncomp(unclassifiedMask);
    stats = regionprops(CC, 'PixelIdxList');

    % Prepare color channels
    rgbDouble = im2double(rgbImage);
    R = rgbDouble(:,:,1);
    G = rgbDouble(:,:,2);
    B = rgbDouble(:,:,3);

    Ihsv = rgb2hsv(rgbDouble);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    % Iterate through each unclassified region
    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;

        % Mean RGB and HSV
        rMean = mean(R(idx));
        gMean = mean(G(idx));
        bMean = mean(B(idx));
        hMean = mean(H(idx));
        sMean = mean(S(idx));
        vMean = mean(V(idx));

        colorVec = [rMean, gMean, bMean];
        hsvMean = [hMean, sMean, vMean];

        % --- Shortcut rule: if hue is in cyan range, classify as water ---
        if hsvMean(1) > 0.42
            labelMap(idx) = 1; % Water
            continue;
        end

        % --- Compute distances to prototypes ---
        dists = [
            norm(colorVec - COLOR_THRESHOLDS.Water);
            norm(colorVec - COLOR_THRESHOLDS.City);
            norm(colorVec - COLOR_THRESHOLDS.Sand);
            norm(colorVec - COLOR_THRESHOLDS.Forest)
        ];

        [~, minIdx] = min(dists);
        labelList = [1, 3, 6, 7];  % Class label codes
        newLabel = labelList(minIdx);

        labelMap(idx) = newLabel;
    end
end

%% reassign shadows

function labelMapOut = reassign_shadow_regions(labelMapIn)
    shadowLabel = 8;
    windowSize = 5;
    pad = floor(windowSize / 2);
    
    % Create kernel for neighborhood summation
    kernel = ones(windowSize, windowSize);
    
    % Preallocate vote maps for labels 0 to 7
    labelVotes = zeros([size(labelMapIn), 8]);  % we don't include shadow (8) as a valid target

    for lbl = 0:7
        mask = (labelMapIn == lbl);
        labelVotes(:,:,lbl+1) = imfilter(double(mask), kernel, 'same', 'replicate');
    end

    % Find shadow pixel positions
    shadowMask = (labelMapIn == shadowLabel);
    [r, c] = find(shadowMask);

    labelMapOut = labelMapIn;

    for i = 1:length(r)
        y = r(i); x = c(i);
        votes = squeeze(labelVotes(y, x, :));
        [~, maxLbl] = max(votes);
        labelMapOut(y, x) = maxLbl - 1;  % convert back to label
    end
end


%% Helper Functions
function finalMask = connectRegionsByEdgeProximity(binaryMask, maxDist, bridgeRadius)
    % CONNECTREGIONSBYEDGEPROXIMITY connects regions based on closest pixel distance
    labeled = bwlabel(binaryMask);
    numRegions = max(labeled(:));
    finalMask = binaryMask;

    % Get list of all region pixel coordinates
    regionPixels = cell(numRegions, 1);
    for i = 1:numRegions
        [y, x] = find(labeled == i);
        regionPixels{i} = [x, y];  % [x, y] format
    end

    % Loop through all unique pairs
    for i = 1:numRegions
        for j = i+1:numRegions
            pts1 = regionPixels{i};
            pts2 = regionPixels{j};

            % Compute all pairwise distances
            D = pdist2(pts1, pts2);
            [minD, idx] = min(D(:));

            if minD <= maxDist
                [p1, p2] = ind2sub(size(D), idx);
                pt1 = pts1(p1, :);
                pt2 = pts2(p2, :);

                % Draw line between pt1 and pt2
                bridge = false(size(binaryMask));
                lineIdx = drawLineBetweenPoints(pt1, pt2, size(binaryMask));
                bridge(lineIdx) = true;

                % Thicken bridge
                bridge = imdilate(bridge, strel('disk', bridgeRadius));
                finalMask = finalMask | bridge;
            end
        end
    end
end



function idx = drawLineBetweenPoints(pt1, pt2, imageSize)
    % Bresenham-style line drawing
    x1 = round(pt1(1)); y1 = round(pt1(2));
    x2 = round(pt2(1)); y2 = round(pt2(2));
    n = max(abs([x2 - x1, y2 - y1])) + 1;
    x = round(linspace(x1, x2, n));
    y = round(linspace(y1, y2, n));
    x = max(min(x, imageSize(2)), 1);
    y = max(min(y, imageSize(1)), 1);
    idx = sub2ind(imageSize, y, x);
end
