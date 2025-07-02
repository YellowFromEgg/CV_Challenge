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

parfor i = 1:length(imgPaths)
    segment_environment(imgPaths{i});
end

function segment_environment(imgPath)
    % Read image once
    I = imread(imgPath);
    gray = rgb2gray(I);
    Ihsv = rgb2hsv(I);
    I_double = im2double(I);
    R = I_double(:,:,1); G = I_double(:,:,2); B = I_double(:,:,3);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    % --- Step 1: Detection modules (pass precomputed data) ---
    labelSnowWater = detection_Snow_Water(I, gray, R, G, B);     
    labelCityLand  = detect_city_land(I, H, S, V, R, G, B);  % no forest here
    labelCityLand  = detect_forest(labelCityLand, H, S, V, R, G, B);  % forest added here     
    labelRivers    = detectRivers(I, gray, R, G, B);       

    % --- Step 2: Combine masks (preserve overwrite logic) ---
    finalLabelMap = labelSnowWater;

    finalLabelMap(labelCityLand == 2) = 2;  % Land
    finalLabelMap(labelCityLand == 3) = 3;  % City
    %finalLabelMap()
    finalLabelMap(labelCityLand == 7) = 7;  % Forest inside City
    finalLabelMap(labelRivers == 5) = 5;    % River

    % --- Step 3: Unclassified pixels by color ---
    finalLabelMap = classify_unclassified_by_color(finalLabelMap, R, G, B, H, S, V);

    % --- Visualization ---
    show_segmented_map(I, finalLabelMap);
    profile viewer
end


function segment_environment_old(imgPath)
    % Read input image
    I = imread(imgPath);

    %% Run detection modules
    %detects Snow, Water based on smoothness; 
    labelSnowWater = detection_Snow_Water(imgPath);     % 0,1,4
    %detects city and land based on edge densities
    labelCityLand  = detect_city_land(I);               % 0,2,3,7 (7 = forest inside city)
    %detects rivers/roads based on long thin edges
    labelRivers    = detectRivers(imgPath);             % 0,5


    % --- Step 2: Combine layers in correct overwrite order
    finalLabelMap = labelSnowWater;             % start with water/snow (1,4)

    finalLabelMap(labelCityLand == 2) = 2;      % Land
    finalLabelMap(labelCityLand == 3) = 3;      % City
    finalLabelMap(labelCityLand == 7) = 7;      % Forest inside City
    finalLabelMap(labelSnowWater == 4) = 4; % Snow
    finalLabelMap(labelRivers == 5) = 5;    % River

    % Fill unclassified using color-based classification
    finalLabelMap = classify_unclassified_by_color(finalLabelMap, I);

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
% new -> 90% city detection
function [labelMap] = detect_city_land_old(I)
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
        if cityRatio > 0.7
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

function [labelMap] = detect_city_land(I, H, S, V, R, G, B)
    gray = im2double(rgb2gray(I));
    stdMap = mat2gray(stdfilt(gray, true(15)));
    baseMask = stdMap > 0.15;
    cityMask = imfill(imclose(bwareaopen(baseMask, 100), strel('disk', 5)), 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    keepIdx = find([props.Area] >= 500);
    labelMap = zeros(size(cityMask), 'uint8');

    % Region-wise classification
    for i = 1:length(keepIdx)
        pix = props(keepIdx(i)).PixelIdxList;
        h = H(pix); s = S(pix); v = V(pix);
        numPix = numel(pix);

        isRedRoof = (h < 0.05 | h > 0.95) & s > 0.3;
        isTanLike = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.4;
        greenish  = h > 0.2 & h < 0.45 & s > 0.25;

        RedSum = sum(isRedRoof);
        TanRatio = sum(isTanLike) / numPix;
        GreenRatio = sum(greenish) / numPix;
        MeanSat = mean(s); MeanVal = mean(v); VarVal = std(v);

        label = 2;
        if RedSum > 28 || GreenRatio > 0.42 || ...
           (TanRatio > 0.1 && GreenRatio < 0.4 && MeanVal > 0.5)
            label = 3;
        elseif MeanSat < 0.1 && VarVal < 0.02
            label = 0;
        end
        labelMap(pix) = label;

    end

    if sum(labelMap(:) == 3) / numel(labelMap) > 0.75
        labelMap(labelMap > 0) = 3;
    end
end

%% detectRivers -> returns label map containing 0,5
function finalRiverMask = detectRivers_old(imgPath)
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

function finalRiverMask = detectRivers(I, grayImg, R, G, B)
    imgSize = size(grayImg);
    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.7) & (G > 0.7) & (B > 0.7);
    riverCandidates = bwareaopen((greenish | whiteish), 100);
    riverMaskColorShape = imclose(riverCandidates, strel('disk', 5));

    CC = bwconncomp(riverMaskColorShape);
    stats = regionprops(CC, 'PixelIdxList', 'MajorAxisLength', 'MinorAxisLength', 'Area', 'Solidity');
    filtered = false(imgSize);

    for i = 1:CC.NumObjects
        major = stats(i).MajorAxisLength;
        minor = stats(i).MinorAxisLength;
        if minor == 0, continue; end
        aspectRatio = major / minor;
        if aspectRatio > 3 && stats(i).Area < 30000 && stats(i).Solidity < 0.95
            filtered(stats(i).PixelIdxList) = true;
        end
    end

    % Edge-based detection
    edges = imfill(edge(grayImg, 'Sobel'), 'holes');
    edges = bwareaopen(imclose(edges, strel('disk', 5)), 100);

    CC_edge = bwconncomp(edges);
    stats_edge = regionprops(CC_edge, 'PixelIdxList', 'MajorAxisLength', 'MinorAxisLength');
    filteredEdge = false(imgSize);

    for i = 1:CC_edge.NumObjects
        minor = stats_edge(i).MinorAxisLength;
        if minor == 0, continue; end
        if stats_edge(i).MajorAxisLength / minor > 2
            filteredEdge(stats_edge(i).PixelIdxList) = true;
        end
    end

    combined = filtered | filteredEdge;
    connected = connectRegionsByEdgeProximity(combined, 10, 2);

    % Final filtering by spatial extent
    CC = bwconncomp(connected);
    stats = regionprops(CC, 'PixelIdxList', 'Area');
    imageDiagonal = norm(imgSize);
    minDist = imageDiagonal / 6;
    finalRiverMask = zeros(imgSize);

    for i = 1:CC.NumObjects
        pix = stats(i).PixelIdxList;
        if numel(pix) < 100, continue; end
        [y, x] = ind2sub(imgSize, pix);
        coords = [x, y];
        if numel(pix) > 300
            try
                K = convhull(coords(:,1), coords(:,2));
                maxDist = max(pdist(coords(K,:)));
            catch
                maxDist = max(pdist(coords));
            end
        else
            maxDist = max(pdist(coords));
        end
        if maxDist >= minDist
            finalRiverMask(pix) = 5;
        end
    end
end

%% detection_Snow_Water -> returns label map containing 0,1,4 
function labelMap = detection_Snow_Water_old(imgPath)
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

function labelMap = detection_Snow_Water(I, grayImg, Rn, Gn, Bn)
    imgSize = size(grayImg);
    edgeDensity = conv2(double(edge(grayImg, 'Canny')), ones(15)/225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(bwareaopen(smoothMask, 500), 'holes');

    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList', 'Area', 'Eccentricity');

    for k = 1:numel(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(stats(k).PixelIdxList) = false;
        end
    end

    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');

    labelMap = zeros(imgSize, 'uint8');
    snowFound = false;

    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        rMean = mean(Rn(idx));
        gMean = mean(Gn(idx));
        bMean = mean(Bn(idx));

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            labelMap(idx) = 4;  % Snow
            snowFound = true;
        elseif bMean > rMean && bMean > gMean
            labelMap(idx) = 1;  % Water
        end
    end

    % Pixel-based water detection
    blueDominant = (Bn > 0.35) & (Bn > Rn + 0.08) & (Bn > Gn + 0.05);
    cyanLike = (Bn > 0.4) & (Gn > 0.4) & (Rn < 0.4) & (abs(Bn - Gn) < 0.15);
    pixelWater = blueDominant | cyanLike;

    % Filter small areas
    L = bwlabel(pixelWater);
    statsWater = regionprops(L, 'Area');
    for i = 1:numel(statsWater)
        if statsWater(i).Area < 3000
            pixelWater(L == i) = false;
        end
    end
    labelMap(pixelWater) = 1;

    if snowFound
        whiteish = (Rn > 0.7) & (Gn > 0.7) & (Bn > 0.7);
        labelMap(whiteish) = 4;
    end
end

%% detection unclassified
function labelMap = classify_unclassified_by_color_old(labelMap, rgbImage)
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

function labelMap = classify_unclassified_by_color(labelMap, R, G, B, H, S, V)
    COLOR_THRESHOLDS = struct( ...
        'Water',  [0.33, 0.38, 0.42], ...
        'City',   [0.4, 0.4, 0.4], ...
        'Sand',   [0.85, 0.75, 0.55], ...
        'Forest', [0.25, 0.40, 0.26] ...
    );

    CC = bwconncomp(labelMap == 0);
    stats = regionprops(CC, 'PixelIdxList');
    labelList = [1, 3, 6, 7];  % Corresponding to color prototypes

    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        rMean = mean(R(idx));
        gMean = mean(G(idx));
        bMean = mean(B(idx));
        hMean = mean(H(idx));
        sMean = mean(S(idx));
        vMean = mean(V(idx));

        if hMean > 0.42
            labelMap(idx) = 1; % Water
            continue;
        end

        colorVec = [rMean, gMean, bMean];
        dists = [
            norm(colorVec - COLOR_THRESHOLDS.Water);
            norm(colorVec - COLOR_THRESHOLDS.City);
            norm(colorVec - COLOR_THRESHOLDS.Sand);
            norm(colorVec - COLOR_THRESHOLDS.Forest)
        ];

        [~, minIdx] = min(dists);
        labelMap(idx) = labelList(minIdx);
    end
end

%% detect forests
function labelMap = detect_forest(labelMap, H, S, V, R, G, B)
    % Detect forest in pixels labeled as land or city (2 or 3)
    mask = (labelMap == 2) | (labelMap == 3);

    % Create boolean masks for green-like appearance
    isGreenHSV = H > 0.2 & H < 0.45 & S > 0.2 & V > 0.2;
    isGreenRGB = (G > R + 0.05) & (G > B + 0.05) & (G > 0.3);

    forestPixels = mask & isGreenHSV & isGreenRGB;

    % Assign label 7 (Forest) to these pixels
    labelMap(forestPixels) = 7;
end
%% Plotting

function show_segmented_map(I, labelMap)
    cmap = [
        0.8 0.8 0.8; 0 0 1; 0.6 0.4 0.2; 1 0 0; 1 1 1; 
        0 1 1; 1 0.9 0.5; 0 0.6 0
    ];
    classNames = {'Unclassified','Water','Land','Urban/Agriculture','Snow','River/Road','Sand','Forest'};

    figure('Name', 'Final Segmentation Result', 'Position', [100 100 1400 600]);
    subplot(1,2,1); imshow(I); title('Original Image');
    subplot(1,2,2); imagesc(labelMap); axis image off;
    colormap(gca, cmap); caxis([0 7]);
    colorbar('Ticks', 0:7, 'TickLabels', classNames);
    title('Segmented Map');
end


%% Helper Functions
function finalMask = connectRegionsByEdgeProximity2(binaryMask, maxDist, bridgeRadius)
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
function finalMask = connectRegionsByEdgeProximity(binaryMask, maxDist, bridgeRadius)
    labeled = bwlabel(binaryMask);
    numRegions = max(labeled(:));
    finalMask = binaryMask;
    regionPixels = cell(numRegions, 1);

    for i = 1:numRegions
        [y, x] = find(labeled == i);
        regionPixels{i} = [x, y];
    end

    for i = 1:numRegions
        for j = i+1:numRegions
            pts1 = regionPixels{i};
            pts2 = regionPixels{j};
            D = pdist2(pts1, pts2);
            [minD, idx] = min(D(:));
            if minD <= maxDist
                [p1, p2] = ind2sub(size(D), idx);
                pt1 = pts1(p1, :);
                pt2 = pts2(p2, :);

                bridge = false(size(binaryMask));
                bridge(drawLineBetweenPoints(pt1, pt2, size(binaryMask))) = true;
                bridge = imdilate(bridge, strel('disk', bridgeRadius));
                finalMask = finalMask | bridge;
            end
        end
    end
end

function idx = drawLineBetweenPoints2(pt1, pt2, imageSize)
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
function idx = drawLineBetweenPoints(pt1, pt2, imageSize)
    x1 = round(pt1(1)); y1 = round(pt1(2));
    x2 = round(pt2(1)); y2 = round(pt2(2));
    n = max(abs([x2 - x1, y2 - y1])) + 1;
    x = round(linspace(x1, x2, n));
    y = round(linspace(y1, y2, n));
    x = max(min(x, imageSize(2)), 1);
    y = max(min(y, imageSize(1)), 1);
    idx = sub2ind(imageSize, y, x);
end
