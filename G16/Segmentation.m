function segmentedMaps = Segmentation(loaded_images)
% Accepts a cell array of loaded RGB images and returns their segmented label maps.

    numImages = length(loaded_images);
    segmentedMaps = cell(1, numImages);

    for i = 1:numImages
        segmentedMaps{i} = segment_image(loaded_images{i});
    end
end

function finalLabelMap = segment_image(I)
% Processes a single RGB image and returns the labeled segmentation map.

    gray = rgb2gray(I);
    Ihsv = rgb2hsv(I);
    I_double = im2double(I);
    R = I_double(:,:,1); G = I_double(:,:,2); B = I_double(:,:,3);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    % --- Step 1: Detection modules ---
    labelSnowWater = detection_Snow_Water(I, gray, R, G, B);  
    labelSnowWater = detect_water(labelSnowWater, H, S, V, R, G, B); 
    labelCityLand  = detect_city_land(I, H, S, V, R, G, B);  
    %labelCityLand  = detect_forest(labelCityLand, H, S, V, R, G, B);  
    %labelRivers    = detectRivers(I, gray, R, G, B); 

    % --- Step 2: Combine masks ---
    finalLabelMap = labelSnowWater;
    finalLabelMap(labelCityLand == 2) = 2;  
    finalLabelMap(labelCityLand == 3) = 3;  
    finalLabelMap(labelCityLand == 7) = 7;  
    %finalLabelMap(labelRivers == 5)   = 5;  

    % --- Step 3: Resolve ambiguous areas ---
    finalLabelMap = classify_unclassified_by_color(finalLabelMap, R, G, B, H, S, V);
    %finalLabelMap = convert_water_surrounded_by_city(finalLabelMap);
    %finalLabelMap = resolve_water_forest_conflict(finalLabelMap, gray, H, S, V, R, G, B);
    
    
    % Combine Forest/Water
    finalLabelMap(finalLabelMap == 7)   = 1;
    % Combine Sand/Land
    finalLabelMap(finalLabelMap == 6)   = 2;
    finalLabelMap(labelSnowWater == 4) = 4; % Re-apply Snow over everything else
    % (Optional visualization: remove comment to enable)
    % show_segmented_map(I, finalLabelMap);
end


%% Detection Functions

%% detect_city_land -> returns label map containing 0,2,3
% new -> 90% city detection

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
function finalRiverMask = detectRivers(rgbImg, grayImg, R, G, B)
   
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
    edgesSobel = edge(im2double(grayImg), 'Sobel');

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
function labelMap = detection_Snow_Water(I, grayImg, Rn, Gn, Bn)
    imgSize = size(grayImg);
    edgeDensity = conv2(edge(im2double(grayImg), 'Canny'), ones(15)/225, 'same');

    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(bwareaopen(smoothMask, 500), 'holes');

    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList', 'Area', 'Eccentricity');

    for k = 1:numel(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 1000
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

function labelMap = detection_Snow_Water2(I, grayImg, Rn, Gn, Bn)
    imgSize = size(grayImg);
    edgeDensity = conv2(edge(im2double(grayImg), 'Canny'), ones(15)/225, 'same');

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

    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        rMean = mean(Rn(idx));
        gMean = mean(Gn(idx));
        bMean = mean(Bn(idx));

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            labelMap(idx) = 4;  % Snow
        elseif bMean > rMean && bMean > gMean
            labelMap(idx) = 1;  % Water
        end
    end

    % Filter small snow areas
    snowMask = (labelMap == 4);
    Lsnow = bwlabel(snowMask);
    statsSnow = regionprops(Lsnow, 'Area');
    for i = 1:numel(statsSnow)
        if statsSnow(i).Area < 3000
            labelMap(Lsnow == i) = 0;
        end
    end

    % Filter small water areas
    blueDominant = (Bn > 0.35) & (Bn > Rn + 0.08) & (Bn > Gn + 0.05);
    cyanLike = (Bn > 0.4) & (Gn > 0.4) & (Rn < 0.4) & (abs(Bn - Gn) < 0.15);
    pixelWater = blueDominant | cyanLike;

    L = bwlabel(pixelWater);
    statsWater = regionprops(L, 'Area');
    for i = 1:numel(statsWater)
        if statsWater(i).Area < 3000
            pixelWater(L == i) = false;
        end
    end
    labelMap(pixelWater) = 1;

    % Apply whiteish map if any snow label exists
    if any(labelMap(:) == 4)
        whiteish = (Rn > 0.7) & (Gn > 0.7) & (Bn > 0.7);
        labelMap(whiteish) = 4;
    end

    % --- SHADOW DETECTION ---
    brightness = (Rn + Gn + Bn) / 3;
    globalMean = mean(brightness(:));
    globalStd = std(brightness(:));

    % Shadows = regions significantly darker than global average
    shadowThresh = globalMean - 1.0 * globalStd;  % adjust factor as needed
    shadowMask = brightness < shadowThresh;

    % Optional: clean up small shadow patches
    Lshadow = bwlabel(shadowMask);
    statsShadow = regionprops(Lshadow, 'Area');
    for i = 1:numel(statsShadow)
        if statsShadow(i).Area < 3000
            shadowMask(Lshadow == i) = false;
        end
    end

    labelMap(shadowMask) = 8;  % Shadow class
end


function labelMap = detect_water(labelMap, H, S, V, R, G, B)
    % Only look at pixels labeled as unclassified or land
    mask = (labelMap == 0) | (labelMap == 2);

    % Water tends to be dark and low saturation
    isDark = V < 0.4;
    isLowSat = S < 0.4;

    % Avoid green-dominant areas (likely forest)
    notGreenish = ~(G > R + 0.05 & G > B + 0.05 & G > 0.3);

    % Clear water is blue-dominant
    isBlueDominant = B > R & B > G;

    % Combine heuristics
    waterPixels = mask & isDark & isLowSat & notGreenish & isBlueDominant;

    % Assign label 1 to detected water
    labelMap(waterPixels) = 1;
end

function labelMap = resolve_water_forest_conflict2(labelMap, gray, H, S, V, R, G, B)

    % Step 1: Candidate ambiguous regions (initially forest or land)
    candidateMask = (labelMap == 2 | labelMap == 7);  % land/forest regions

    hueMask = H > 0.2 & H < 0.5;
    colorSimilarity = abs(G - B) < 0.08;
    saturationOK = S > 0.2;
    brightnessOK = V > 0.2;

    candidates = candidateMask & hueMask & colorSimilarity & saturationOK & brightnessOK;

    % Step 2: Compute texture features (local std dev or entropy)
    localStd = stdfilt(gray, true(5));
    localEntropy = entropyfilt(gray, true(5));

    smooth = (localStd < 0.05) & (localEntropy < 3.5);  % tune these thresholds

    % Step 3: Reassign smooth-looking forest to water
    reassignToWater = candidates & smooth;

    % Optional: reassign very rough water to forest if misclassified
    % reassignToForest = (labelMap == 1) & (~smooth) & (localEntropy > 4.5);

    labelMap(reassignToWater) = 1;
    % labelMap(reassignToForest) = 7;

end
function labelMap = resolve_water_forest_conflict(labelMap, gray, H, S, V, R, G, B)

    % Step 1: Find ambiguous "green" regions marked as forest
    greenHue = H > 0.2 & H < 0.45;
    mediumSat = S > 0.2;
    mediumBright = V > 0.2;
    greenish = greenHue & mediumSat & mediumBright;

    % Where it's labeled forest
    forestMask = labelMap == 7;

    % Candidate pixels that might be misclassified water
    candidates = forestMask & greenish;

    % Step 2: Compute local structure (stdfilt)
    texture = stdfilt(gray, true(5));
    lowTexture = texture < 0.05;

    % Step 3: Use regionprops to filter elongated smooth regions (i.e., water)
    CC = bwconncomp(candidates & lowTexture);
    stats = regionprops(CC, 'PixelIdxList', 'MajorAxisLength', 'MinorAxisLength', 'Solidity');

    for i = 1:CC.NumObjects
        major = stats(i).MajorAxisLength;
        minor = stats(i).MinorAxisLength;
        solidity = stats(i).Solidity;

        if minor == 0, continue; end
        aspectRatio = major / minor;

        % Water tends to be long and thin and smooth
        if aspectRatio > 2 && solidity > 0.8
            labelMap(stats(i).PixelIdxList) = 1;  % Water
        end
    end
end

function labelMap = convert_water_surrounded_by_city(finalLabelMap)
    labelMap = finalLabelMap;  % Copy input
    waterLabel = 1;
    cityLabel = 3;

    % Get water regions
    waterMask = (finalLabelMap == waterLabel);
    CC = bwconncomp(waterMask);
    cityMask = (finalLabelMap == cityLabel);

    % Structuring element for border detection
    se = strel('square', 3);

    for k = 1:CC.NumObjects
        regionIdx = CC.PixelIdxList{k};

        % Create binary mask of region
        regionMask = false(size(finalLabelMap));
        regionMask(regionIdx) = true;

        % Get border of region (dilate - region)
        borderMask = imdilate(regionMask, se) & ~regionMask;

        % Border pixels that touch city
        cityTouchCount = sum(cityMask(borderMask));
        totalBorder = nnz(borderMask);

        if totalBorder > 0 && (cityTouchCount / totalBorder) >= 0.9
            labelMap(regionIdx) = cityLabel;
        end
    end
end

%% detection unclassified

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
function finalMask = connectRegionsByEdgeProximity(binaryMask, maxDist, bridgeRadius)
    % CONNECTREGIONSBYEDGEPROXIMITY connects nearby regions using shortest path bridging,
    % optimized for speed and memory using knnsearch (no full pdist2 matrix).

    % Label connected components
    labeled = bwlabel(binaryMask);
    numRegions = max(labeled(:));
    finalMask = binaryMask;

    % Extract pixel coordinates for each region
    regionPixels = cell(numRegions, 1);
    for i = 1:numRegions
        [y, x] = find(labeled == i);
        regionPixels{i} = [x, y];  % use [x, y] format
    end

    % Try to connect each unique pair
    for i = 1:numRegions
        pts1 = regionPixels{i};
        if size(pts1,1) > 3000, continue; end  % skip overly large regions

        for j = i+1:numRegions
            pts2 = regionPixels{j};
            if size(pts2,1) > 3000, continue; end  % skip overly large regions

            % Use nearest neighbor instead of full pairwise distance
            try
                [idx, minDists] = knnsearch(pts2, pts1, 'K', 1);
            catch
                continue;  % skip problematic pairs
            end
            [minDist, minIdx] = min(minDists);

            if minDist <= maxDist
                pt1 = pts1(minIdx, :);
                pt2 = pts2(idx(minIdx), :);

                % Draw line between pt1 and pt2
                bridge = false(size(binaryMask));
                lineIdx = drawLineBetweenPoints(pt1, pt2, size(binaryMask));
                bridge(lineIdx) = true;

                % Thicken the bridge to ensure connection
                bridge = imdilate(bridge, strel('disk', bridgeRadius));
                finalMask = finalMask | bridge;
            end
        end
    end
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
