function segmentedMaps = Segmentation(loaded_images)
% SEGMENTATION Processes a cell array of RGB images and returns segmented label maps.
% Input:
%   loaded_images - cell array of RGB images
% Output:
%   segmentedMaps - cell array of labeled segmentation maps

    numImages = length(loaded_images);
    segmentedMaps = cell(1, numImages);

    for i = 1:numImages
        segmentedMaps{i} = segment_image(loaded_images{i});
    end
end

function finalLabelMap = segment_image(I)
% SEGMENT_IMAGE Segments a single RGB image into labeled regions.
% Returns a map of classes like water, land, city, snow, etc.

    gray = rgb2gray(I);
    Ihsv = rgb2hsv(I);
    I_double = im2double(I);
    R = I_double(:,:,1); G = I_double(:,:,2); B = I_double(:,:,3);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    % Step 1: Detect basic classes
    labelSnowWater = detection_Snow_Water(I, gray, R, G, B);  % Initial snow/water detection
    labelSnowWater = detect_water(labelSnowWater, H, S, V, R, G, B);  % Refine water detection
    labelCityLand  = detect_city_land(I, H, S, V, R, G, B);  % Detect urban and land regions 
    %labelCityLand  = detect_forest(labelCityLand, H, S, V, R, G, B);  
    %labelRivers    = detectRivers(I, gray, R, G, B); 

   % Step 2: Combine label maps
    finalLabelMap = labelSnowWater;
    finalLabelMap(labelCityLand == 2) = 2;  % Land
    finalLabelMap(labelCityLand == 3) = 3;  % Urban
    finalLabelMap(labelCityLand == 7) = 7;  % Forest (optional)
    %finalLabelMap(labelRivers == 5)   = 5;  

    % Step 3: Classify remaining unclassified pixels using color matching
    finalLabelMap = classify_unclassified_by_color(finalLabelMap, R, G, B, H, S, V);
    %finalLabelMap = convert_water_surrounded_by_city(finalLabelMap);
    %finalLabelMap = resolve_water_forest_conflict(finalLabelMap, gray, H, S, V, R, G, B);
    
    
    % Merge certain similar classes for simplicity
    finalLabelMap(finalLabelMap == 7) = 1;  % Forest -> Water
    finalLabelMap(finalLabelMap == 6) = 2;  % Sand -> Land
    finalLabelMap(labelSnowWater == 4) = 4; % Re-apply Snow over everything else

    % Optional: show_segmented_map(I, finalLabelMap);  % Uncomment to visualize
end


%% Detection Functions

%% detect_city_land -> returns label map containing 0,2,3

function [labelMap] = detect_city_land(I, H, S, V, R, G, B)
% DETECT_CITY_LAND identifies city and land regions using texture and color.

    gray = im2double(rgb2gray(I));
    stdMap = mat2gray(stdfilt(gray, true(15)));  % Compute local texture
    baseMask = stdMap > 0.15;  % High texture regions = likely man-made
    cityMask = imfill(imclose(bwareaopen(baseMask, 100), strel('disk', 5)), 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    keepIdx = find([props.Area] >= 500);  % Ignore tiny regions
    labelMap = zeros(size(cityMask), 'uint8');

    % Analyze each region's color to classify
    for i = 1:length(keepIdx)
        pix = props(keepIdx(i)).PixelIdxList;
        h = H(pix); s = S(pix); v = V(pix);
        numPix = numel(pix);

        % Heuristics to detect red roofs, tan areas, or vegetation
        isRedRoof = (h < 0.05 | h > 0.95) & s > 0.3;
        isTanLike = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.4;
        greenish  = h > 0.2 & h < 0.45 & s > 0.25;

        % Compute feature metrics
        RedSum = sum(isRedRoof);
        TanRatio = sum(isTanLike) / numPix;
        GreenRatio = sum(greenish) / numPix;
        MeanSat = mean(s); MeanVal = mean(v); VarVal = std(v);

        % Label assignment logic
        label = 2;  % Default: land
        if RedSum > 28 || GreenRatio > 0.42 || ...
           (TanRatio > 0.1 && GreenRatio < 0.4 && MeanVal > 0.5)
            label = 3;  % Urban
        elseif MeanSat < 0.1 && VarVal < 0.02
            label = 0;  % Unclassified (possibly snow/flat)
        end
        labelMap(pix) = label;
    end

    % If image is mostly city, convert all detected to urban
    if sum(labelMap(:) == 3) / numel(labelMap) > 0.75
        labelMap(labelMap > 0) = 3;
    end
end

%% detectRivers -> returns label map containing 0,5
function finalRiverMask = detectRivers(rgbImg, grayImg, R, G, B)
% DETECTRIVERS attempts to identify river-like regions using both color and shape heuristics.
% Returns a label map where river pixels are labeled as 5.

    mask = true(size(grayImg));

    % Normalize channels to range [0, 1]
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;

    % Step 1: Color and shape-based candidate detection
    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.7) & (G > 0.7) & (B > 0.7);
    riverCandidates = (greenish | whiteish) & mask;
    riverCandidates = bwareaopen(riverCandidates, 100);

    % Remove holes and smooth shapes
    riverMaskColorShape = imclose(riverCandidates, strel('disk', 5));

    % Step 1.5: Filter for long, thin regions (likely rivers)
    CC = bwconncomp(riverMaskColorShape);
    stats = regionprops(CC, 'PixelIdxList', 'MajorAxisLength', 'MinorAxisLength', 'Area', 'Solidity');
    filteredMask = false(size(riverMaskColorShape));

    for i = 1:CC.NumObjects
        major = stats(i).MajorAxisLength;
        minor = stats(i).MinorAxisLength;
        area = stats(i).Area;
        solidity = stats(i).Solidity;

        if minor == 0, continue; end
        aspectRatio = major / minor;

        if aspectRatio > 3 && area < 30000 && solidity < 0.95
            filteredMask(CC.PixelIdxList{i}) = true;
        end
    end
    riverMaskColorShape = filteredMask;

    % Step 2: Sobel edge-based detection (alternative shape cue)
    edgesSobel = edge(im2double(grayImg), 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 100);
    BW_Sobel = imclose(BW_Sobel, strel('disk', 5));

    % Filter elongated edge shapes
    CC_shape = bwconncomp(BW_Sobel);
    stats_shape = regionprops(CC_shape, 'PixelIdxList', 'MajorAxisLength', 'MinorAxisLength');
    filteredMask = false(size(riverMaskColorShape));

    for i = 1:CC_shape.NumObjects
        major = stats_shape(i).MajorAxisLength;
        minor = stats_shape(i).MinorAxisLength;
        if minor == 0, continue; end
        if major / minor > 2
            filteredMask(CC_shape.PixelIdxList{i}) = true;
        end
    end
    BW_Sobel = filteredMask;

    % Step 3: Combine both candidates
    combinedCandidates = riverMaskColorShape | BW_Sobel;

    % Step 4: Connect fragmented river parts based on proximity
    connectedEdges = connectRegionsByEdgeProximity(combinedCandidates, 10, 2);

    % Step 5: Final filter by size and elongation
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');
    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = zeros(size(grayImg));

    for i = 1:CC.NumObjects
        if stats(i).Area < 100, continue; end
        pixIdx = stats(i).PixelIdxList;
        [yCoords, xCoords] = ind2sub(size(mask), pixIdx);
        coords = [xCoords, yCoords];
        if numel(pixIdx) < 2, continue; end

        % Estimate region span via convex hull or max pairwise distance
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
            finalRiverMask(pixIdx) = 5;  % Label as river
        end
    end
end

%% detection_Snow_Water -> returns label map containing 0,1,4 
function labelMap = detection_Snow_Water(I, grayImg, Rn, Gn, Bn)
% DETECTION_SNOW_WATER uses smoothness and brightness to detect snow/water.

    imgSize = size(grayImg);
    edgeDensity = conv2(edge(im2double(grayImg), 'Canny'), ones(15)/225, 'same');
    smoothMask = edgeDensity < 0.05;  % Smooth regions
    smoothMask = imfill(bwareaopen(smoothMask, 500), 'holes');

    % Keep smooth, sufficiently large, round regions
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList', 'Area', 'Eccentricity');
    for k = 1:numel(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(stats(k).PixelIdxList) = false;
        end
    end

    % Classify regions as snow or water
    labelMap = zeros(imgSize, 'uint8');
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');
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

    % Pixel-level water fallback detection
    blueDominant = (Bn > 0.35) & (Bn > Rn + 0.08) & (Bn > Gn + 0.05);
    cyanLike = (Bn > 0.4) & (Gn > 0.4) & (Rn < 0.4) & (abs(Bn - Gn) < 0.15);
    pixelWater = blueDominant | cyanLike;

    % Remove small blobs
    L = bwlabel(pixelWater);
    statsWater = regionprops(L, 'Area');
    for i = 1:numel(statsWater)
        if statsWater(i).Area < 3000
            pixelWater(L == i) = false;
        end
    end
    labelMap(pixelWater) = 1;

    % If snow exists, reinforce white areas as snow
    if snowFound
        whiteish = (Rn > 0.7) & (Gn > 0.7) & (Bn > 0.7);
        labelMap(whiteish) = 4;
    end
end

function labelMap = detect_water(labelMap, H, S, V, R, G, B)
% DETECT_WATER refines water detection using HSV thresholds.

    mask = (labelMap == 0) | (labelMap == 2);  % Only unclassified or land

    isDark = V < 0.4;
    isLowSat = S < 0.4;
    notGreenish = ~(G > R + 0.05 & G > B + 0.05 & G > 0.3);
    isBlueDominant = B > R & B > G;

    waterPixels = mask & isDark & isLowSat & notGreenish & isBlueDominant;
    labelMap(waterPixels) = 1;
end


%% detection unclassified

function labelMap = classify_unclassified_by_color(labelMap, R, G, B, H, S, V)
% CLASSIFY_UNCLASSIFIED_BY_COLOR assigns a class to unclassified regions based on average RGB.

    COLOR_THRESHOLDS = struct( ...
        'Water',  [0.33, 0.38, 0.42], ...
        'City',   [0.4, 0.4, 0.4], ...
        'Sand',   [0.85, 0.75, 0.55], ...
        'Forest', [0.25, 0.40, 0.26] ...
    );

    CC = bwconncomp(labelMap == 0);
    stats = regionprops(CC, 'PixelIdxList');
    labelList = [1, 3, 6, 7];  % Water, City, Sand, Forest

    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        colorVec = [mean(R(idx)), mean(G(idx)), mean(B(idx))];
        hMean = mean(H(idx));
        if hMean > 0.42
            labelMap(idx) = 1; % Water
            continue;
        end

        % Find closest prototype color
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
%% Plotting

function show_segmented_map(I, labelMap)
% SHOW_SEGMENTED_MAP displays original image next to its segmented label map.

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

%% Approachers that didn't work
function labelMap = resolve_water_forest_conflict(labelMap, gray, H, S, V, R, G, B)
% RESOLVE_WATER_FOREST_CONFLICT tries to fix forest regions misclassified due to greenish water.

    % Step 1: Get candidate forest regions with green hues
    greenHue = H > 0.2 & H < 0.45;
    mediumSat = S > 0.2;
    mediumBright = V > 0.2;
    greenish = greenHue & mediumSat & mediumBright;
    forestMask = labelMap == 7;
    candidates = forestMask & greenish;

    % Step 2: Compute low-texture regions (likely smooth water)
    texture = stdfilt(gray, true(5));
    lowTexture = texture < 0.05;

    % Step 3: Keep only smooth, elongated regions
    CC = bwconncomp(candidates & lowTexture);
    stats = regionprops(CC, 'PixelIdxList', 'MajorAxisLength', 'MinorAxisLength', 'Solidity');

    for i = 1:CC.NumObjects
        major = stats(i).MajorAxisLength;
        minor = stats(i).MinorAxisLength;
        solidity = stats(i).Solidity;

        if minor == 0, continue; end
        aspectRatio = major / minor;

        if aspectRatio > 2 && solidity > 0.8
            labelMap(stats(i).PixelIdxList) = 1;  % Reclassify as water
        end
    end
end

function labelMap = convert_water_surrounded_by_city(finalLabelMap)
% CONVERT_WATER_SURROUNDED_BY_CITY relabels small water regions as city if
% they are almost entirely surrounded by urban regions.

    labelMap = finalLabelMap;
    waterLabel = 1;
    cityLabel = 3;

    waterMask = (finalLabelMap == waterLabel);
    CC = bwconncomp(waterMask);
    cityMask = (finalLabelMap == cityLabel);
    se = strel('square', 3);  % For 8-connected neighborhood

    for k = 1:CC.NumObjects
        regionIdx = CC.PixelIdxList{k};
        regionMask = false(size(finalLabelMap));
        regionMask(regionIdx) = true;

        borderMask = imdilate(regionMask, se) & ~regionMask;

        cityTouchCount = sum(cityMask(borderMask));
        totalBorder = nnz(borderMask);

        if totalBorder > 0 && (cityTouchCount / totalBorder) >= 0.9
            labelMap(regionIdx) = cityLabel;
        end
    end
end

function labelMap = detect_forest(labelMap, H, S, V, R, G, B)
% DETECT_FOREST identifies forest areas based on green dominance in HSV and RGB spaces.
% It relabels regions marked as land/city to forest where appropriate.

    mask = (labelMap == 2) | (labelMap == 3);  % Land or City
    isGreenHSV = H > 0.2 & H < 0.45 & S > 0.2 & V > 0.2;
    isGreenRGB = (G > R + 0.05) & (G > B + 0.05) & (G > 0.3);
    forestPixels = mask & isGreenHSV & isGreenRGB;

    labelMap(forestPixels) = 7;  % Forest
end
