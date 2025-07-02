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
    identify(imgPaths{i});
end

function identify(imgPath)
    rgbImg = imread(imgPath);

    % --- Step 1: Smooth Regions (Snow / Water) ---
    [labelSmooth, maskUnclassified] = classifySmoothRegionsRefined(rgbImg);

    % --- Step 2: River Detection by Color/Shape ---
    [riverMask1, maskUnclassified] = classifyRiversThinCurves(rgbImg, maskUnclassified);

    % --- Step 3: River Detection by Sobel Edge Structures ---
    [riverMask2, maskUnclassified] = detectRiverViaSobelEdges(rgbImg, maskUnclassified);

    % Combine river detections
    combinedRiverMask = riverMask1 | riverMask2;
    maskUnclassified(combinedRiverMask) = false;

    % --- Step 4: Remaining Pixel Classification ---
    labelRemaining = classifyRemainingPixels(rgbImg, maskUnclassified);

    % --- Step 5: Combine All Results ---
    labelCombined = zeros(size(labelSmooth));
    labelCombined(labelSmooth == 1) = 5;  % Snow
    labelCombined(labelSmooth == 2) = 6;  % Water
    labelCombined(combinedRiverMask) = 2; % River
    labelCombined(maskUnclassified & labelRemaining == 1) = 1;  % Forest
    labelCombined(maskUnclassified & labelRemaining == 3) = 3;  % City
    labelCombined(maskUnclassified & labelRemaining == 4) = 4;  % Rest

    % --- Color Setup ---
    cmap = [
    0 1 0;        % Forest
    0 0 1;        % River
    0.5 0.5 0.5;  % City
    0.6 0.4 0.2;  % Rest
    1 1 1;        % Snow
    0 0.6 1;      % Water
    0.96 0.87 0.7; % Sand
    0.8 0.6 0.2   % Agriculture
    ];
    labels = {'Forest','River','City','Rest','Snow','Water','Sand','Agriculture'};

    % --- Plot Results ---
    figure;

    subplot(2,2,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,2,2);
    imagesc(labelSmooth); axis image off;
    title('Step 1: Smooth Region Classification');
    colormap(gca, [1 1 1; 0 0.6 1]); caxis([1 2]);
    colorbar('Ticks', [1 2], 'TickLabels', {'Not Smooth','Smooth'});

    subplot(2,2,3);
    imshow(combinedRiverMask); title('Step 2: River Detection');

    subplot(2,2,4);
    imagesc(labelCombined); axis image off;
    title('Step 3: Final Land Cover Classification');
    colormap(gca, cmap); caxis([1 8]);
    colorbar('Ticks', 1:8, 'TickLabels', labels);
end

function [labelMap, maskUnclassified] = classifySmoothRegionsRefined(img)
    R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
    grayImg = rgb2gray(img);
    edges = edge(grayImg, 'Canny');

    % Edge density map
    edgeDensity = conv2(double(edges), ones(15) / 225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    % FIRST: get connected components
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'Area', 'Eccentricity');

    % Remove irregular/small shapes
    for k = 1:length(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(CC.PixelIdxList{k}) = false;
        end
    end

    % Get updated CC and stats after cleaning
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');

    % Prepare output maps
    labelMap = zeros(size(grayImg));
    maskUnclassified = true(size(grayImg));

    % Classify smooth regions
    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        rMean = mean(double(R(idx))/255);
        gMean = mean(double(G(idx))/255);
        bMean = mean(double(B(idx))/255);

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            label = 1; % Snow
        elseif bMean > rMean && bMean > gMean
            label = 2; % Water
        else
            label = 0;
        end

        labelMap(idx) = label;
        if label ~= 0
            maskUnclassified(idx) = false;
        end
    end
end

function [riverMask, updatedMaskUnclassified] = detectRiverViaSobelEdges(rgbImg, mask)
    grayImg = rgb2gray(rgbImg);
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);

    % Connect fragmented edges
    connectedEdges = imclose(BW_Sobel, strel('disk', 3));

    % Filter connected regions by diagonal length
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'BoundingBox', 'PixelIdxList');

    [h, w] = size(grayImg);
    minDiagonal = 0.1 * sqrt(h^2 + w^2);

    riverMask = false(h, w);
    for i = 1:CC.NumObjects
        pixelIdx = stats(i).PixelIdxList;
        
        bbox = stats(i).BoundingBox;
        diagLength = sqrt(bbox(3)^2 + bbox(4)^2);
        if diagLength >= minDiagonal
            riverMask(pixelIdx) = true;
        end
       
    end

    % Update unclassified mask
    updatedMaskUnclassified = mask & ~riverMask;
end
function [riverMask, updatedMaskUnclassified] = classifyRiversThinCurves(rgbImg, mask)
    % Normalize channels
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;

    % --- 1. Create river candidate mask ---
    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.75) & (G > 0.75) & (B > 0.75);
    riverCandidates = (greenish | whiteish) & mask;

    % Remove tiny noise
    riverCandidates = bwareaopen(riverCandidates, 20);

    % --- 2. Handle whiteish separately ---
    whiteMask = whiteish & mask;
    whiteMask = bwareaopen(whiteMask, 20);
    CC_white = bwconncomp(whiteMask);
    stats_white = regionprops(CC_white, 'Area', 'Eccentricity', 'PixelIdxList');

    validWhite = false(size(R));
    for i = 1:CC_white.NumObjects
        if stats_white(i).Eccentricity > 0.85 && stats_white(i).Area < 300
            validWhite(stats_white(i).PixelIdxList) = true;
        end
    end

    % --- 3. Combine and filter connected regions by skeleton length ---
    mergedRiverMask = (greenish & mask) | validWhite;
    CC = bwconncomp(mergedRiverMask);
    stats = regionprops(CC, 'PixelIdxList', 'BoundingBox', 'Area');

    imageDiagonal = sqrt(size(mergedRiverMask,1)^2 + size(mergedRiverMask,2)^2);
    minLength = 0.1 * imageDiagonal;

    riverMask = false(size(mergedRiverMask));

    for i = 1:CC.NumObjects
        if stats(i).Area < 100  % Skip very small regions
            continue;
        end

        % Bounding box crop to speed up processing
        bbox = round(stats(i).BoundingBox);
        x1 = max(1, bbox(1));
        y1 = max(1, bbox(2));
        x2 = min(size(mask,2), x1 + bbox(3) - 1);
        y2 = min(size(mask,1), y1 + bbox(4) - 1);

        % Extract region and compute skeleton
        subMask = false(size(mask));
        subMask(stats(i).PixelIdxList) = true;
        subRegion = subMask(y1:y2, x1:x2);
        skel = bwskel(subRegion);

        if nnz(skel) >= minLength
            riverMask(stats(i).PixelIdxList) = true;
        end
    end

    % --- 4. Finalize ---
    updatedMaskUnclassified = mask & ~riverMask;
end

function cityMask = detectCityBlocks(rgbImg, mask)
    grayImg = rgb2gray(rgbImg);
    edges = edge(grayImg, 'Sobel');

    % Hough Transform to find straight lines (potential city layout)
    [H, T, R] = hough(edges);
    peaks = houghpeaks(H, 20, 'Threshold', ceil(0.3 * max(H(:))));
    lines = houghlines(edges, T, R, peaks, 'FillGap', 10, 'MinLength', 20);

    % Create a mask from the lines
    cityLines = false(size(grayImg));
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        cityLines = insertShape(uint8(cityLines), 'Line', [xy(1, :) xy(2, :)], 'Color', 'white', 'LineWidth', 3);
    end
    cityLines = im2bw(rgb2gray(cityLines));

    % Unnatural color detection
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;
    unnaturalColor = (abs(R - G) > 0.1) & (abs(G - B) > 0.1) & (abs(R - B) > 0.1);

    % Local contrast (entropy-based)
    entropyMap = entropyfilt(grayImg, true(9));
    highEntropy = mat2gray(entropyMap) > 0.7;

    % Combine all conditions
    cityMask = cityLines | (unnaturalColor & highEntropy);
    cityMask = cityMask & mask;
    cityMask = bwareaopen(cityMask, 50);
end

function labelMap = classifyRemainingPixels(rgbImg, mask)
    % Normalize and extract channels
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;
    gray = rgb2gray(rgbImg);
    edgeMap = edge(gray, 'Canny');
    edgeDensity = conv2(double(edgeMap), ones(5)/25, 'same');
    brightness = (R + G + B) / 3;
    maxCat = max(cat(3, R, G, B), [], 3);
    minCat = min(cat(3, R, G, B), [], 3);
    saturation = maxCat - minCat;


    % Initialize
    labelMap = zeros(size(R));

    % Add before existing forest/snow/city/rest logic:
    cityMaskFromFunc = detectCityBlocks(rgbImg, mask);
    
    % Refined Water detection using Cyan tint
    isCyanWater = (B > 0.4) & (G > 0.4) & (R < 0.3) & (B - R > 0.2);
    waterMask = isCyanWater & mask & ~cityMaskFromFunc;

    % --- 1. Forest ---
    %forestMask = (G > 0.25) & (R < 0.25) & ((R + G + B)/3 < 0.4) & mask;
    
    lowEdge = edgeDensity < 0.05;
    greenish = (G > R + 0.05) & (G > B + 0.05);
    lowBrightness = (R + G + B)/3 < 0.4;
    forestMask = greenish & lowBrightness & lowEdge & ~waterMask & ~cityMaskFromFunc & mask;

    %greenish = (G > R + 0.02) & (G > B + 0.02);
    %moderateBrightness = (R + G + B)/3 < 0.5;
    %forestMask = greenish & moderateBrightness & lowEdge & mask;


    % --- 2. River (placeholder, assume handled earlier) ---
    riverMask = false(size(R));  % Leave untouched here — filled elsewhere

    

    % --- City ---
    colorNeutral = abs(R - G) < 0.05 & abs(G - B) < 0.05;
    midBrightness = brightness > 0.35 & brightness < 0.85;
    highEdge = conv2(double(edgeMap), ones(9)/81, 'same') > 0.05;

    unnaturalColor = (R > 0.5 & G < 0.4 & B < 0.4) | ...
                     (B > 0.5 & G > 0.5 & R < 0.4) | ...
                     (R > 0.3 & G > 0.3 & B > 0.6);

    cityMask = (colorNeutral & highEdge & midBrightness | unnaturalColor) & ...
               mask & ~forestMask & ~waterMask & ~cityMaskFromFunc;

    % --- 3. Snow ---
    
    whiteish = (R > 0.75) & (G > 0.75) & (B > 0.75);
    lowEdge = edgeDensity < 0.02;

    snowMask = whiteish & lowEdge & mask & ~forestMask & ~cityMask & ~waterMask & ~cityMaskFromFunc;

    % --- 4. Sand ---

    sandLike = (R > 0.6) & (G < 0.55) & (B < 0.45);
    lowSat = saturation < 0.2;
    sandMask = sandLike & lowSat & brightness > 0.5 & ...
               mask & ~forestMask & ~cityMask & ~snowMask & ~waterMask & ~cityMaskFromFunc;

    % --- 4. Agriculture / Soil ---

    midBrightness = brightness > 0.35 & brightness < 0.75;
    lowSat = saturation < 0.25;
    highTexture = edgeDensity > 0.01;
    agricultureMask = midBrightness & lowSat & highTexture & mask;
    soilMask = agricultureMask & ~(forestMask | cityMask | snowMask | riverMask) & ~waterMask & ~cityMaskFromFunc & mask;

    % --- 6. Rest ---
    restMask = mask & ~(forestMask | snowMask | cityMask | riverMask | soilMask) & ~waterMask & ~cityMaskFromFunc;


    % --- Assign Labels ---

    % 1 = Forest, 2 = River, 3 = City, 4 = Rest, 5 = Snow
    labelMap(forestMask) = 1;
    labelMap(riverMask) = 2;  % Placeholder only
    labelMap(cityMaskFromFunc)   = 3;
    labelMap(restMask)   = 4;
    labelMap(snowMask)   = 5;  % Snow added directly here
    labelMap(waterMask) = 6;
    labelMap(soilMask) = 7;
    labelMap(sandMask) = 8;
end



