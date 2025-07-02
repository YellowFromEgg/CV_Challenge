imgPath = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg";
imgPath1 ="C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg";
imgPath2 = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg";



%% Method 1

evaluateLandCoverMethods(imgPath);

function evaluateLandCoverMethods(imgPath)
    % EVALUATELANDCOVERMETHODS
    % Displays original image and results of three land cover classification methods

    % --- Load image ---
    rgbImg = im2double(imread(imgPath));

    % Prepare figure
    figure('Name', 'Land Cover Classification Evaluation', 'NumberTitle', 'off');
    
    % --- 1. Original Image ---
    subplot(2,2,1);
    imshow(rgbImg);
    title('Original Image');

    % --- 2. findRegionsFromRGBImage ---
    subplot(2,2,2);
    labelImg1 = classify_findRegionsFromRGBImage(rgbImg);
    imshow(labelImg1);
    title('findRegionsFromRGBImage');

    % --- 3. analyzeRGBLandcover ---
    subplot(2,2,3);
    labelImg2 = classifyLandCoverEnhanced(rgbImg);
    imshow(labelImg2);
    title('analyzeRGBLandcover');

    % --- 4. classifyLandCoverColorRiver ---
    subplot(2,2,4);
    labelImg3 = classify_classifyLandCoverColorRiver(rgbImg);
    imshow(labelImg3);
    title('classifyLandCoverColorRiver');
end

function overlay = classify_findRegionsFromRGBImage(rgbImg)
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);
    vegMask = (G > R) & (G > B) & (G > 0.35);
    waterMask = (B > G) & (B > R) & (B > 0.35);
    landMask = ~(vegMask | waterMask);
    labelImg = zeros(size(R));
    labelImg(waterMask) = 1;
    labelImg(vegMask) = 2;
    labelImg(landMask) = 3;
    cmap = [0 0 1; 0 1 0; 0.6 0.3 0];
    overlay = labeloverlay(rgbImg, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
end

%not used
function overlay = classify_analyzeRGBLandcover(rgbImg)
    R = rgbImg(:,:,1); G = rgbImg(:,:,2); B = rgbImg(:,:,3);
    hsv = rgb2hsv(rgbImg); S = hsv(:,:,2); V = hsv(:,:,3);
    ExG = 2*G - R - B;
    vegMask = ExG > 0.05;
    waterMask = (S < 0.3) & (V > 0.4) & (B > R) & (B > G);
    urbanMask = (S < 0.25) & (V > 0.6) & abs(R-G) < 0.1 & abs(R-B) < 0.1;
    landMask = ~(vegMask | waterMask | urbanMask);
    labelImg = zeros(size(R));
    labelImg(waterMask) = 1;
    labelImg(vegMask) = 2;
    labelImg(urbanMask) = 3;
    labelImg(landMask) = 4;
    cmap = [0 0 1; 0 1 0; 1 0 0; 0.6 0.3 0];
    overlay = labeloverlay(rgbImg, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
end

function overlay = classify_classifyLandCoverColorRiver(rgbImg)
    R = rgbImg(:,:,1); G = rgbImg(:,:,2); B = rgbImg(:,:,3);
    h = size(R, 1); w = size(R, 2);
    darkGreen = (G > 0.25) & (R < 0.25) & ((R + G + B)/3 < 0.4);
    forestMask = darkGreen;
    riverMask = (G > R + 0.05) & (G > B + 0.05) & (G > 0.35);
    riverMask = imclose(riverMask, strel('disk', 1));
    riverMask = bwareaopen(riverMask, 50);
    cityMask = ((R + G + B)/3 > 0.3) & ~forestMask & ~riverMask;
    restMask = ~(forestMask | riverMask | cityMask);
    labelImg = zeros(h, w);
    labelImg(forestMask) = 1;
    labelImg(riverMask) = 2;
    labelImg(cityMask) = 3;
    labelImg(restMask) = 4;
    cmap = [0 1 0; 0 0 1; 0.5 0.5 0.5; 0.6 0.4 0.2];
    overlay = labeloverlay(rgbImg, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
end


function overlay = classifyLandCoverEnhanced(rgbImg)
    % Convert to double for processing
    rgbImg = im2double(rgbImg);
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);
    [h, w, ~] = size(rgbImg);
    
    %% Convert to HSV color space
    hsvImg = rgb2hsv(rgbImg);
    H = hsvImg(:,:,1); S = hsvImg(:,:,2); V = hsvImg(:,:,3);

    %% Approximate NDVI (from R and G only)
    ndvi = (G - R) ./ (G + R + eps);

    %% Grayscale and texture (entropy)
    grayImg = rgb2gray(rgbImg);
    entropyMap = entropyfilt(grayImg);

    %% Forest detection using NDVI and texture
    forestMask = (ndvi > 0.1) & (entropyMap > 3);

    %% River detection using HSV (blue hue range)
    riverMask = (H > 0.55 & H < 0.67) & (S > 0.25) & (V > 0.2);
    riverMask = imclose(riverMask, strel('disk', 2));
    riverMask = bwareaopen(riverMask, 50);

    %% City/urban detection: brighter, less green, low texture
    avgBrightness = (R + G + B) / 3;
    cityMask = (avgBrightness > 0.35) & ~forestMask & ~riverMask & (entropyMap < 3);

    %% Rest (uncategorized)
    restMask = ~(forestMask | riverMask | cityMask);

    %% Assign labels
    labelImg = zeros(h, w);
    labelImg(forestMask) = 1;  % Green
    labelImg(riverMask) = 2;   % Blue
    labelImg(cityMask) = 3;    % Gray
    labelImg(restMask) = 4;    % Brown

    %% Visualization colormap
    cmap = [0 1 0;         % Forest - Green
            0 0 1;         % River - Blue
            0.5 0.5 0.5;   % City - Gray
            0.6 0.4 0.2];  % Rest - Brown

    %% Overlay output
    overlay = labeloverlay(rgbImg, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
end

%% Method 2: kmeans -> bad

rgb = im2double(imread(imgPath));
[regionMask] = kmeansColorRegions(rgb, 6);
showRegionBoundaries(regionMask);

classMap = classifyRegionsByAppearance(rgb, regionMask);
showRegionClassification(rgb, classMap);

% --- K-means based Region Segmentation ---
function regionMask = kmeansColorRegions(rgb, k)
    lab = rgb2lab(imgaussfilt(rgb, 1));
    ab = reshape(lab(:,:,2:3), [], 2);
    ab = normalize(ab);
    warning('off', 'stats:kmeans:FailedToConverge');
    pixelLabels = kmeans(ab, k, 'Replicates', 5, 'MaxIter', 300, 'Start', 'plus');
    warning('on', 'stats:kmeans:FailedToConverge');
    labelImg = reshape(pixelLabels, size(rgb,1), size(rgb,2));
    regionMask = medfilt2(labelImg, [5 5]);
end

% --- Show Region Boundaries ---
function showRegionBoundaries(regionMask)
    figure;
    Lrgb = label2rgb(regionMask, 'jet', 'k', 'shuffle');
    imshow(Lrgb);
    title('Region Detection via Color + Connectivity');
end

% --- Region-Based Land Cover Classification ---
function classMap = classifyRegionsByAppearance(rgb, regionMask)
    numRegions = max(regionMask(:));
    [R, G, B] = deal(rgb(:,:,1), rgb(:,:,2), rgb(:,:,3));
    classMap = zeros(size(regionMask)); % Initialize

    for i = 1:numRegions
        idx = find(regionMask == i);
        meanR = mean(R(idx));
        meanG = mean(G(idx));
        meanB = mean(B(idx));
        meanRGB = [meanR, meanG, meanB];

        hsvColor = rgb2hsv(reshape(meanRGB, [1 1 3]));
        hue = hsvColor(1); sat = hsvColor(2); val = hsvColor(3);

        props = regionprops(regionMask == i, 'Eccentricity', 'Solidity', 'Area');

        if isempty(props)
            label = 7; % Fallback if no region found
        elseif all(meanRGB > 0.85) && sat < 0.15
            label = 2; % Snow
        elseif meanB > meanG && meanB > meanR && val < 0.5
            label = 3; % Water
        elseif meanG > meanR + 0.05 && meanG > meanB + 0.05
            label = 1; % Forest
        elseif props(1).Eccentricity > 0.95 && props(1).Area > 100
            label = 4; % River
        elseif sat < 0.3 && val > 0.6
            label = 5; % City
        elseif meanG > 0.4 && meanR > 0.3 && sat > 0.2
            label = 6; % Agriculture
        else
            label = 7; % Other / Unclassified
        end

        classMap(idx) = label;
    end
end

% --- Show Region Classification Overlay ---
function showRegionClassification(rgb, classMap)
    cmap = [0 1 0;        % 1 - Forest (green)
            1 1 1;        % 2 - Snow (white)
            0 0 1;        % 3 - Water (blue)
            0 1 1;        % 4 - River (cyan)
            0.5 0.5 0.5;  % 5 - City (gray)
            1 1 0;        % 6 - Agriculture (yellow)
            0.6 0.4 0.2]; % 7 - Other (brown)

    overlay = labeloverlay(rgb, classMap, 'Colormap', cmap, 'Transparency', 0.4);
    figure;
    imshow(overlay);
    title('Region-Based Land Cover Classification');
end
%% Method 3

rgb = im2double(imread(imgPath));
labelImg = detectRegionsByColorRules(rgb);
showColorRegionResultSideBySide(rgb, labelImg);

function labelImg = detectRegionsByColorRules(rgb)
    rgb = im2double(rgb);
    R = rgb(:,:,1);
    G = rgb(:,:,2);
    B = rgb(:,:,3);
    hsv = rgb2hsv(rgb);
    S = hsv(:,:,2);

    labelImg = zeros(size(R)); % 0 = unlabeled

    % 1 = Vegetation
    vegMask = (G > R) & (G > B) & (G > 0.35);
    labelImg(vegMask) = 1;

    % 2 = Water (Blue)
    waterBlue = (B > G) & (B > R) & (B > 0.35);
    labelImg(waterBlue) = 2;

    % 3 = Water (Cyan/White River)
    waterCyan = (B > 0.3) & (G > 0.3) & (R < 0.25) & (abs(B - G) < 0.1);
    labelImg(waterCyan) = 3;

    % 4 = City / Urban
    cityMask = ((R + G + B)/3 > 0.4) & (S < 0.25);
    labelImg(cityMask) = 4;

    % 5 = Snow
    snowMask = (R > 0.85) & (G > 0.85) & (B > 0.85) & (S < 0.1);
    labelImg(snowMask) = 5;

    % 6 = Agriculture
    agriMask = (G > 0.35) & (R > 0.3) & (S > 0.25) & ~vegMask;
    labelImg(agriMask) = 6;
end

function showColorRegionResultSideBySide(rgb, labelImg)
    % Define colormap for 6 classes
    cmap = [
        0 1 0;      % 1 - Vegetation (green)
        0 0 1;      % 2 - Water (blue)
        0 1 1;      % 3 - River (cyan)
        0.5 0.5 0.5;% 4 - City (gray)
        1 1 1;      % 5 - Snow (white)
        1 1 0       % 6 - Agriculture (yellow)
    ];

    % Plot side-by-side
    figure('Name', 'Color-Based Region Detection', 'NumberTitle', 'off');

    % Left: classified regions
    subplot(1, 2, 1);
    imagesc(labelImg);
    colormap(gca, cmap);
    axis image off;
    title('Classified Regions');

    % Right: original RGB
    subplot(1, 2, 2);
    imshow(rgb);
    title('Original Image');
end

%% 
labelImg = detectColorRegionsOnly(rgb);
showColorRegionGroups(rgb, labelImg);
function labelImg = detectColorRegionsOnly(rgb)
    rgb = im2double(rgb);
    R = rgb(:,:,1);
    G = rgb(:,:,2);
    B = rgb(:,:,3);

    labelImg = zeros(size(R));

    % Define pure color-based regions
    % 1 = Green (Vegetation-like)
    greenish = (G > R + 0.05) & (G > B + 0.05);
    labelImg(greenish) = 1;

    % 2 = Blue (Deep water)
    bluish = (B > R + 0.05) & (B > G + 0.05);
    labelImg(bluish) = 2;

    % 3 = Cyan (River-like)
    cyan = (B > 0.35) & (G > 0.35) & (R < 0.3) & (abs(B - G) < 0.1);
    labelImg(cyan) = 3;

    % 4 = Brown/Orange (Soil, Dry areas, Roofs)
    brownish = (R > G) & (G > B) & (R > 0.4) & (R < 0.75);
    labelImg(brownish) = 4;

    % 5 = Gray (City, Roads)
    grayish = (abs(R - G) < 0.05) & (abs(G - B) < 0.05) & (R > 0.2) & (R < 0.8);
    labelImg(grayish) = 5;

    % 6 = White (Snow, clouds)
    white = (R > 0.85) & (G > 0.85) & (B > 0.85);
    labelImg(white) = 6;
end

function showColorRegionGroups(rgb, labelImg)
    cmap = [
        0 1 0;      % 1 - Green
        0 0 1;      % 2 - Blue
        0 1 1;      % 3 - Cyan
        0.8 0.4 0;  % 4 - Brown/Orange
        0.5 0.5 0.5;% 5 - Gray
        1 1 1       % 6 - White
    ];

    figure('Name', 'Stage 1: Pure Color-Based Region Detection');
    subplot(1,2,1);
    imagesc(labelImg);
    colormap(gca, cmap);
    axis image off;
    title('Color Group Map');

    subplot(1,2,2);
    imshow(rgb);
    title('Original Image');
end

%% Method: grayscale


segmentRegionsBySmoothedGradient(rgb, 0.1);  % test 0.02–0.04

function segmentRegionsBySmoothedGradient(rgb, gradientThreshold)
    % 1. Convert to grayscale
    gray = im2gray(im2double(rgb));

    % 2. Smooth image
    smoothGray = imgaussfilt(gray, 2);  % suppress noise, preserve structure

    % 3. Compute gradient magnitude
    [Gmag, ~] = imgradient(smoothGray);

    % 4. Low-gradient thresholding
    flatMask = Gmag < gradientThreshold;

    % 5. Morphological cleanup
    flatMask = imclose(flatMask, strel('disk', 2));
    flatMask = bwareaopen(flatMask, 100);

    % 6. Label regions
    CC = bwconncomp(flatMask, 8);
    labeled = labelmatrix(CC);

    % 7. Assign color per region
    colorRegions = label2rgb(labeled, 'jet', 'k', 'shuffle');

    % 8. Plot results: grayscale | labeled regions | original
    figure('Name', 'Gradient-Based Region Segmentation');

    % Left: grayscale
    subplot(1,3,1);
    imshow(gray);
    title('Grayscale Image');

    % Middle: region map
    subplot(1,3,2);
    imshow(colorRegions);
    title(sprintf('Low-Gradient Regions (\\nabla < %.3f)', gradientThreshold));

    % Right: original
    subplot(1,3,3);
    imshow(rgb);
    title('Original RGB Image');
end
