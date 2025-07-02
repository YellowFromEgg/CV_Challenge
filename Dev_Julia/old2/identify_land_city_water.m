% Load image
img = imread("C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Dev_Keno\Theresienwiese\2024_05.png");  % Replace with your image filename
img = imread("C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg");
% Call segmentation function
%mask = segment_land_city_water(img);

imgPath = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg";
imgPath1 ="C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg";
imgPath2 = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg";
%findRegionsFromRGBImage(imgPath);
%findRainforestRegions(imgPath);
%analyzeRGBLandcover(imgPath);

%findRegionsWithShapeAnalysis(imgPath)
%findRegionsWithImprovedRiverDetection(imgPath);
%classifyLandCoverWithRiverForestCityLand(imgPath)
%classifyLandCoverFinal(imgPath)
classifyLandCoverColorRiver(imgPath)
classifyLandCoverColorRiver(imgPath2)

function mask = segment_land_city_water(img)
% INPUT:  img - RGB image (normalized or uint8)
% OUTPUT: mask - labeled image: 1 = Land, 2 = City, 3 = Water

% Convert to double if needed
if ~isfloat(img)
    img = im2double(img);
end

% Convert to HSV
hsvImg = rgb2hsv(img);
H = hsvImg(:,:,1);
S = hsvImg(:,:,2);
V = hsvImg(:,:,3);

% Convert to grayscale for texture analysis
grayImg = rgb2gray(img);

% Compute local entropy (texture)
entropyMap = entropyfilt(grayImg);
entropyNorm = mat2gray(entropyMap); % Normalize for thresholding

% Initialize mask
mask = zeros(size(H));

% ----------------------------
% CLASS 1: LAND (vegetation)
% ----------------------------
% - Green hues
% - Moderate texture
landColor = (H > 0.20 & H < 0.45) & (S > 0.25) & (V > 0.2);
landTexture = entropyNorm > 0.3;  % has structure
landMask = landColor & landTexture;

% ----------------------------
% CLASS 2: CITY (urban areas)
% ----------------------------
% - Low saturation (grayish)
% - High brightness OR high texture
cityColor = ((S < 0.25 & V > 0.35) | (V > 0.65 & S < 0.35));
cityTexture = entropyNorm > 0.4;
cityMask = cityColor & cityTexture & ~landMask;

% ----------------------------
% CLASS 3: WATER (deep blue, flat texture)
% ----------------------------
% - Dark blue hue
% - Low entropy (flat region)
waterColor = (H > 0.55 & H < 0.65) & (S > 0.4) & (V < 0.5);
lowTexture = entropyNorm < 0.25;
waterMask = waterColor & lowTexture;

% Assign labels
mask(landMask) = 1;
mask(cityMask) = 2;
mask(waterMask) = 3;

% Visualization
figure;
imagesc(mask);
colormap([0.2 0.8 0.2; 0.5 0.5 0.5; 0.3 0.5 1]); % green = land, gray = city, blue = water
colorbar('Ticks', [1,2,3], 'TickLabels', {'Land','City','Water'});
title('Segmented Land / City / Water');
end

function findRegionsFromRGBImage(imgPath)
% FINDREGIONSFROMRGBIMAGE Segments vegetation, water, and land from an RGB image (e.g., PNG/JPG).
% Uses RGB color thresholds (heuristics), NOT spectral indices.

    % Read and normalize image
    rgbImg = im2double(imread(imgPath));
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);

    % Simple rules based on color intensities
    vegMask = (G > R) & (G > B) & (G > 0.35); % Vegetation: green dominant
    waterMask = (B > G) & (B > R) & (B > 0.35); % Water: blue dominant
    landMask = ~(vegMask | waterMask); % The rest = assumed land

    % Create labeled image: 0=background, 1=water, 2=vegetation, 3=land
    labelImg = zeros(size(R));
    labelImg(waterMask) = 1;
    labelImg(vegMask) = 2;
    labelImg(landMask) = 3;

    % Overlay colors: [black; blue; green; brown]
    overlay = labeloverlay(rgbImg, labelImg, ...
        'Colormap', [0 0 1; 0 1 0; 0.6 0.3 0], ...
        'Transparency', 0.4);

    % Show result
    figure
    imshow(overlay)
    title("Estimated Water, Vegetation, and Land Regions")

    % Legend
    annotation("textbox", [0.75 0.55 0.2 0.05], "String", "Water", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.48 0.2 0.05], "String", "Vegetation", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0]);
    annotation("textbox", [0.75 0.41 0.2 0.05], "String", "Land", ...
        "BackgroundColor", [0.6 0.3 0], "Color", [1 1 1]);
end

function findRainforestRegions(imgPath)
% FINDRAINFORESTREGIONS - Improved water & vegetation detection from RGB image
% Works on PNG/JPEG satellite images (e.g., rainforest + river)

    % Load and normalize image
    rgb = im2double(imread(imgPath));
    R = rgb(:,:,1);
    G = rgb(:,:,2);
    B = rgb(:,:,3);

    % --- Improved Vegetation Detection ---
    % Excess Green Index (ExG) works well for forests
    ExG = 2 * G - R - B;
    vegMask = ExG > 0.05; % Tune threshold if needed

    % --- Improved Water Detection ---
    % Water tends to be darker, more blue than green
    waterIndex = B - G;
    darkMask = (R + G + B)/3 < 0.4; % Optional: water tends to be darker
    waterMask = (waterIndex > 0.03) & darkMask;

    % Land = anything else
    landMask = ~(vegMask | waterMask);

    % --- Create label image ---
    labelImg = zeros(size(R));
    labelImg(waterMask) = 1;    % Water = 1
    labelImg(vegMask) = 2;      % Vegetation = 2
    labelImg(landMask) = 3;     % Land = 3

    % Overlay colors
    cmap = [0 0 1;     % Water - Blue
            0 1 0;     % Vegetation - Green
            0.6 0.3 0];% Land - Brown

    overlay = labeloverlay(rgb, labelImg, 'Colormap', cmap, 'Transparency', 0.4);

    % --- Display ---
    figure
    imshow(overlay)
    title('Refined Detection: Water, Vegetation, and Land')

    % Legend
    annotation("textbox", [0.75 0.55 0.2 0.05], "String", "Water", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1], "FontSize", 12);
    annotation("textbox", [0.75 0.48 0.2 0.05], "String", "Vegetation", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0], "FontSize", 12);
    annotation("textbox", [0.75 0.41 0.2 0.05], "String", "Land", ...
        "BackgroundColor", [0.6 0.3 0], "Color", [1 1 1], "FontSize", 12);
end

function analyzeRGBLandcover(imgPath)
% ANALYZERGBLANDCOVER - Detects water, vegetation, land, and urban areas in RGB satellite images.

    % Load and normalize image
    rgb = im2double(imread(imgPath));
    R = rgb(:,:,1);
    G = rgb(:,:,2);
    B = rgb(:,:,3);

    % Convert to HSV for brightness/saturation analysis
    hsv = rgb2hsv(rgb);
    H = hsv(:,:,1);
    S = hsv(:,:,2);
    V = hsv(:,:,3);

    % --- Vegetation: Excess Green Index ---
    ExG = 2*G - R - B;
    vegMask = ExG > 0.05;

    % --- Water: Low saturation, moderately bright ---
    waterMask = (S < 0.3) & (V > 0.4) & (B > R) & (B > G);

    % --- Urban: Low saturation, high brightness, neutral color
    urbanMask = (S < 0.25) & (V > 0.6) & abs(R-G) < 0.1 & abs(R-B) < 0.1;

    % --- Land: Remaining areas
    landMask = ~(vegMask | waterMask | urbanMask);

    % --- Label Image ---
    labelImg = zeros(size(R));
    labelImg(waterMask) = 1;   % Water
    labelImg(vegMask) = 2;     % Vegetation
    labelImg(urbanMask) = 3;   % Urban
    labelImg(landMask) = 4;    % Land

    % --- Overlay and Display ---
    cmap = [0 0 1;       % Blue - Water
            0 1 0;       % Green - Vegetation
            1 0 0;       % Red - Urban
            0.6 0.3 0];  % Brown - Land

    overlay = labeloverlay(rgb, labelImg, 'Colormap', cmap, 'Transparency', 0.4);

    figure
    imshow(overlay)
    title('Detected Regions: Water, Vegetation, Urban, Land')

    % --- Legend ---
    annotation("textbox", [0.75 0.6 0.2 0.05], "String", "Water", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1], "FontSize", 12);
    annotation("textbox", [0.75 0.53 0.2 0.05], "String", "Vegetation", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0], "FontSize", 12);
    annotation("textbox", [0.75 0.46 0.2 0.05], "String", "Urban", ...
        "BackgroundColor", [1 0 0], "Color", [1 1 1], "FontSize", 12);
    annotation("textbox", [0.75 0.39 0.2 0.05], "String", "Land", ...
        "BackgroundColor", [0.6 0.3 0], "Color", [1 1 1], "FontSize", 12);
end

function segmentAndClassifyRegions1(imgPath)
% SEGMENTANDCLASSIFYREGIONS
% Segments an RGB image and classifies each region as:
% River (elongated), Vegetation (greenish), or City (other)

    % === Step 1: Read and normalize image ===
    rgb = im2double(imread(imgPath));
    [height, width, ~] = size(rgb);

    red = rgb(:,:,1);
    green = rgb(:,:,2);
    blue = rgb(:,:,3);

    % Convert to LAB for perceptual clustering
    labImg = rgb2lab(rgb);
    ab = reshape(labImg(:,:,2:3), [], 2);

    % === Step 2: K-means clustering on color ===
    nColors = 5; % adjustable
    pixel_labels = imsegkmeans(single(ab), nColors, 'NumAttempts', 3);
    %pixel_labels = reshape(pixel_labels, height, width);

    % === Step 3: Initialize label image ===
    labelImg = zeros(height, width);

    % === Step 4: Analyze each cluster separately ===
    for classID = 1:nColors
        mask = pixel_labels == classID;

        % Find spatially connected regions within this cluster
        cc = bwconncomp(mask);
        regions = regionprops(cc, 'PixelIdxList', 'BoundingBox');

        for j = 1:numel(regions)
            mask = pixel_labels == classID;
            cc = bwconncomp(mask);
            regions = regionprops(cc, 'PixelIdxList', 'BoundingBox');
            
            % Find linear indices of the full image for this classID mask
            maskLinearIndices = find(mask);
            
            for j = 1:numel(regions)
                % These are indices into the mask, so we convert to full image indices
                idxInMask = regions(j).PixelIdxList;
                fullIdx = maskLinearIndices(idxInMask);  % ✅ FIX
            
                % Get mean RGB
                meanR = mean(red(fullIdx));
                meanG = mean(green(fullIdx));
                meanB = mean(blue(fullIdx));
                meanRGB = [meanR, meanG, meanB];
            end

            % Convert to HSV for hue analysis
            hsvColor = rgb2hsv(reshape(meanRGB, 1, 1, 3));
            hue = hsvColor(1);

            % Bounding box shape
            bbox = regions(j).BoundingBox;
            aspectRatio = max(bbox(3)/bbox(4), bbox(4)/bbox(3));

            % === Classification logic ===
            if hue > 0.2 && hue < 0.45 && meanG > meanR && meanG > meanB
                label = 2; % Vegetation
            elseif aspectRatio > 4
                label = 1; % River
            else
                label = 3; % City
            end

            labelImg(idx) = label;
        end
    end

    % === Step 5: Visualization ===
    cmap = [0 0 1;       % Blue - River
            0 1 0;       % Green - Vegetation
            1 0 0];      % Red - City

    overlay = labeloverlay(rgb, labelImg, 'Colormap', cmap, 'Transparency', 0.4);

    figure
    imshow(overlay)
    title('Region-Based Classification: River, Vegetation, City')

    % === Step 6: Legend ===
    annotation("textbox", [0.75 0.6 0.2 0.05], "String", "River", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1], "FontSize", 12);
    annotation("textbox", [0.75 0.53 0.2 0.05], "String", "Vegetation", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0], "FontSize", 12);
    annotation("textbox", [0.75 0.46 0.2 0.05], "String", "City", ...
        "BackgroundColor", [1 0 0], "Color", [1 1 1], "FontSize", 12);
end

function segmentAndClassifyRegions(imgPath)
% SEGMENTANDCLASSIFYREGIONS
% Segments and classifies RGB image regions as River, Vegetation, or City
% using color and shape descriptors.

    % === Step 1: Load image ===
    rgb = im2double(imread(imgPath));
    [height, width, ~] = size(rgb);
    red = rgb(:,:,1);
    green = rgb(:,:,2);
    blue = rgb(:,:,3);

    % === Step 2: Segment with K-means in LAB color space ===
    lab = rgb2lab(rgb);
    ab = reshape(lab(:,:,2:3), [], 2);
    nColors = 5;
    pixelLabels = imsegkmeans(single(ab), nColors, 'NumAttempts', 3);

    % === Step 3: Output label map ===
    labelImg = zeros(height, width);

    % === Step 4: Process each color cluster ===
    for k = 1:nColors
        clusterMask = (pixelLabels == k);  % binary mask of this cluster

        % Get connected components with absolute indices
        cc = bwconncomp(clusterMask);  % these indices are relative to full image
        stats = regionprops(cc, 'PixelIdxList', 'BoundingBox');

        for r = 1:numel(stats)
            idx = stats(r).PixelIdxList;

            % Validate: make sure idx are all within image bounds
            if max(idx) > numel(red)
                warning("Skipping region with invalid indices.");
                continue
            end

            % Mean RGB
            meanR = mean(red(idx));
            meanG = mean(green(idx));
            meanB = mean(blue(idx));
            meanRGB = [meanR, meanG, meanB];

            % Convert to HSV
            hsvColor = rgb2hsv(reshape(meanRGB, [1 1 3]));
            hue = hsvColor(1);

            % Shape aspect ratio
            bbox = stats(r).BoundingBox;
            aspectRatio = max(bbox(3)/bbox(4), bbox(4)/bbox(3));

            % === Classification ===
            if hue > 0.2 && hue < 0.45 && meanG > meanR && meanG > meanB
                label = 2; % Vegetation
            elseif aspectRatio > 4
                label = 1; % River
            else
                label = 3; % City
            end

            labelImg(idx) = label;
        end
    end

    % === Step 5: Overlay and display ===
    cmap = [0 0 1;   % River - Blue
            0 1 0;   % Vegetation - Green
            1 0 0];  % City - Red

    overlay = labeloverlay(rgb, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
    figure
    imshow(overlay)
    title('Region-Based Classification: River, Vegetation, City')

    % === Step 6: Legend ===
    annotation("textbox", [0.75 0.6 0.2 0.05], "String", "River", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1], "FontSize", 12);
    annotation("textbox", [0.75 0.53 0.2 0.05], "String", "Vegetation", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0], "FontSize", 12);
    annotation("textbox", [0.75 0.46 0.2 0.05], "String", "City", ...
        "BackgroundColor", [1 0 0], "Color", [1 1 1], "FontSize", 12);
end

function findRegionsWithShapeAnalysis(imgPath)
% FINDREGIONSWITHSHAPEANALYSIS Segments and labels vegetation, city, and river using color and shape.
% This version enhances the previous method by analyzing region shapes.

    % Read and normalize image
    rgbImg = im2double(imread(imgPath));
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);

    % Step 1: Initial vegetation detection based on green dominance
    vegMask = (G > R) & (G > B) & (G > 0.35);

    % Step 2: Extract all non-vegetation regions
    nonVegMask = ~vegMask;

    % Step 3: Label connected regions in non-vegetation areas
    CC = bwconncomp(nonVegMask);
    labeled = zeros(size(R));
    regionStats = regionprops(CC, 'Area', 'BoundingBox', 'PixelIdxList');

    for i = 1:length(regionStats)
        region = regionStats(i);
        bb = region.BoundingBox;
        aspectRatio = max(bb(3)/bb(4), bb(4)/bb(3));  % Long axis / short axis

        % Heuristic: river-like if long and thin
        if aspectRatio > 3 && region.Area > 100  % Adjust thresholds as needed
            labeled(region.PixelIdxList) = 1; % River
        else
            labeled(region.PixelIdxList) = 2; % City
        end
    end

    % Step 4: Vegetation mask labeled separately
    labeled(vegMask) = 3;

    % Final overlay colors: [blue=river, gray=city, green=vegetation]
    overlay = labeloverlay(rgbImg, labeled, ...
        'Colormap', [0 0 1; 0.5 0.5 0.5; 0 1 0], ...
        'Transparency', 0.4);

    % Show result
    figure
    imshow(overlay)
    title("Regions: River (blue), City (gray), Vegetation (green)")

    % Legend
    annotation("textbox", [0.75 0.55 0.2 0.05], "String", "River", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.48 0.2 0.05], "String", "City", ...
        "BackgroundColor", [0.5 0.5 0.5], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.41 0.2 0.05], "String", "Vegetation", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0]);
end

function findRegionsWithImprovedRiverDetection(imgPath)
% FINDREGIONSWITHIMPROVEDRIVERDETECTION
% Segments vegetation, city, and river using color, morphology, and shape.
% Improves river detection using skeletonization of cyan-colored regions.

    % Read and normalize image
    rgbImg = im2double(imread(imgPath));
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);

    % --- 1. Vegetation mask (green dominant) ---
    vegMask = (G > R) & (G > B) & (G > 0.35);

    % --- 2. River candidate mask using cyan color hint ---
    riverHint = (B > 0.2) & (G > 0.2) & (B > R) & ((B - R) > 0.05);

    % Morphological cleaning: connect fragmented river pieces
    riverHint = imclose(riverHint, strel('disk', 3));
    riverHint = bwareaopen(riverHint, 100);  % remove small blobs

    % Skeletonization to find long, thin structures
    riverSkel = bwskel(riverHint, 'MinBranchLength', 50);
    riverMask = imdilate(riverSkel, strel('disk', 1));  % make visible

    % Optional: Refine river mask with original hint (safety check)
    riverMask = riverMask & riverHint;

    % --- 3. City mask = non-vegetation, non-river ---
    cityMask = ~(vegMask | riverMask);

    % --- 4. Final labeling ---
    % Label matrix: 0 = background, 1 = river, 2 = city, 3 = vegetation
    labelImg = zeros(size(R));
    labelImg(riverMask) = 1;
    labelImg(cityMask) = 2;
    labelImg(vegMask) = 3;

    % --- 5. Visualize ---
    overlay = labeloverlay(rgbImg, labelImg, ...
        'Colormap', [0 0 1; 0.5 0.5 0.5; 0 1 0], ...
        'Transparency', 0.4);

    figure
    imshow(overlay)
    title("Regions: River (blue), City (gray), Vegetation (green)")

    % Legend
    annotation("textbox", [0.75 0.55 0.2 0.05], "String", "River", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.48 0.2 0.05], "String", "City", ...
        "BackgroundColor", [0.5 0.5 0.5], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.41 0.2 0.05], "String", "Vegetation", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0]);
end

function classifyLandCoverWithRiverForestCityLand(imgPath)
% CLASSIFYLANDCOVERWITHRIVERFORESTCITYLAND
% Classifies each pixel as Forest, River, City, or Land using color + shape heuristics

    % --- Load and normalize image ---
    rgbImg = im2double(imread(imgPath));
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);
    [h, w, ~] = size(rgbImg);

    % --- Forest detection: includes dark forests too ---
brightGreen = (G > R + 0.05) & (G > B + 0.05) & (G > 0.35);
darkGreen = (G > 0.25) & (R < 0.25) & ((R + G + B)/3 < 0.4);
forestMask = brightGreen | darkGreen;

% --- River detection with better cyan targeting ---
cyanMask = (B > 0.35) & (G > 0.35) & (R < 0.25) & ...
           (abs(B - G) < 0.1) & (B - R > 0.1);
cyanClean = imclose(cyanMask, strel('disk', 2));
cyanClean = bwareaopen(cyanClean, 100);

riverMask = false(h, w);
stats = regionprops(cyanClean, 'PixelIdxList', 'Eccentricity', ...
                    'MajorAxisLength', 'MinorAxisLength', 'Solidity', 'Area');

for i = 1:numel(stats)
    region = stats(i);
    aspectRatio = region.MajorAxisLength / max(region.MinorAxisLength, 1e-5);

    if region.Eccentricity > 0.8 && ...
       aspectRatio > 3 && ...
       region.Solidity < 0.9 && ...
       region.Area > 100
        riverMask(region.PixelIdxList) = true;
    end
end

% --- City: remaining bright non-green non-cyan areas ---
nonGreen = ~(forestMask | cyanMask);
brightEnough = (R + G + B)/3 > 0.3;
cityMask = brightEnough & nonGreen & ~riverMask;

% --- Land: everything else ---
landMask = ~(forestMask | riverMask | cityMask);

    % --- Final label image ---
    % Label values: 1=Forest, 2=River, 3=City, 4=Land
    labelImg = zeros(h, w);
    labelImg(forestMask) = 1;
    labelImg(riverMask) = 2;
    labelImg(cityMask) = 3;
    labelImg(landMask) = 4;

    % --- Visualization ---
    cmap = [0 1 0;       % Forest - green
            0 0 1;       % River - blue
            0.5 0.5 0.5; % City - gray
            0.6 0.4 0.2];% Land - brownish

    overlay = labeloverlay(rgbImg, labelImg, ...
        'Colormap', cmap, ...
        'Transparency', 0.4);

    figure
    imshow(overlay)
    title("Land Cover Classification: Forest (green), River (blue), City (gray), Land (brown)")

    % --- Legend ---
    annotation("textbox", [0.75 0.61 0.2 0.05], "String", "Forest", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0]);
    annotation("textbox", [0.75 0.54 0.2 0.05], "String", "River", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.47 0.2 0.05], "String", "City", ...
        "BackgroundColor", [0.5 0.5 0.5], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.40 0.2 0.05], "String", "Land", ...
        "BackgroundColor", [0.6 0.4 0.2], "Color", [1 1 1]);

end

function classifyLandCoverFinal(imgPath)
% CLASSIFYLANDCOVERFINAL
% Classifies an RGB image into Forest, River, City, and Unlabeled based on color and shape.

    % --- Load and normalize image ---
    rgbImg = im2double(imread(imgPath));
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);
    [h, w, ~] = size(rgbImg);

    % --- 1. Forest Detection (bright + dark green) ---
    brightGreen = (G > R + 0.05) & (G > B + 0.05) & (G > 0.35);
    darkGreen = (G > 0.2) & (R < 0.2) & ((R + G + B)/3 < 0.35);
    forestMask = brightGreen | darkGreen;

    % --- 2. City Detection (non-green, bright) ---
    cityMask = ((R + G + B)/3 > 0.3) & ...
               ~forestMask;

    % --- 3. River Detection (shape-based only) ---
    % Create candidate mask: non-forest, non-city, possibly blue-tinted
    candidate = ~forestMask & ~cityMask & (B > G | B > R);

    % Morphological cleanup
    candidate = imclose(candidate, strel('disk', 2));
    candidate = bwareaopen(candidate, 100);

    % Region shape analysis
    riverMask = false(h, w);
    stats = regionprops(candidate, 'PixelIdxList', 'Eccentricity', ...
        'MajorAxisLength', 'MinorAxisLength', 'Solidity', 'Area');

    for i = 1:numel(stats)
        region = stats(i);
        aspectRatio = region.MajorAxisLength / max(region.MinorAxisLength, 1e-5);

        if region.Eccentricity > 0.95 && ...
           aspectRatio > 5 && ...
           region.Solidity < 0.8 && ...
           region.Area > 150
            riverMask(region.PixelIdxList) = true;
        end
    end

    % --- 4. Unlabeled (rest) ---
    restMask = ~(forestMask | riverMask | cityMask);

    % --- 5. Create label image ---
    % Labels: 1=Forest, 2=River, 3=City, 4=Unlabeled
    labelImg = zeros(h, w);
    labelImg(forestMask) = 1;
    labelImg(riverMask) = 2;
    labelImg(cityMask) = 3;
    labelImg(restMask) = 4;

    % --- 6. Visualization ---
    cmap = [0 1 0;       % Forest - green
            0 0 1;       % River - blue
            0.5 0.5 0.5; % City - gray
            0.6 0.4 0.2];% Unlabeled - brown

    overlay = labeloverlay(rgbImg, labelImg, ...
        'Colormap', cmap, 'Transparency', 0.4);

    figure
    imshow(overlay)
    title("Land Cover Classification: Forest (green), River (blue), City (gray), Unlabeled (brown)")

    % --- 7. Legend ---
    annotation("textbox", [0.75 0.60 0.2 0.05], "String", "Forest", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0]);
    annotation("textbox", [0.75 0.53 0.2 0.05], "String", "River", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.46 0.2 0.05], "String", "City", ...
        "BackgroundColor", [0.5 0.5 0.5], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.39 0.2 0.05], "String", "Unlabeled", ...
        "BackgroundColor", [0.6 0.4 0.2], "Color", [1 1 1]);

end

function classifyLandCoverColorRiver(imgPath)
% CLASSIFYLANDCOVERCOLORRIVER
% Classifies an RGB image into Forest, River (white/cyan), City, and Unlabeled
% based on refined color thresholds (no shape logic).

    % --- Load and normalize image ---
    rgbImg = im2double(imread(imgPath));
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);
    [h, w, ~] = size(rgbImg);

    % --- 1. Forest Detection (bright + dark green) ---

    
    darkGreen = (G > 0.25) & (R < 0.25) & ((R + G + B)/3 < 0.4);
    forestMask = darkGreen;

    % --- 2. River Detection (color-based: white/cyan/blue tones) ---
    riverMask = (G > R + 0.05) & (G > B + 0.05) & (G > 0.35);

    % Morphological cleanup (optional)
    riverMask = imclose(riverMask, strel('disk', 1));
    riverMask = bwareaopen(riverMask, 50);

    % --- 3. City Detection (non-green, bright areas) ---
    cityMask = ((R + G + B)/3 > 0.3) & ...
               ~forestMask & ~riverMask;

    % --- 4. Unlabeled (rest) ---
    restMask = ~(forestMask | riverMask | cityMask);

    % --- 5. Label Image ---
    % Labels: 1 = Forest, 2 = River, 3 = City, 4 = Unlabeled
    labelImg = zeros(h, w);
    labelImg(forestMask) = 1;
    labelImg(riverMask) = 2;
    labelImg(cityMask) = 3;
    labelImg(restMask) = 4;

    % --- 6. Visualization ---
    cmap = [0 1 0;       % Forest - green
            0 0 1;       % River - blue
            0.5 0.5 0.5; % City - gray
            0.6 0.4 0.2];% Unlabeled - brown

    overlay = labeloverlay(rgbImg, labelImg, ...
        'Colormap', cmap, 'Transparency', 0.4);

    figure
    imshow(overlay)
    title("Land Cover Classification: Forest (green), River (blue), City (gray), Unlabeled (brown)")

    % --- 7. Legend ---
    annotation("textbox", [0.75 0.60 0.2 0.05], "String", "Forest", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0]);
    annotation("textbox", [0.75 0.53 0.2 0.05], "String", "River", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.46 0.2 0.05], "String", "City", ...
        "BackgroundColor", [0.5 0.5 0.5], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.39 0.2 0.05], "String", "Unlabeled", ...
        "BackgroundColor", [0.6 0.4 0.2], "Color", [1 1 1]);

end
