imgPaths1 = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1990.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2005.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2010.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2015.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2000.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2002.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2004.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1990.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1995.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2000.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2003.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2005.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2010.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2015.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2020.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2017.jpg"
    %"C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Wiesn\3_2020.jpg"
};

imgPaths2 = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2015_07.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2015_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2016_07.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2017_04.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2018_04.jpg"

    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2019_03.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2019_06.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2020_03.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2021_06.jpg" };

outputFolder = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Dev_Julia\results";

% Beispiel: imgPaths = {"img1.png", "img2.png", ..., "img6.png"};

numImages = length(imgPaths);
rows = ceil(sqrt(numImages));
cols = ceil(numImages / rows);

figure('Name', 'Connected Edges (Rivers)', 'Position', [100 100 1400 800]);

for i = 1:numImages
    connectedEdges = detectRivers(imgPaths{i});
    
    subplot(rows, cols, i);
    imshow(connectedEdges);
    title(sprintf('Image %d', i));
end


for i = 1:length(imgPaths)
    imgPath = imgPaths{i};
    connectedEdges = detectRivers(imgPath);

    I = imread(imgPath);
    %CityMask = detect_city_land_dense(I);

    isCityOrLand = findWideUrbanAreas(connectedEdges);
    filteredRiverMask = connectedEdges & ~isCityOrLand;
    
    figure;
    subplot(2,2,1); imshow(I); title(sprintf('Original %d', i));
    subplot(2,2,2); imshow(connectedEdges); title('Connected Edges');
    subplot(2,2,3); imshow(isCityOrLand); title('Connected Edges');
    subplot(2,2,4); imshow(filteredRiverMask); title('Filtered Rivers');
end


function connectedEdges = detectRivers(imgPath)
    rgbImg = imread(imgPath);
    grayImg = rgb2gray(rgbImg);
    mask = true(size(grayImg));

    % Normalize channels
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;

    % Step 1: Color-based candidates
    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.7) & (G > 0.7) & (B > 0.7);
    riverCandidates = (greenish | whiteish) & mask;
    riverCandidates = bwareaopen(riverCandidates, 10);

    riverMaskColorShape = riverCandidates;

    % Step 2: Sobel edges
    edgesSobel = edge(grayImg, 'Sobel');
    edgesCanny = edge(grayImg, 'Canny', [0.03 0.2]);
    edgesCombined = edgesCanny;
    BW_Sobel = imfill(edgesCombined, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 100);
    BW_Sobel = imclose(BW_Sobel, strel('disk', 6));

    % Step 3: Combine
    combinedCandidates = riverMaskColorShape | BW_Sobel;

    % Step 4: Adaptive closing
    windowSize = 21;
    densityMap = conv2(double(combinedCandidates), ones(windowSize), 'same');
    densityMapNorm = mat2gray(densityMap);
    denseThreshold = 0.15;

    denseMask = densityMapNorm > denseThreshold;
    sparseMask = ~denseMask;

    closedSparse = imclose(combinedCandidates & sparseMask, strel('disk', 5));
    closedDense  = imclose(combinedCandidates & denseMask,  strel('disk', 2));
    connectedEdges = closedSparse | closedDense;
end

function filteredRiverMask = detect_nonCity_nonLand_rivers(imgPath)
    % --- Step 1: Detect river candidates (binary mask)
    connectedEdges = detectRivers(imgPath);  % binary logical mask

    % --- Step 2: Get city/land classification map
    I = imread(imgPath);
    CityMask = detect_city_land(I);  % labelMap: 0=background, 2=land, 3=city, etc.

    % --- Step 3: Filter river candidates
    isCityOrLand = CityMask == 2 | CityMask == 3;
    filteredRiverMask = connectedEdges & ~isCityOrLand;

    % --- Step 4: Visualization
    figure;
    subplot(1,3,1); imshow(I); title('Original Image');
    subplot(1,3,2); imshow(connectedEdges); title('All River Candidates');
    subplot(1,3,3); imshow(filteredRiverMask); title('Only Non-City/Non-Land Rivers');
end

function [labelMap] = detect_city_land(I)
    % --- Step 1: Preprocessing ---
    gray = im2double(rgb2gray(I));
    stdMap = mat2gray(stdfilt(gray, true(15)));
    baseMask = stdMap > 0.15;

    % --- Step 2: Morphological cleaning and filtering ---
    cityMask = baseMask; %imfill(imclose(bwareaopen(baseMask, 100), strel('disk', 5)), 'holes');

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

        % --- Shadow detection ---
        if mean(v) < 0.12 && std(v) < 0.05
            labelMap(pix) = 8; % Shadow
            continue; % Skip further classification
        end

        % --- Other region classification ---
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

function [CityMask] = detect_city_land_dense(I)
    % Step 1: Convert to grayscale and compute local variation
    gray = im2double(rgb2gray(I));
    stdMap = stdfilt(gray, true(15));  % 15×15 window
    stdMap = mat2gray(stdMap);

    % Step 2: Threshold on local variation
    baseMask = stdMap > 0.12;  % You can adjust this if too much/little is selected

    % Step 3: Morphological processing to find dense regions
    cityMask = imfill(imclose(bwareaopen(baseMask, 200), strel('disk', 5)), 'holes');

    % Step 4: Keep only large, contiguous urban-scale regions
    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'Area', 'PixelIdxList');
    CityMask = zeros(size(cityMask), 'uint8');

    for i = 1:length(props)
        if props(i).Area > 3000  % Only really large dense zones
            CityMask(props(i).PixelIdxList) = 3;  % 3 = City
        end
    end
end

function isCity = findWideUrbanAreas(binaryMask)
    % This function filters the input binary mask to find only large, wide areas
    % that resemble urban structures or broad cleared regions.
    % Returns isCity as a logical matrix (same size as input)

    % Connected component analysis
    CC = bwconncomp(binaryMask);
    props = regionprops(CC, 'Area', 'MajorAxisLength', 'MinorAxisLength', 'PixelIdxList');

    isCity = false(size(binaryMask));

    % Tuning parameters
    minArea = 1000;       % Minimum area of urban blob
    maxAspectRatio = 2.5; % To filter out long/thin rivers (high ratio)

    for i = 1:CC.NumObjects
        area = props(i).Area;
        major = props(i).MajorAxisLength;
        minor = props(i).MinorAxisLength;

        if minor == 0
            continue;
        end

        aspect = major / minor;

        % Keep only wide + large regions
        if area > minArea && aspect < maxAspectRatio
            isCity(props(i).PixelIdxList) = true;
        end
    end
end
