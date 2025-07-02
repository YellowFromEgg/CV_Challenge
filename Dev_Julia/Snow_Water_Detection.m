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
    detection_Snow_Water(imgPaths{i});
end

function detection_Snow_Water_old(imgPath)
    % Read image
    img = imread(imgPath);
    R = double(img(:,:,1)) / 255;
    G = double(img(:,:,2)) / 255;
    B = double(img(:,:,3)) / 255;
    grayImg = rgb2gray(img);
    edges = edge(grayImg, 'Canny');

    % Edge density map
    edgeDensity = conv2(double(edges), ones(15) / 225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    % Connected components and filtering
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'Area', 'Eccentricity');

    for k = 1:length(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(CC.PixelIdxList{k}) = false;
        end
    end

    % Recompute components
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');
    labelMap = zeros(size(grayImg));

    % Region-based classification
    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        rMean = mean(R(idx));
        gMean = mean(G(idx));
        bMean = mean(B(idx));

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            label = 1; % Snow
        elseif bMean > rMean && bMean > gMean
            label = 2; % Water
        else
            label = 0;
        end

        labelMap(idx) = label;
    end

    % Check if any snow detected in region-based classification
    snowFound = any(labelMap(smoothMask) == 1);

    if snowFound
        % Pixel-wise snow classification only inside smoothMask
        whiteish = (R > 0.7) & (G > 0.7) & (B > 0.7);
        refinedSnow = whiteish & smoothMask;

        % Remove small false positives
        refinedSnow = bwareaopen(refinedSnow, 1000);

        % Update snow label
        labelMap(labelMap == 1) = 0;      % Clear existing snow
        labelMap(refinedSnow) = 1;        % Apply refined snow
    end

    % Create RGB map for visualization
    detectionRGB = zeros([size(labelMap), 3]);
    detectionRGB(:,:,1) = labelMap == 1; % Snow (white)
    detectionRGB(:,:,2) = labelMap == 1;
    detectionRGB(:,:,3) = labelMap == 1;
    detectionRGB(:,:,3) = detectionRGB(:,:,3) + (labelMap == 2); % Water (blue)

    % Display results
    figure;
    subplot(1,2,1); imshow(img); title('Original Image');
    subplot(1,2,2); imshow(detectionRGB); title('Snow (White), Water (Blue)');
end


%% good
function detection_Snow_Water2(imgPath)
    % Read image
    img = imread(imgPath);
    R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
    grayImg = rgb2gray(img);
    edges = edge(grayImg, 'Canny');

    % Edge density map
    edgeDensity = conv2(double(edges), ones(15) / 225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    % Connected components and filtering
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'Area', 'Eccentricity');

    for k = 1:length(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(CC.PixelIdxList{k}) = false;
        end
    end

    % Recompute components
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');
    labelMap = zeros(size(grayImg));

    % Classify regions
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
    end

    % Check if snow was found in region-based classification
    snowFound = any(labelMap(smoothMask) == 1);
    


    % --- New Step: Pixel-wise Water Detection by Color ---
    Rn = double(R) / 255;
    Gn = double(G) / 255;
    Bn = double(B) / 255;
    
    blueDominant = (Bn > 0.35) & (Bn > Rn + 0.08) & (Bn > Gn + 0.05);
    cyanLike = (Bn > 0.4) & (Gn > 0.4) & (Rn < 0.4) & (abs(Bn - Gn) < 0.15);
    
    pixelWaterCandidates = blueDominant | cyanLike;
    
    % --- Keep only large shapes (area filter) ---
    waterShapeMask = bwlabel(pixelWaterCandidates);
    statsWater = regionprops(waterShapeMask, 'Area');
    
    minArea = 3000;  % adjust as needed
    for i = 1:numel(statsWater)
        if statsWater(i).Area < minArea
            pixelWaterCandidates(waterShapeMask == i) = 0;
        end
    end
    
    % Mark those areas as water
    labelMap(pixelWaterCandidates) = 2;

    if snowFound
        % Pixel-wise snow detection across the full image
        whiteish = (double(R) / 255 > 0.7) & (double(G) / 255 > 0.7) & (double(B) / 255 > 0.7);
        labelMap(whiteish) = 1;
    end

    % Create RGB map for visualization
    detectionRGB = zeros([size(labelMap), 3]);
    detectionRGB(:,:,1) = labelMap == 1; % Snow: white (R=1,G=1,B=1)
    detectionRGB(:,:,2) = labelMap == 1;
    detectionRGB(:,:,3) = labelMap == 1;
    detectionRGB(:,:,3) = detectionRGB(:,:,3) + (labelMap == 2); % Water: blue

    % Display result
    figure;
    subplot(1,2,1); imshow(img); title('Original Image');
    subplot(1,2,2); imshow(detectionRGB); title('Snow (White), Water (Blue)');
end
%% same but runtime optimized
function detection_Snow_Water(imgPath)
    % Read image
    img = imread(imgPath);
    grayImg = rgb2gray(img);

    % Normalize RGB once
    Rn = double(img(:,:,1)) / 255;
    Gn = double(img(:,:,2)) / 255;
    Bn = double(img(:,:,3)) / 255;

    % --- Step 1: Snow Detection (Region-based) ---
    edges = edge(grayImg, 'Canny');
    edgeDensity = conv2(double(edges), ones(15) / 225, 'same');

    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList', 'Area', 'Eccentricity');

    % Vectorized mask cleanup
    for k = 1:length(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(stats(k).PixelIdxList) = false;
        end
    end

    % Recompute connected components once
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');

    labelMap = zeros(size(grayImg), 'uint8');  % Use smaller type

    % Classify regions (loop needed here)
    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;

        rMean = mean(Rn(idx));
        gMean = mean(Gn(idx));
        bMean = mean(Bn(idx));

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            label = 1;  % Snow
        elseif bMean > rMean && bMean > gMean
            label = 2;  % Water (preliminary)
        else
            label = 0;
        end

        if label > 0
            labelMap(idx) = label;
        end
    end

    snowFound = any(labelMap(smoothMask) == 1);

    % --- Step 2: Pixel-wise Water Detection (Color-based) ---
    blueDominant = (Bn > 0.35) & (Bn > Rn + 0.08) & (Bn > Gn + 0.05);
    cyanLike = (Bn > 0.4) & (Gn > 0.4) & (Rn < 0.4) & (abs(Bn - Gn) < 0.15);

    pixelWaterCandidates = blueDominant | cyanLike;

    % Area filtering
    waterLabel = bwlabel(pixelWaterCandidates);
    statsWater = regionprops(waterLabel, 'Area');

    minArea = 3000;
    for i = 1:numel(statsWater)
        if statsWater(i).Area < minArea
            pixelWaterCandidates(waterLabel == i) = false;
        end
    end

    labelMap(pixelWaterCandidates) = 2;

    % --- Optional Full-Image Snow Pass ---
    if snowFound
        whiteish = (Rn > 0.7) & (Gn > 0.7) & (Bn > 0.7);
        labelMap(whiteish) = 1;
    end

    % --- Visualization ---
    detectionRGB = zeros([size(labelMap), 3], 'uint8');
    detectionRGB(:,:,1) = uint8(labelMap == 1) * 255;
    detectionRGB(:,:,2) = uint8(labelMap == 1) * 255;
    detectionRGB(:,:,3) = uint8((labelMap == 1) | (labelMap == 2)) * 255;

    figure;
    subplot(1,2,1); imshow(img); title('Original Image');
    subplot(1,2,2); imshow(detectionRGB); title('Snow (White), Water (Blue)');
end


%% sometimes better sometimes worse
function detection_Snow_Water3(imgPath)
    % Read and convert image
    img = imread(imgPath);
    R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
    grayImg = rgb2gray(img);

    % --- Step 1: Snow Detection (Unchanged, Works Well) ---
    edges = edge(grayImg, 'Canny');
    edgeDensity = conv2(double(edges), ones(15) / 225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    % Region filtering for snow
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'Area', 'Eccentricity');
    for k = 1:length(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(CC.PixelIdxList{k}) = false;
        end
    end

    % Recompute regions
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');
    labelMap = zeros(size(grayImg));

    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        rMean = mean(double(R(idx))/255);
        gMean = mean(double(G(idx))/255);
        bMean = mean(double(B(idx))/255);

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            label = 1; % Snow
        elseif bMean > rMean && bMean > gMean
            label = 2; % Water (preliminary)
        else
            label = 0;
        end
        labelMap(idx) = label;
    end

    % --- Step 2: Color-Based Water Detection (Improved) ---
    hsvImg = rgb2hsv(img);
    H = hsvImg(:,:,1); S = hsvImg(:,:,2); V = hsvImg(:,:,3);
    Rn = double(R) / 255;
    Gn = double(G) / 255;
    Bn = double(B) / 255;

    % Flexible water criteria: blue-cyan color + saturation
    blueish = (Bn > 0.35) & (Bn > Rn + 0.05);
    cyanLike = (Bn > 0.4) & (Gn > 0.4) & (Rn < 0.4) & (abs(Bn - Gn) < 0.15);
    hsvWater = (H > 0.5 & H < 0.72) & (S > 0.2) & (V > 0.2 & V < 0.9);

    pixelWaterCandidates = (blueish | cyanLike | hsvWater);

    % Morphological cleaning
    pixelWaterCandidates = imfill(pixelWaterCandidates, 'holes');
    pixelWaterCandidates = bwareaopen(pixelWaterCandidates, 1000); % remove small junk

    % Area-based filtering
    waterLabel = bwlabel(pixelWaterCandidates);
    statsWater = regionprops(waterLabel, 'Area');
    minArea = 3000;
    for i = 1:numel(statsWater)
        if statsWater(i).Area < minArea
            pixelWaterCandidates(waterLabel == i) = 0;
        end
    end

    % Mark those areas as water
    labelMap(pixelWaterCandidates) = 2;

    % --- Step 3: Optional Snow Pixel Check ---
    snowFound = any(labelMap(smoothMask) == 1);
    if snowFound
        whiteish = (Rn > 0.7) & (Gn > 0.7) & (Bn > 0.7);
        labelMap(whiteish) = 1;
    end

    % --- Step 4: Visualization ---
    detectionRGB = zeros([size(labelMap), 3]);
    detectionRGB(:,:,1) = labelMap == 1; % Snow: white
    detectionRGB(:,:,2) = labelMap == 1;
    detectionRGB(:,:,3) = labelMap == 1;
    detectionRGB(:,:,3) = detectionRGB(:,:,3) + (labelMap == 2); % Water: blue

    figure;
    subplot(1,2,1); imshow(img); title('Original Image');
    subplot(1,2,2); imshow(detectionRGB); title('Snow (White), Water (Blue)');
end

function detection_Snow_Water_worse(imgPath)
    % Read and convert image
    img = imread(imgPath);
    R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
    grayImg = rgb2gray(img);

    % --- Step 1: Snow Detection (Unchanged) ---
    edges = edge(grayImg, 'Canny');
    edgeDensity = conv2(double(edges), ones(15) / 225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    % Region filtering
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'Area', 'Eccentricity');
    for k = 1:length(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(CC.PixelIdxList{k}) = false;
        end
    end

    % Recompute regions and classify
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');
    labelMap = zeros(size(grayImg));
    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        rMean = mean(double(R(idx))/255);
        gMean = mean(double(G(idx))/255);
        bMean = mean(double(B(idx))/255);

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            label = 1; % Snow
        elseif bMean > rMean && bMean > gMean
            label = 2; % Preliminary water
        else
            label = 0;
        end

        labelMap(idx) = label;
    end

    % --- Step 2: Refined Water Detection ---
    Rn = double(R) / 255;
    Gn = double(G) / 255;
    Bn = double(B) / 255;
    hsvImg = rgb2hsv(img);
    H = hsvImg(:,:,1); S = hsvImg(:,:,2); V = hsvImg(:,:,3);

    % RGB logic
    blueish = (Bn > 0.4) & (Bn > Rn + 0.1) & (Bn > Gn + 0.05);
    cyanLike = (Bn > 0.45) & (Gn > 0.45) & (Rn < 0.35) & (abs(Bn - Gn) < 0.1);
    rgbCandidate = blueish | cyanLike;

    % HSV logic
    hsvWater = (H > 0.5 & H < 0.68) & (S > 0.25) & (V > 0.25 & V < 0.9);

    % Combine both criteria
    pixelWaterCandidates = rgbCandidate & hsvWater;

    % Clean and filter
    pixelWaterCandidates = imfill(pixelWaterCandidates, 'holes');
    pixelWaterCandidates = bwareaopen(pixelWaterCandidates, 2500);

    % Area-based filtering
    waterLabel = bwlabel(pixelWaterCandidates);
    statsWater = regionprops(waterLabel, 'Area');
    minArea = 4000;
    for i = 1:numel(statsWater)
        if statsWater(i).Area < minArea
            pixelWaterCandidates(waterLabel == i) = 0;
        end
    end

    % Assign final water label
    labelMap(pixelWaterCandidates) = 2;

    % --- Step 3: Optional Snow Extension ---
    snowFound = any(labelMap(smoothMask) == 1);
    if snowFound
        whiteish = (Rn > 0.7) & (Gn > 0.7) & (Bn > 0.7);
        labelMap(whiteish) = 1;
    end

    % --- Step 4: Visualization ---
    detectionRGB = zeros([size(labelMap), 3]);
    detectionRGB(:,:,1) = labelMap == 1; % Snow: white
    detectionRGB(:,:,2) = labelMap == 1;
    detectionRGB(:,:,3) = labelMap == 1;
    detectionRGB(:,:,3) = detectionRGB(:,:,3) + (labelMap == 2); % Water: blue

    figure;
    subplot(1,2,1); imshow(img); title('Original Image');
    subplot(1,2,2); imshow(detectionRGB); title('Snow (White), Water (Blue)');
end

function detection_Snow_Waterold2(imgPath)
    % Read image
    img = imread(imgPath);
    R = double(img(:,:,1)) / 255;
    G = double(img(:,:,2)) / 255;
    B = double(img(:,:,3)) / 255;
    grayImg = rgb2gray(img);
    edges = edge(grayImg, 'Canny');

    % Edge density map
    edgeDensity = conv2(double(edges), ones(15) / 225, 'same');
    smoothMask = edgeDensity < 0.05;
    smoothMask = imfill(smoothMask, 'holes');
    smoothMask = bwareaopen(smoothMask, 500);

    % Connected components and filtering
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'Area', 'Eccentricity');

    for k = 1:length(stats)
        if stats(k).Eccentricity < 0.6 || stats(k).Area < 100
            smoothMask(CC.PixelIdxList{k}) = false;
        end
    end

    % Recompute components
    CC = bwconncomp(smoothMask);
    stats = regionprops(CC, 'PixelIdxList');
    labelMap = zeros(size(grayImg));

    % Region-based classification
    for k = 1:CC.NumObjects
        idx = stats(k).PixelIdxList;
        rMean = mean(R(idx));
        gMean = mean(G(idx));
        bMean = mean(B(idx));

        if rMean > 0.75 && gMean > 0.75 && bMean > 0.75
            label = 1; % Snow
        elseif bMean > rMean && bMean > gMean
            label = 2; % Water
        else
            label = 0;
        end

        labelMap(idx) = label;
    end

    % Pixel-wise snow detection (overrides previous snow classification)
    whiteish = (R > 0.7) & (G > 0.7) & (B > 0.7);
    labelMap(whiteish) = 1;

    % Create RGB map for visualization
    detectionRGB = zeros([size(labelMap), 3]);
    detectionRGB(:,:,1) = labelMap == 1; % Snow (white)
    detectionRGB(:,:,2) = labelMap == 1;
    detectionRGB(:,:,3) = labelMap == 1;
    detectionRGB(:,:,3) = detectionRGB(:,:,3) + (labelMap == 2); % Water (blue)

    % Display result
    figure;
    subplot(1,2,1); imshow(img); title('Original Image');
    subplot(1,2,2); imshow(detectionRGB); title('Snow (White), Water (Blue)');
end
