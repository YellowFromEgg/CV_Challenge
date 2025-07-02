imgPaths = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2017.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Wiesn\3_2020.jpg"
};

for i = 1:length(imgPaths)
    I = imread(imgPaths{i});
    cityMask = detect_city_by_variation(I);
    imshow(labeloverlay(rgb2gray(I), cityMask, 'Transparency', 0.4));
    title('Urban Area Detected via Spectral Differentiation');
end


function cityMask = detect_city_by_variation(I)
    % Convert to grayscale
    gray = rgb2gray(I);

    % Normalize
    gray = im2double(gray);

    % Compute local standard deviation (variation)
    windowSize = 15;
    stdMap = stdfilt(gray, true(windowSize));

    % Normalize for display and thresholding
    stdMap = mat2gray(stdMap);

    % Threshold: urban = high std
    cityMask = stdMap > 0.15;  % threshold may need tuning

    % Post-processing: clean, fill, smooth
    cityMask = bwareaopen(cityMask, 100);
    cityMask = imclose(cityMask, strel('disk', 5));
    cityMask = imfill(cityMask, 'holes');

    % Display
    figure;
    subplot(1,3,1); imshow(I); title('Original Image');
    subplot(1,3,2); imshow(stdMap, []); title('Local Std Map');
    subplot(1,3,3); imshow(labeloverlay(gray, cityMask, 'Transparency', 0.4));
    title('Detected City Regions via Local Variation');
end


function cityMask = detect_city_spectral(I)
    % Convert to HSV
    hsv = rgb2hsv(I);
    H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);

    % Convert to LAB
    lab = rgb2lab(I);
    A = lab(:,:,2); B = lab(:,:,3);  % L is not very useful here

    % Normalize for thresholding
    Vn = mat2gray(V); Sn = mat2gray(S);
    An = mat2gray(A); Bn = mat2gray(B);

    % Urban areas typically have:
    % - mid brightness (not dark like sea, not bright like sand)
    % - moderate saturation
    % - distinct color components (A/B channels deviate from flat)

    % Combine multiple heuristics
    urbanMask = ...
        (Vn > 0.25 & Vn < 0.75) & ...         % exclude very dark (sea) and very bright (desert)
        (Sn > 0.15 & Sn < 0.8) & ...          % exclude extremely flat or overly saturated areas
        (abs(An - 0.5) > 0.05 | abs(Bn - 0.5) > 0.05); % A/B deviation = urban surfaces

    % Post-process
    cityMask = bwareaopen(urbanMask, 100);
    cityMask = imclose(cityMask, strel('disk', 10));
    cityMask = imfill(cityMask, 'holes');
end




for i = 1:length(imgPaths)
    imgPath = imgPaths{i};
    %stepwise_city_detection(imgPath);
end


function city_area_detection(imagePath)
    % Read and preprocess image
    I = imread(imagePath);
    gray = rgb2gray(I);
    edges = edge(gray, 'Canny');

    % Morphological processing to close gaps
    se = strel('rectangle', [3 3]);
    closedEdges = imclose(edges, se);

    % Fill holes to prepare for regionprops
    filled = imfill(closedEdges, 'holes');

    % Remove small areas
    filled = bwareaopen(filled, 100);

    % Get properties of connected components
    props = regionprops(filled, 'BoundingBox', 'Extent');

    % Classify rectangular regions (high extent suggests rectangle)
    cityMask = false(size(gray));
    for k = 1:length(props)
        if props(k).Extent > 0.7
            bb = round(props(k).BoundingBox);
            % Ensure indices are within bounds
            x1 = max(1, bb(1));
            y1 = max(1, bb(2));
            x2 = min(size(gray,2), x1 + bb(3));
            y2 = min(size(gray,1), y1 + bb(4));
            cityMask(y1:y2, x1:x2) = true;
        end
    end

    % Create a binary map of city (true) vs. non-city (false)
    cityMask = imclose(cityMask, strel('disk', 10)); % smooth regions

    % Visualization
    figure;
    subplot(1,2,1);
    imshow(I);
    title('Original Image');

    subplot(1,2,2);
    imshow(labeloverlay(gray, cityMask, 'Transparency', 0.5));
    title('City (highlighted) vs. Non-City Areas');
end

function city_area_line_density(imagePath)
    % Read and preprocess image
    I = imread(imagePath);
    gray = rgb2gray(I);
    edges = edge(gray, 'Canny');

    % Detect lines using Hough transform
    [H, theta, rho] = hough(edges);
    peaks = houghpeaks(H, 500, 'Threshold', ceil(0.3 * max(H(:))));
    lines = houghlines(edges, theta, rho, peaks, 'FillGap', 5, 'MinLength', 20);

    % Create blank image for line drawing
    lineImg = zeros(size(gray));
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        lineImg = insertShape(uint8(lineImg), 'Line', [xy(1,:) xy(2,:)], ...
                              'Color', 'white', 'LineWidth', 1);
    end
    lineImg = rgb2gray(lineImg) > 0;

    % Compute line density in sliding windows
    winSize = 50;
    step = 25;
    [rows, cols] = size(lineImg);
    cityMask = false(rows, cols);
    thresholdDensity = 0.05;

    for r = 1:step:rows - winSize
        for c = 1:step:cols - winSize
            block = lineImg(r:r+winSize-1, c:c+winSize-1);
            density = sum(block(:)) / (winSize^2);
            if density > thresholdDensity
                cityMask(r:r+winSize-1, c:c+winSize-1) = true;
            end
        end
    end

    % Smooth and clean mask
    cityMask = imclose(cityMask, strel('disk', 10));
    cityMask = imfill(cityMask, 'holes');

    % Display results
    figure;
    subplot(1,2,1);
    imshow(I);
    title('Original Image');

    subplot(1,2,2);
    imshow(labeloverlay(gray, cityMask, 'Transparency', 0.5));
    title('City Areas Based on Line Density');
end

function stepwise_city_detection2(imagePath)
    %% Step 0: Load and Prepare
    I = imread(imagePath);
    gray = rgb2gray(I);
    edges = edge(gray, 'Canny');

    %% Step 1: Detect Straight Lines using Hough Transform
    % Local contrast enhancement
        % 1. Local contrast boost
    enhanced = adapthisteq(gray);

    % 2. Enhance directional features (Laplace of Gaussian)
    h = fspecial('log', [7 7], 1);
    filtered = imfilter(enhanced, h, 'replicate');

    % 3. Multi-threshold Canny edge fusion
    edgeLow = edge(filtered, 'Canny', [0.03 0.1]);
    edgeMed = edge(filtered, 'Canny', [0.05 0.2]);
    edgeHigh = edge(filtered, 'Canny', [0.1 0.3]);
    edges = edgeLow | edgeMed | edgeHigh;

    % 4. Morphologically clean edges (close gaps, remove dust)
    edges = imclose(edges, strel('line', 3, 0));
    edges = imclose(edges, strel('line', 3, 90));
    edges = bwareaopen(edges, 30);

    % 5. Hough Transform
    [H, T, R] = hough(edges);
    P = houghpeaks(H, 500, 'Threshold', ceil(0.25 * max(H(:))));
    lines = houghlines(edges, T, R, P, 'FillGap', 8, 'MinLength', 15);

    % 6. Render lines into image
    lineImage = zeros(size(gray));
    for k = 1:length(lines)
        xy = [lines(k).point1, lines(k).point2];
        lineImage = insertShape(uint8(lineImage), 'Line', xy, 'Color', 'white', 'LineWidth', 1);
    end
    lineImageBW = rgb2gray(lineImage) > 0;

    %% Step 2: Detect Geometric Shapes (Rectangles)
        %% Step 2: Detect Overall Geometric Shapes (Polygon Grouping)
    % Create binary image of lines
    shapeEdges = imdilate(lineImageBW, strel('disk', 1));
    shapeEdges = imfill(shapeEdges, 'holes');
    shapeEdges = bwareaopen(shapeEdges, 100);

    % Find boundaries of connected components
    [B, L] = bwboundaries(shapeEdges, 'noholes');
    labeled = labelmatrix(bwconncomp(shapeEdges));

    rectMask = false(size(gray));
    for k = 1:length(B)
        boundary = B{k};
        if length(boundary) < 20, continue; end
    
        % Approximate shape using polygon simplification
        boundary = boundary(:, [2 1]);  % convert to [x y]
        poly = polyshape(boundary, 'Simplify', true);
    
        % Skip broken or invalid shapes
        if poly.NumRegions == 0 || poly.NumHoles > 0 || isempty(poly.Vertices)
            continue;
        end
    
        % Get vertices (clean x, y arrays)
        xy = poly.Vertices;
        x = xy(:,1);
        y = xy(:,2);
    
        % Filter based on polygon complexity and area
        if size(xy,1) >= 4 && size(xy,1) <= 10
            mask = poly2mask(x, y, size(gray,1), size(gray,2));
            if bwarea(mask) > 100
                rectMask = rectMask | mask;
            end
        end
    end


    %% Step 3: Detect High-Density Shape Areas = City
    win = 50;
    step = 25;
    [rows, cols] = size(rectMask);
    cityMask = false(rows, cols);
    for r = 1:step:rows - win
        for c = 1:step:cols - win
            region = rectMask(r:r+win-1, c:c+win-1);
            density = sum(region(:)) / (win^2);
            if density > 0.15
                cityMask(r:r+win-1, c:c+win-1) = true;
            end
        end
    end

    % Smooth city area boundaries
    cityMask = imopen(cityMask, strel('disk', 5));
    cityMask = imclose(cityMask, strel('disk', 10));
    cityMask = imfill(cityMask, 'holes');

    %% Display All Steps
    figure('Name', 'City Area Detection Step-by-Step', 'Position', [100 100 1600 500]);

    subplot(1,4,1);
    imshow(I); title('Original Image');

    subplot(1,4,2);
    imshow(lineImageBW); title('Step 1: Straight Lines');

    subplot(1,4,3);
    imshow(rectMask); title('Step 2: Rectangular Structures');

    subplot(1,4,4);
    imshow(labeloverlay(gray, cityMask, 'Transparency', 0.5));
    title('Step 3: City Detection by Density');
end

function stepwise_city_detection(imagePath)
    %% Load and optionally resize
    I = imread(imagePath);
    if max(size(I)) > 1024
        I = imresize(I, 0.5); % downscale for speed
    end
    gray = rgb2gray(I);

    %% STEP 1: Fast Edge + Line Detection
    enhanced = adapthisteq(gray); % local contrast
    filtered = imgaussfilt(enhanced, 1); % simple Gaussian instead of LoG
    edges = edge(filtered, 'Canny', [0.05 0.2]); % single-pass Canny

    edges = imclose(edges, strel('line', 3, 0));
    edges = imclose(edges, strel('line', 3, 90));
    edges = bwareaopen(edges, 30);

    [H, T, R] = hough(edges, 'Theta', -90:2:89); % coarser angle res
    P = houghpeaks(H, 250, 'Threshold', ceil(0.3 * max(H(:))));
    lines = houghlines(edges, T, R, P, 'FillGap', 8, 'MinLength', 20);

    % Render lines manually (faster than insertShape)
    lineImage = false(size(gray));
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        lineMask = drawLineMask(xy, size(gray));
        lineImage = lineImage | lineMask;
    end

    %% STEP 2: Fast Shape Detection
    shapeEdges = imdilate(lineImage, strel('disk', 1));
    shapeEdges = imfill(shapeEdges, 'holes');
    shapeEdges = bwareaopen(shapeEdges, 100);

    B = bwboundaries(shapeEdges, 'noholes');
    rectMask = false(size(gray));
    for k = 1:length(B)
        boundary = B{k};
        if size(boundary,1) < 20, continue; end

        xy = boundary(:, [2 1]);                   % [x y]
        xy = unique(xy, 'rows');                   % remove duplicates
        xy = reducepoly(xy, 0.01);                 % simplify
        if size(xy,1) < 4 || size(xy,1) > 10, continue; end
        mask = poly2mask(xy(:,1), xy(:,2), size(gray,1), size(gray,2));
        if bwarea(mask) > 100
            rectMask = rectMask | mask;
        end
    end

    %% STEP 3: Density-Based City Detection
    win = 50; step = 25;
    [rows, cols] = size(rectMask);
    cityMask = false(rows, cols);
    for r = 1:step:(rows - win)
        for c = 1:step:(cols - win)
            block = rectMask(r:r+win-1, c:c+win-1);
            if nnz(block) / (win^2) > 0.15
                cityMask(r:r+win-1, c:c+win-1) = true;
            end
        end
    end

    cityMask = imopen(cityMask, strel('disk', 5));
    cityMask = imclose(cityMask, strel('disk', 10));
    cityMask = imfill(cityMask, 'holes');

    %% Display
    figure('Visible', 'on', 'Position', [100 100 1600 500]);
    subplot(1,4,1); imshow(I); title('Original');
    subplot(1,4,2); imshow(lineImage); title('Edges + Lines');
    subplot(1,4,3); imshow(rectMask); title('Polygon Regions');
    subplot(1,4,4); imshow(labeloverlay(gray, cityMask, 'Transparency', 0.5)); title('City Area');
end

function mask = drawLineMask(xy, imgSize)
    % Bresenham line rasterization into a binary mask
    x1 = round(xy(1,1)); y1 = round(xy(1,2));
    x2 = round(xy(2,1)); y2 = round(xy(2,2));
    mask = false(imgSize);
    [rr, cc] = drawline(y1, x1, y2, x2);
    valid = rr > 0 & rr <= imgSize(1) & cc > 0 & cc <= imgSize(2);
    mask(sub2ind(imgSize, rr(valid), cc(valid))) = true;
end

function [rr, cc] = drawline(y1, x1, y2, x2)
    % Vectorized version of Bresenham's algorithm
    n = max(abs([x2 - x1, y2 - y1])) + 1;
    cc = round(linspace(x1, x2, n));
    rr = round(linspace(y1, y2, n));
end
