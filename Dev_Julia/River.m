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
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2003.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2005.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2010.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2015.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_2020.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2017.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Wiesn\3_2020.jpg"
};

outputFolder = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Dev_Julia\results";

for i = 1:length(imgPaths)
    detectRivers(imgPaths{i});
    %detectRiversHSV(imgPaths{i})
    %detectRivers_RapidContour(imgPaths{i})
end

%% good needs better bridging
function detectRivers_2(imgPath)
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

    % --- Step 4: Adaptive Closing Based on Local Density ---
    windowSize = 21;
    densityMap = conv2(double(combinedCandidates), ones(windowSize), 'same');
    densityMapNorm = mat2gray(densityMap);  % Normalize to [0, 1]

    denseThreshold = 0.15;
    denseMask = densityMapNorm > denseThreshold;
    sparseMask = ~denseMask;

    % Apply region-aware closing
    closedSparse = imclose(combinedCandidates & sparseMask, strel('disk', 5));
    closedDense  = imclose(combinedCandidates & denseMask,  strel('disk', 2));
    connectedEdges = closedSparse | closedDense;

    % --- Step 5: Connected Components and Filtering ---
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');

    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = false(size(grayImg));

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
            finalRiverMask(pixIdx) = true;
        end
    end

    % --- Display Results ---
    figure;

    subplot(2,3,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,3,2);
    imshow(riverCandidates); title('Color-Based River Candidates');

    subplot(2,3,3);
    imshow(riverMaskColorShape); title('Color-Based River Shape');

    subplot(2,3,4);
    imshow(BW_Sobel); title('Sobel Binary Mask (Filled)');

    subplot(2,3,5);
    imshow(connectedEdges); title('Adaptive Connected Edges');

    subplot(2,3,6);
    imshow(finalRiverMask); title('Final River Detection');
end
%% better bridgeing
function detectRivers_3(imgPath)
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
    maxBridgeDistance = 60;  % how far apart to try connecting
    bridgeRadius = 2;        % thickness of bridges
    connectedEdges = connectNearbyRiverRegions(combinedCandidates, maxBridgeDistance, bridgeRadius);

    % --- Step 5: Connected Components and Filtering ---
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');

    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = false(size(grayImg));

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
            finalRiverMask(pixIdx) = true;
        end
    end

    % --- Display Results ---
    figure;

    subplot(2,4,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,4,2);
    imshow(riverCandidates); title('Color-Based River Candidates');

    subplot(2,4,3);
    imshow(riverMaskColorShape); title('Color-Based River Shape');

    subplot(2,4,4);
    imshow(BW_Sobel); title('Sobel Binary Mask (Filled)');

    subplot(2,4,5);
    imshow(combinedCandidates); title('Combined');

    subplot(2,4,6);
    imshow(connectedEdges); title('Adaptive Connected Edges');

    subplot(2,4,7);
    imshow(finalRiverMask); title('Final River Detection');
end

function detectRivers(imgPath)
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
    finalRiverMask = false(size(grayImg));

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
            finalRiverMask(pixIdx) = true;
        end
    end

    % --- Display Results ---
    figure;

    subplot(2,4,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,4,2);
    imshow(riverCandidates); title('Color-Based River Candidates');

    subplot(2,4,3);
    imshow(riverMaskColorShape); title('Color-Based River Shape');

    subplot(2,4,4);
    imshow(BW_Sobel); title('Sobel Binary Mask (Filled)');

    subplot(2,4,5);
    imshow(combinedCandidates); title('Combined');

    subplot(2,4,6);
    imshow(connectedEdges); title('Adaptive Connected Edges');

    subplot(2,4,7);
    imshow(finalRiverMask); title('Final River Detection');
end

function detectRivers2(imgPath)
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

    % --- Filter for Long and Thin Shapes Only ---
    CC_shape = bwconncomp(riverMaskColorShape);
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
        if aspectRatio > 3
            filteredMask(CC_shape.PixelIdxList{i}) = true;
        end
    end
    
    riverMaskColorShape = filteredMask;

    % --- Step 2: Sobel Edge Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 100);
    BW_Sobel = imclose(BW_Sobel, strel('disk', 5));

    % --- Step 3: Combine Candidates ---
    combinedCandidates = riverMaskColorShape | BW_Sobel;

    % --- Step 4: Adaptive Closing Based on Local Density ---
    windowSize = 21;
    densityMap = conv2(double(combinedCandidates), ones(windowSize), 'same');
    densityMapNorm = mat2gray(densityMap);  % Normalize to [0, 1]

    denseThreshold = 0.15;
    denseMask = densityMapNorm > denseThreshold;
    sparseMask = ~denseMask;

    % Apply region-aware closing
    closedSparse = imclose(combinedCandidates & sparseMask, strel('disk', 5));
    closedDense  = imclose(combinedCandidates & denseMask,  strel('disk', 2));
    connectedEdges = closedSparse | closedDense;

    % --- Step 5: Connected Components and Filtering ---
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');

    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = false(size(grayImg));

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
            finalRiverMask(pixIdx) = true;
        end
    end

    % --- Display Results ---
    figure;

    subplot(2,3,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,3,2);
    imshow(riverCandidates); title('Color-Based River Candidates');

    subplot(2,3,3);
    imshow(riverMaskColorShape); title('Color-Based River Shape');

    subplot(2,3,4);
    imshow(BW_Sobel); title('Sobel Binary Mask (Filled)');

    subplot(2,3,5);
    imshow(connectedEdges); title('Adaptive Connected Edges');

    subplot(2,3,6);
    imshow(finalRiverMask); title('Final River Detection');
end
function detectRivers_adaptive(imgPath)
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

    % --- Filter for Long and Thin Shapes Only ---
    CC_shape = bwconncomp(riverMaskColorShape);
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
        if aspectRatio > 4 && minor > 100
            filteredMask(CC_shape.PixelIdxList{i}) = true;
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

    % --- Step 5: Connect edges from combined candidates ---
    % Count the number of non-zero pixels
    numPixels = nnz(combinedCandidates);
    
    % Get total image area
    imageArea = numel(combinedCandidates);
    
    % Compute pixel density (0 to 1)
    pixelDensity = numPixels / imageArea;
    
    % Invert and scale to define radius (adaptive logic)
    % High density → small radius, low density → large radius
    % Map density inversely to a radius between minRadius and maxRadius
    minRadius = 3;
    maxRadius = 15;
    adaptiveRadius = round(maxRadius - pixelDensity * (maxRadius - minRadius));
    
    % Create adaptive structuring element
    se = strel('disk', adaptiveRadius);
    
    % Perform closing with adaptive radius
    connectedEdges = imclose(combinedCandidates, se);

    
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');

    % --- Step 6: Geometric Filtering (Max Distance) ---
    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = false(size(grayImg));

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

        % Efficiently estimate max distance from convex hull points
        if numel(pixIdx) > 300  % optimize for large regions
            try
                K = convhull(coords(:,1), coords(:,2));
                boundaryCoords = coords(K, :);
                maxDist = max(pdist(boundaryCoords));
            catch
                maxDist = max(pdist(coords));  % fallback
            end
        else
            maxDist = max(pdist(coords));
        end

        if maxDist >= minDist
            finalRiverMask(pixIdx) = true;
        end
    end

    % --- Display Results ---
    figure;

    subplot(2,3,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,3,2);
    imshow(riverCandidates); title('Color-Based River Candidates');

    subplot(2,3,3);
    imshow(riverMaskColorShape); title('Color-Based River Shape');

    subplot(2,3,4);
    imshow(BW_Sobel); title('Sobel Binary Mask (Filled)');

    subplot(2,3,5);
    imshow(connectedEdges); title('Connected Combined Candidates');

    subplot(2,3,6);
    imshow(finalRiverMask); title('Final River Detection');
end
%%
function detectRivers_good2(imgPath)
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

    % --- Step 2: Sobel Edge Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 100);
    BW_Sobel = imclose(BW_Sobel, strel('disk', 5));

    % --- Step 3: Combine Candidates ---
    combinedCandidates = riverMaskColorShape | BW_Sobel;

    % --- Step 4: Adaptive Closing Based on Local Density ---
    windowSize = 21;
    densityMap = conv2(double(combinedCandidates), ones(windowSize), 'same');
    densityMapNorm = mat2gray(densityMap);  % Normalize to [0, 1]

    denseThreshold = 0.15;
    denseMask = densityMapNorm > denseThreshold;
    sparseMask = ~denseMask;

    % Apply region-aware closing
    closedSparse = imclose(combinedCandidates & sparseMask, strel('disk', 5));
    closedDense  = imclose(combinedCandidates & denseMask,  strel('disk', 2));
    connectedEdges = closedSparse | closedDense;

    % --- Step 5: Connected Components and Filtering ---
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');

    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = false(size(grayImg));

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
            finalRiverMask(pixIdx) = true;
        end
    end

    % --- Display Results ---
    figure;

    subplot(2,3,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,3,2);
    imshow(riverCandidates); title('Color-Based River Candidates');

    subplot(2,3,3);
    imshow(riverMaskColorShape); title('Color-Based River Shape');

    subplot(2,3,4);
    imshow(BW_Sobel); title('Sobel Binary Mask (Filled)');

    subplot(2,3,5);
    imshow(connectedEdges); title('Adaptive Connected Edges');

    subplot(2,3,6);
    imshow(finalRiverMask); title('Final River Detection');
end
%% good but bounding box not minor/major
function detectRivers_good(imgPath)
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

    % --- Filter for Long and Thin Shapes Only ---
    CC_shape = bwconncomp(riverMaskColorShape);
    stats_shape = regionprops(CC_shape, 'PixelIdxList', 'BoundingBox');
    
    filteredMask = false(size(riverMaskColorShape));
    
    for i = 1:CC_shape.NumObjects
        bbox = stats_shape(i).BoundingBox;
        width = bbox(3);
        height = bbox(4);
        
        if width == 0 || height == 0
            continue;
        end
        
        aspectRatio = max(width, height) / min(width, height);
        
        % Keep shapes with high aspect ratio (e.g., > 3)
        if aspectRatio > 3
            filteredMask(CC_shape.PixelIdxList{i}) = true;
        end
    end
    
    riverMaskColorShape = filteredMask;

    % --- Step 2: Sobel Edge Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 100);
    BW_Sobel = imclose(BW_Sobel, strel('disk', 5));

    % --- Step 3: Combine Candidates ---
    combinedCandidates = riverMaskColorShape | BW_Sobel;

    % --- Step 4: Adaptive Closing Based on Local Density ---
    windowSize = 21;
    densityMap = conv2(double(combinedCandidates), ones(windowSize), 'same');
    densityMapNorm = mat2gray(densityMap);  % Normalize to [0, 1]

    denseThreshold = 0.15;
    denseMask = densityMapNorm > denseThreshold;
    sparseMask = ~denseMask;

    % Apply region-aware closing
    closedSparse = imclose(combinedCandidates & sparseMask, strel('disk', 5));
    closedDense  = imclose(combinedCandidates & denseMask,  strel('disk', 2));
    connectedEdges = closedSparse | closedDense;

    % --- Step 5: Connected Components and Filtering ---
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');

    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = false(size(grayImg));

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
            finalRiverMask(pixIdx) = true;
        end
    end
    % --- Display Results ---
    figure;

    subplot(2,3,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,3,2);
    imshow(riverCandidates); title('Color-Based River Candidates');

    subplot(2,3,3);
    imshow(riverMaskColorShape); title('Color-Based River Shape');

    subplot(2,3,4);
    imshow(BW_Sobel); title('Sobel Binary Mask (Filled)');

    subplot(2,3,5);
    imshow(connectedEdges); title('Adaptive Connected Edges');

    subplot(2,3,6);
    imshow(finalRiverMask); title('Final River Detection');
end
%%
function finalMask = connectNearbyRiverRegions(riverMaskColorShape, maxBridgeDistance, bridgeRadius)
    % CONNECTNEARBYYRIVERREGIONS connects distinct nearby shapes using bridges
    % Inputs:
    %   riverMaskColorShape: binary mask of river shapes
    %   maxBridgeDistance: max allowed distance to try bridging (e.g. 25)
    %   bridgeRadius: how thick the connecting bridge is (e.g. 2-4)
    % Output:
    %   finalMask: original mask with connecting bridges added

    % Label regions
    labeled = bwlabel(riverMaskColorShape);
    numRegions = max(labeled(:));
    finalMask = riverMaskColorShape;

    % Get centroids for quick filtering
    props = regionprops(labeled, 'Centroid');
    centroids = cat(1, props.Centroid);

    % For each pair of regions, try connecting
    for i = 1:numRegions
        for j = i+1:numRegions
            % Estimate distance between region centroids
            dist = norm(centroids(i,:) - centroids(j,:));
            if dist > maxBridgeDistance
                continue;
            end

            % Get region masks
            mask1 = (labeled == i);
            mask2 = (labeled == j);

            % Dilate each mask slightly
            dilated1 = imdilate(mask1, strel('disk', bridgeRadius));
            dilated2 = imdilate(mask2, strel('disk', bridgeRadius));

            % Check if they touch
            if ~any(dilated1 & dilated2, 'all')
                continue;
            end

            % Find closest pair of pixels
            [y1, x1] = find(mask1);
            [y2, x2] = find(mask2);
            D = pdist2([x1 y1], [x2 y2]);
            [~, idx] = min(D(:));
            [p1, p2] = ind2sub(size(D), idx);
            pt1 = [x1(p1), y1(p1)];
            pt2 = [x2(p2), y2(p2)];

            % Draw bridge line between pt1 and pt2
            bridge = false(size(riverMaskColorShape));
            bridge(sub2ind(size(bridge), ...
                round(linspace(pt1(2), pt2(2), 100)), ...
                round(linspace(pt1(1), pt2(1), 100)))) = true;

            % Dilate bridge to desired thickness
            bridge = imdilate(bridge, strel('disk', bridgeRadius));

            % Add bridge to final mask
            finalMask = finalMask | bridge;
        end
    end
end
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

function detectRivers_notadapt(imgPath)
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

    % --- Filter for Long and Thin Shapes Only ---
    CC_shape = bwconncomp(riverMaskColorShape);
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
        if aspectRatio > 4 && minor > 100
            filteredMask(CC_shape.PixelIdxList{i}) = true;
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

    % --- Step 5: Connect edges from combined candidates ---
    connectedEdges = imclose(combinedCandidates, strel('disk', 10));
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'PixelIdxList', 'Area');

    % --- Step 6: Geometric Filtering (Max Distance) ---
    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minDist = (1/6) * imageDiagonal;
    finalRiverMask = false(size(grayImg));

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

        % Efficiently estimate max distance from convex hull points
        if numel(pixIdx) > 300  % optimize for large regions
            try
                K = convhull(coords(:,1), coords(:,2));
                boundaryCoords = coords(K, :);
                maxDist = max(pdist(boundaryCoords));
            catch
                maxDist = max(pdist(coords));  % fallback
            end
        else
            maxDist = max(pdist(coords));
        end

        if maxDist >= minDist
            finalRiverMask(pixIdx) = true;
        end
    end

    % --- Display Results ---
    figure;

    subplot(2,3,1);
    imshow(rgbImg); title('Original Image');

    subplot(2,3,2);
    imshow(riverCandidates); title('Color-Based River Candidates');

    subplot(2,3,3);
    imshow(riverMaskColorShape); title('Color-Based River Shape');

    subplot(2,3,4);
    imshow(BW_Sobel); title('Sobel Binary Mask (Filled)');

    subplot(2,3,5);
    imshow(connectedEdges); title('Connected Combined Candidates');

    subplot(2,3,6);
    imshow(finalRiverMask); title('Final River Detection');
end