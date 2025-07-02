imgPaths2 = {
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
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2021_06.jpg" };

outputFolder = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Dev_Julia\results";

for i = 1:length(imgPaths)
    detectRivers(imgPaths{i});
    %detectRiversHSV(imgPaths{i})
    %detectRivers_RapidContour(imgPaths{i})
end

function finalRiverMask = detectRiversnew(imgPath)
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
    riverCandidates1 = bwareaopen(riverCandidates, 100); % remove small shapes

    riverMaskColorShape1 = riverCandidates1;
    riverMaskColorShape = imclose(riverMaskColorShape1, strel('disk', 5));

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


function connectedEdges = detectRivers(imgPath)
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


function detectRivers_accep(imgPath)
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
    riverCandidates = bwareaopen(riverCandidates, 100); %remove small shapes

    % --- Step 2: Thin White Structure Boosting ---
    whiteMask = whiteish & mask;
    whiteMask = bwareaopen(whiteMask, 100);
    CC_white = bwconncomp(whiteMask);
    stats_white = regionprops(CC_white, 'Area', 'Eccentricity', 'PixelIdxList');

    validWhite = false(size(R));
    for i = 1:CC_white.NumObjects
        if stats_white(i).Eccentricity > 0.85 && stats_white(i).Area < 300
            validWhite(stats_white(i).PixelIdxList) = true;
        end
    end

    riverMaskColorShape = (greenish & mask) | validWhite;

    % --- Step 3: Sobel Edge Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 100);
    BW_Sobel = imclose(BW_Sobel, strel('disk', 5));
    % --- Step 4: Combine Candidates ---
    combinedCandidates = riverMaskColorShape | BW_Sobel;

    % --- Step 5: Connect edges from combined candidates ---
    connectedEdges = imclose(combinedCandidates, strel('disk', 3));
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



function detectRivers_old2(imgPath)
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
    riverCandidates = bwareaopen(riverCandidates, 20);

    % --- Step 2: Thin White Structure Boosting ---
    whiteMask = whiteish & mask;
    whiteMask = bwareaopen(whiteMask, 20);
    CC_white = bwconncomp(whiteMask);
    stats_white = regionprops(CC_white, 'Area', 'Eccentricity', 'PixelIdxList');

    validWhite = false(size(R));
    for i = 1:CC_white.NumObjects
        if (stats_white(i).Eccentricity > 0.85) & (stats_white(i).Area < 300)
            validWhite(stats_white(i).PixelIdxList) = true;
        end
    end

    riverMaskColorShape = (greenish & mask) | validWhite;

    % --- Step 3: Sobel Edge Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);

    % --- Step 4: Combine Candidates ---
    combinedCandidates = riverMaskColorShape | BW_Sobel;

    % --- Step 5: Connect edges from combined candidates ---
    connectedEdges = imclose(combinedCandidates, strel('disk', 3));
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'BoundingBox', 'PixelIdxList', 'Area');

    % --- Step 6: Shape-based Filtering ---
    imageDiagonal = sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    minLength = 0.1 * imageDiagonal;
    finalRiverMask = false(size(grayImg));

    for i = 1:CC.NumObjects
        if stats(i).Area < 100
            continue;
        end

        bbox = round(stats(i).BoundingBox);
        x1 = max(1, bbox(1));
        y1 = max(1, bbox(2));
        x2 = min(size(mask,2), x1 + bbox(3) - 1);
        y2 = min(size(mask,1), y1 + bbox(4) - 1);

        subMask = false(size(mask));
        subMask(stats(i).PixelIdxList) = true;
        subRegion = subMask(y1:y2, x1:x2);

        skel = bwskel(subRegion);

        len = nnz(skel);
        bboxW = bbox(3); bboxH = bbox(4);
        aspectRatio = max(bboxW, bboxH) / (min(bboxW, bboxH) + eps);

        if len >= minLength && aspectRatio > 3
            finalRiverMask(stats(i).PixelIdxList) = true;
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




function detectRivers_ok(imgPath)
    rgbImg = imread(imgPath);
    grayImg = rgb2gray(rgbImg);
    mask = true(size(grayImg));

    % Normalize channels
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;

    % --- Color/Shape-based River Detection ---
    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.7) & (G > 0.7) & (B > 0.7);
    riverCandidates = (greenish | whiteish) & mask;
    riverCandidates = bwareaopen(riverCandidates, 20);  % cleanup small spots

    whiteMask = whiteish & mask;
    whiteMask = bwareaopen(whiteMask, 20);
    CC_white = bwconncomp(whiteMask);
    stats_white = regionprops(CC_white, 'Area', 'Eccentricity', 'PixelIdxList');

    validWhite = false(size(R));
    for i = 1:CC_white.NumObjects
        if (stats_white(i).Eccentricity > 0.85) & (stats_white(i).Area < 300)
            validWhite(stats_white(i).PixelIdxList) = true;
        end
    end

    mergedRiverMask = (greenish & mask) | validWhite;
    CC = bwconncomp(mergedRiverMask);
    stats = regionprops(CC, 'PixelIdxList', 'BoundingBox', 'Area');

    imageDiagonal = sqrt(size(mergedRiverMask,1)^2 + size(mergedRiverMask,2)^2);
    minLength = 0.1 * imageDiagonal;
    riverMaskColorShape = false(size(mergedRiverMask));

    for i = 1:CC.NumObjects
        if stats(i).Area < 100
            continue;
        end

        bbox = round(stats(i).BoundingBox);
        x1 = max(1, bbox(1));
        y1 = max(1, bbox(2));
        x2 = min(size(mask,2), x1 + bbox(3) - 1);
        y2 = min(size(mask,1), y1 + bbox(4) - 1);
        
        % Create subregion
        subMask = false(size(mask));
        subMask(stats(i).PixelIdxList) = true;
        subRegion = subMask(y1:y2, x1:x2);
        
        % Compute skeleton
        skel = bwskel(subRegion);
        
        % Compute geometric features
        len = nnz(skel);
        bboxW = bbox(3); bboxH = bbox(4);
        aspectRatio = max(bboxW, bboxH) / (min(bboxW, bboxH) + eps);
        
        % Shape-based filtering
        if len >= minLength && aspectRatio > 3
            riverMaskColorShape(stats(i).PixelIdxList) = true;
        end
    end

    % --- Sobel-based River Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);
    connectedEdges = imclose(BW_Sobel, strel('disk', 3));

    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'BoundingBox', 'PixelIdxList');

    minDiagonal = 0.1 * sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    riverMaskSobel = false(size(grayImg));

    for i = 1:CC.NumObjects
        bbox = stats(i).BoundingBox;
        diagLength = sqrt(bbox(3)^2 + bbox(4)^2);
        if diagLength >= minDiagonal
            riverMaskSobel(stats(i).PixelIdxList) = true;
        end
    end

    % --- Combine Both Masks ---
    finalRiverMask = riverMaskColorShape | riverMaskSobel;

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
    imshow(connectedEdges); title('Connected Edges');
    
    subplot(2,3,6);
    imshow(finalRiverMask); title('Combined River Detection');
end


function detectRivers_RapidContour_bad(imgPath)
    I = imread(imgPath);
    if size(I, 3) == 3
        I = rgb2gray(I);  % Convert to grayscale if RGB
    end
    I = im2uint8(I);  % Ensure 8-bit for consistency

    [m, n] = size(I);
    s = 1;  % search radius
    minCtr = 0.05;

    % Prepare shifted subimages
    rr = s+1:m-s; rrN = rr-s; rrS = rr+s;
    cc = s+1:n-s; ccE = cc+s; ccW = cc-s;
    CEN = I(rr, cc);
    NN = I(rrN, cc); SS = I(rrS, cc);
    EE = I(rr, ccE); WW = I(rr, ccW);
    NE = I(rrN, ccE); SE = I(rrS, ccE);
    SW = I(rrS, ccW); NW = I(rrN, ccW);

    % --- Local maxima & minima
    MX1 = padarray(CEN > NN & CEN > SS, [s s]);
    MX2 = padarray(CEN > NW & CEN > SE, [s s]);
    MX3 = padarray(CEN > WW & CEN > EE, [s s]);
    MX4 = padarray(CEN > NE & CEN > SW, [s s]);
    MN1 = padarray(CEN < NN & CEN < SS, [s s]);
    MN2 = padarray(CEN < NE & CEN < SW, [s s]);
    MN3 = padarray(CEN < WW & CEN < EE, [s s]);
    MN4 = padarray(CEN < NW & CEN < SE, [s s]);

    Cmax = uint8(MX1 + MX2 + MX3 + MX4);
    Cmin = uint8(MN1 + MN2 + MN3 + MN4);

    % --- Range (contrast) image
    R = colfilt(I, [2 2] + s, 'sliding', @range);
    CtrXtr = R(logical(Cmax) | logical(Cmin));
    thrXtr = max(CtrXtr) * minCtr;
    Blow = R < thrXtr;
    Cmax(Blow) = 0;
    Cmin(Blow) = 0;

    % --- Ridge (bright lines) and River (dark lines) maps
    Mrdg = Cmax >= 2;
    Mriv = Cmin >= 2;
    Mrdg = bwmorph(Mrdg, 'clean');
    Mriv = bwmorph(Mriv, 'clean');
    Mrdg = bwmorph(Mrdg, 'thin', inf);
    Mriv = bwmorph(Mriv, 'thin', inf);

    % --- Edge detection on range image
    RCN = R(rr, cc);
    Rnn = R(rrN, cc); Rss = R(rrS, cc);
    Ree = R(rr, ccE); Rww = R(rr, ccW);
    Rne = R(rrN, ccE); Rse = R(rrS, ccE);
    Rsw = R(rrS, ccW); Rnw = R(rrN, ccW);

    RX1 = padarray(RCN > Rnn & RCN > Rss, [s s]);
    RX2 = padarray(RCN > Rnw & RCN > Rse, [s s]);
    RX3 = padarray(RCN > Rww & RCN > Ree, [s s]);
    RX4 = padarray(RCN > Rne & RCN > Rsw, [s s]);
    Cedg = uint8(RX1 + RX2 + RX3 + RX4);

    Cedg = padarray(Cedg(rr, cc), [s s]);
    BWedg = Cedg >= 1;
    CtrEdg = R(BWedg);
    thrEdg = max(CtrEdg) * minCtr * 2;
    BlowEdg = CtrEdg < thrEdg;
    IxEdg = find(BWedg);
    IxLow = IxEdg(BlowEdg);
    Cedg(IxLow) = 0;

    Medg = Cedg >= 2;
    Medg = bwmorph(Medg, 'clean');
    Medg = bwmorph(Medg, 'thin', inf);

    % --- Display results
    figure('Name', 'Rapid Contour River Detection', 'Color', 'w');

    subplot(2, 3, 1); imshow(I); title('Original Image');
    subplot(2, 3, 2); imshow(R, []); title('Local Contrast (Range)');
    subplot(2, 3, 3); imshow(logical(Cmin)); title('Raw Minima Mask');
    subplot(2, 3, 4); imshow(Mriv); title('River Mask');
    subplot(2, 3, 5); imshow(Medg); title('Edge Map');
    subplot(2, 3, 6); imshow(Mrdg); title('Ridge Map');
end


function detectRivers_excludedShapes(imgPath)
    rgbImg = imread(imgPath);
    grayImg = rgb2gray(rgbImg);
    mask = true(size(grayImg));

    % Normalize channels
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;

    % --- Color/Shape-based River Detection ---
    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.75) & (G > 0.75) & (B > 0.75);
    riverCandidates = (greenish | whiteish) & mask;
    riverCandidates = bwareaopen(riverCandidates, 20);

    whiteMask = whiteish & mask;
    whiteMask = bwareaopen(whiteMask, 20);
    CC_white = bwconncomp(whiteMask);
    stats_white = regionprops(CC_white, 'Area', 'Eccentricity', 'PixelIdxList');

    validWhite = false(size(R));
    for i = 1:CC_white.NumObjects
        if (stats_white(i).Eccentricity > 0.85) & (stats_white(i).Area < 300)
            validWhite(stats_white(i).PixelIdxList) = true;
        end
    end

    mergedRiverMask = (greenish & mask) | validWhite;
    CC = bwconncomp(mergedRiverMask);
    stats = regionprops(CC, 'PixelIdxList', 'BoundingBox', 'Area');

    imageDiagonal = sqrt(size(mergedRiverMask,1)^2 + size(mergedRiverMask,2)^2);
    minLength = 0.1 * imageDiagonal;
    riverMaskColorShape = false(size(mergedRiverMask));

    for i = 1:CC.NumObjects
        if stats(i).Area < 100
            continue;
        end

        bbox = round(stats(i).BoundingBox);
        x1 = max(1, bbox(1));
        y1 = max(1, bbox(2));
        x2 = min(size(mask,2), x1 + bbox(3) - 1);
        y2 = min(size(mask,1), y1 + bbox(4) - 1);

        subMask = false(size(mask));
        subMask(stats(i).PixelIdxList) = true;
        subRegion = subMask(y1:y2, x1:x2);
        skel = bwskel(subRegion);

        if nnz(skel) >= minLength
            riverMaskColorShape(stats(i).PixelIdxList) = true;
        end
    end

    % --- Sobel-based River Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);
    connectedEdges = imclose(BW_Sobel, strel('disk', 3));

    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'BoundingBox', 'PixelIdxList');

    minDiagonal = 0.1 * sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    riverMaskSobel = false(size(grayImg));

    for i = 1:CC.NumObjects
        bbox = stats(i).BoundingBox;
        diagLength = sqrt(bbox(3)^2 + bbox(4)^2);
        if diagLength >= minDiagonal
            riverMaskSobel(stats(i).PixelIdxList) = true;
        end
    end

    % --- Combine Both Masks ---
    finalRiverMask = riverMaskColorShape | riverMaskSobel;

    % --- Display Results ---
    figure;
    subplot(1,2,1);
    imshow(rgbImg); title('Original Image');
    subplot(1,2,2);
    imshow(finalRiverMask); title('Combined River Detection');
end

function detectRivers_old(imgPath)
    rgbImg = imread(imgPath);
    grayImg = rgb2gray(rgbImg);
    mask = true(size(grayImg));

    % Normalize channels
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;

    % --- Color/Shape-based River Detection ---
    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.75) & (G > 0.75) & (B > 0.75);
    riverCandidates = (greenish | whiteish) & mask;
    riverCandidates = bwareaopen(riverCandidates, 20);

    whiteMask = whiteish & mask;
    whiteMask = bwareaopen(whiteMask, 20);
    CC_white = bwconncomp(whiteMask);
    stats_white = regionprops(CC_white, 'Area', 'Eccentricity', 'PixelIdxList');

    validWhite = false(size(R));
    for i = 1:CC_white.NumObjects
        if (stats_white(i).Eccentricity > 0.85) & (stats_white(i).Area < 300)
            validWhite(stats_white(i).PixelIdxList) = true;
        end
    end

    mergedRiverMask = (greenish & mask) | validWhite;
    CC = bwconncomp(mergedRiverMask);
    stats = regionprops(CC, 'PixelIdxList', 'BoundingBox', 'Area');

    imageDiagonal = sqrt(size(mergedRiverMask,1)^2 + size(mergedRiverMask,2)^2);
    minLength = 0.1 * imageDiagonal;
    riverMaskColorShape = false(size(mergedRiverMask));

    for i = 1:CC.NumObjects
        if stats(i).Area < 100
            continue;
        end

        bbox = round(stats(i).BoundingBox);
        x1 = max(1, bbox(1));
        y1 = max(1, bbox(2));
        x2 = min(size(mask,2), x1 + bbox(3) - 1);
        y2 = min(size(mask,1), y1 + bbox(4) - 1);
        
        % Create subregion
        subMask = false(size(mask));
        subMask(stats(i).PixelIdxList) = true;
        subRegion = subMask(y1:y2, x1:x2);
        
        % Compute skeleton
        skel = bwskel(subRegion);
        
        % Compute geometric features
        len = nnz(skel);
        bboxW = bbox(3); bboxH = bbox(4);
        aspectRatio = max(bboxW, bboxH) / (min(bboxW, bboxH) + eps); % +eps to avoid division by 0
        
        % Shape-based filtering
        if len >= minLength && aspectRatio > 3
            riverMaskColorShape(stats(i).PixelIdxList) = true;
        end
    end

    % --- Sobel-based River Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);
    connectedEdges = imclose(BW_Sobel, strel('disk', 3));

    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'BoundingBox', 'PixelIdxList');

    minDiagonal = 0.1 * sqrt(size(grayImg,1)^2 + size(grayImg,2)^2);
    riverMaskSobel = false(size(grayImg));

    for i = 1:CC.NumObjects
        bbox = stats(i).BoundingBox;
        diagLength = sqrt(bbox(3)^2 + bbox(4)^2);
        if diagLength >= minDiagonal
            riverMaskSobel(stats(i).PixelIdxList) = true;
        end
    end

    % --- Combine Both Masks ---
    finalRiverMask = riverMaskColorShape | riverMaskSobel;

    % --- Display Results ---
    figure;
    subplot(1,2,1);
    imshow(rgbImg); title('Original Image');
    subplot(1,2,2);
    imshow(finalRiverMask); title('Combined River Detection');
end

function detectRiversHSV(imgPath)
    I = imread(imgPath);

    % --- STEP 1: Preprocessing ---
    gray = rgb2gray(I);
    gray = im2double(gray);
    stdMap = stdfilt(gray, true(15));
    stdMap = mat2gray(stdMap);
    baseMask = stdMap > 0.5;

    % --- STEP 2: Extract high-variation regions ---
    cityMask = bwareaopen(baseMask, 100);
    cityMask = imclose(cityMask, strel('disk', 5));
    cityMask = imfill(cityMask, 'holes');

    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'Area');
    minArea = 500;
    keepIdx = find([props.Area] >= minArea);
    filteredMask = ismember(labelmatrix(CC), keepIdx);

    % --- STEP 3: HSV Classification ---
    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    labelMap = zeros(size(filteredMask)); % 0 = background

    for i = 1:length(keepIdx)
        idx = keepIdx(i);
        pix = props(idx).PixelIdxList;

        h = H(pix); s = S(pix); v = V(pix);

        % Classification logic
        isRedRoof = (h < 0.05 | h > 0.95) & s > 0.3;
        isGrayish = s < 0.25 & v > 0.3 & v < 0.8;
        isTanLike = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.4;
        greenish = h > 0.2 & h < 0.45 & s > 0.25;

        RedRatio = sum(isRedRoof) / numel(pix);
        GrayRatio = sum(isGrayish) / numel(pix);
        TanRatio = sum(isTanLike) / numel(pix);
        GreenRatio = sum(greenish) / numel(pix);
        VarVal = std(double(v));

        % River condition
        if GreenRatio > 0.4 && GrayRatio < 0.1 && VarVal < 0.02
            labelMap(pix) = 1; % River
        end
    end

    riverMask = labelMap == 1;

    % --- Display Results ---
    figure;
    subplot(1,4,1); imshow(I); title('Original Image');
    subplot(1,4,2); imshow(stdMap, []); title('Local Std Map');
    subplot(1,4,2); imshow(baseMask, []); title('Local Std Map');
    subplot(1,4,3); imshow(riverMask); title('HSV-Based River Detection');
end