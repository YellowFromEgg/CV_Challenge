% Master MATLAB script for comparing multiple region segmentation approaches
% Includes: Original image + 6 segmentation methods
imgPath = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg";
imgPath1 ="C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg";
imgPath2 = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg";
imgPath3 = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2000.jpg";


RegionDetectionEdges(imgPath);
RegionDetectionEdges(imgPath1);
RegionDetectionEdges(imgPath2);
RegionDetectionEdges(imgPath3);
%RegionDetectionEdges(imgPath4)

function RegionDetectionEdges(imgPath)
    % Load image
    img = imread(imgPath);
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    % --- Sobel Edge Detection ---
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);
    [B_Sobel, ~] = bwboundaries(BW_Sobel, 'noholes');

    % --- Connect Nearby Edge Fragments ---
    connectedEdges = imclose(BW_Sobel, strel('disk', 3)); % Connect small gaps

    % --- Filter by Diagonal Length Only ---
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'BoundingBox', 'PixelIdxList');

    [h, w] = size(grayImg);
    minDiagonal = 0.1 * sqrt(h^2 + w^2);

    riverMask = false(h, w);

    for i = 1:CC.NumObjects
        bbox = stats(i).BoundingBox;
        diagLength = sqrt(bbox(3)^2 + bbox(4)^2);
        if diagLength >= minDiagonal
            riverMask(stats(i).PixelIdxList) = true;
        end
    end

    % --- Plotting ---
    figure;
    subplot(2,3,1); imshow(img); title('Original Image');

    subplot(2,3,2); imshow(edgesSobel); title('Raw Sobel Edges');

    subplot(2,3,3); imshow(img); hold on;
    for k = 1:length(B_Sobel)
        boundary = B_Sobel{k};
        plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1);
    end
    title('Sobel - Detected Shapes');

    subplot(2,3,4); imshow(connectedEdges); title('Connected Edge Fragments');

    subplot(2,3,5); imshow(false(size(connectedEdges))); title('---'); axis off;

    subplot(2,3,6); imshow(img); hold on;
    boundaries = bwboundaries(riverMask);
    for k = 1:length(boundaries)
        plot(boundaries{k}(:,2), boundaries{k}(:,1), 'g', 'LineWidth', 1.5);
    end
    title('Filtered - Long Diagonal Regions Only');
end


function RegionDetectionEdges_better(imgPath)
    % Load image
    img = imread(imgPath);

    % Convert to grayscale
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    % Sobel edge detection
    edgesSobel = edge(grayImg, 'Sobel');
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);
    [B_Sobel, ~] = bwboundaries(BW_Sobel, 'noholes');

    % Connect edges
    connectedEdges = imclose(BW_Sobel, strel('disk', 10));  % fast morphological op


    % Precompute values
    [h, w] = size(grayImg);
    minLength = 0.05 * sqrt(h^2 + w^2);

    % Connected components and properties
    CC = bwconncomp(connectedEdges);
    stats = regionprops(CC, 'BoundingBox', 'Area', 'PixelIdxList');

    riverMask = false(h, w);

    for i = 1:CC.NumObjects
        if stats(i).Area < 100  % skip small areas early
            continue;
        end

        % Cropped region mask for efficiency
        bbox = round(stats(i).BoundingBox);
        x1 = max(1, bbox(1)); y1 = max(1, bbox(2));
        x2 = min(w, x1 + bbox(3) - 1);
        y2 = min(h, y1 + bbox(4) - 1);

        regionMask = false(h, w);
        regionMask(stats(i).PixelIdxList) = true;
        cropped = regionMask(y1:y2, x1:x2);

        % Skeleton length check
        skel = bwskel(cropped);
        if nnz(skel) >= minLength
            riverMask(stats(i).PixelIdxList) = true;
        end
    end

    % --- Plotting ---
    figure;

    subplot(2,3,1); imshow(img); title('Original Image');
    subplot(2,3,2); imshow(edgesSobel); title('Raw Sobel Edges');
    
    subplot(2,3,3); imshow(img); hold on;
    for k = 1:length(B_Sobel)
        boundary = B_Sobel{k};
        plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1);
    end
    title('Sobel - Detected Shapes');

    subplot(2,3,4); imshow(connectedEdges); title('Connected Edge Fragments');

    subplot(2,3,5); imshow(img); hold on;
    riverBoundaries = bwboundaries(riverMask);
    for k = 1:length(riverBoundaries)
        plot(riverBoundaries{k}(:,2), riverBoundaries{k}(:,1), 'g', 'LineWidth', 1.5);
    end
    title('Filtered Sobel - Long River Only');
end


function RegionDetectionEdges_good(imgPath)
    % Load image
    img = imread(imgPath);

    % Convert to grayscale if needed
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    % Optional blur
    blurredImg = grayImg; % Or use: imgaussfilt(grayImg, 2);
    
    % --- Edge Detection ---
    edgesSobel = edge(blurredImg, 'Sobel');

    % --- Contour detection from Sobel ---
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);
    [B_Sobel, ~] = bwboundaries(BW_Sobel, 'noholes');


    % Label connected regions
    CC = bwconncomp(BW_Sobel);
    labeled = labelmatrix(CC);
    stats = regionprops(CC, 'PixelIdxList');

    % Image diagonal for length threshold
    [h, w] = size(grayImg);
    minLength = 0.05 * sqrt(h^2 + w^2);

    % Initialize final river mask
    riverMask = false(h, w);

    % Keep only long structures
    for i = 1:CC.NumObjects
        regionMask = false(h, w);
        regionMask(CC.PixelIdxList{i}) = true;
        skel = bwskel(regionMask);
        if nnz(skel) >= minLength
            riverMask = riverMask | regionMask;
        end
    end

    % --- Plotting result ---
    % --- Plotting all results ---
    figure;
    
    % 1. Original Image
    subplot(2,3,1);
    imshow(img);
    title('Original Image');
    

    
    % 5. Sobel Contours
    subplot(2,3,2);
    imshow(img); hold on;
    for k = 1:length(B_Sobel)
       boundary = B_Sobel{k};
       plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1);
    end
    title('Sobel - Detected Shapes');
    
    subplot(2,3,3);
    imshow(img); hold on;
    boundaries = bwboundaries(riverMask);
    for k = 1:length(boundaries)
        boundary = boundaries{k};
        plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1.5);
    end
    title('Filtered Sobel - Long River Only');
end


function RegionDetectionEdges2(imgPath)
    % Load the image
    img = imread(imgPath);
    
    % Convert to grayscale if needed
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % Optional blur
    blurredImg = grayImg; % Or use: imgaussfilt(grayImg, 2);
    
    % --- Edge Detection ---
    edgesCanny = edge(blurredImg, 'Canny');
    edgesSobel = edge(blurredImg, 'Sobel');
    
    % --- Contour detection from Canny ---
    BW_Canny = imfill(edgesCanny, 'holes');
    BW_Canny = bwareaopen(BW_Canny, 50);
    [B_Canny, ~] = bwboundaries(BW_Canny, 'noholes');
    
    % --- Contour detection from Sobel ---
    BW_Sobel = imfill(edgesSobel, 'holes');
    BW_Sobel = bwareaopen(BW_Sobel, 50);
    [B_Sobel, ~] = bwboundaries(BW_Sobel, 'noholes');
    
    % --- Plotting all results ---
    figure;
    
    % 1. Original Image
    subplot(2,3,1);
    imshow(img);
    title('Original Image');
    
    % 2. Canny Edges
    subplot(2,3,2);
    imshow(edgesCanny);
    title('Canny Edge Detection');
    
    % 3. Canny Contours
    subplot(2,3,3);
    imshow(img); hold on;
    for k = 1:length(B_Canny)
       boundary = B_Canny{k};
       plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 1);
    end
    title('Canny - Detected Shapes');
    
    % 4. Sobel Edges
    subplot(2,3,5);
    imshow(edgesSobel);
    title('Sobel Edge Detection');
    
    % 5. Sobel Contours
    subplot(2,3,6);
    imshow(img); hold on;
    for k = 1:length(B_Sobel)
       boundary = B_Sobel{k};
       plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1);
    end
    title('Sobel - Detected Shapes');
    
    % (Leave subplot 2,1,4 empty or add histogram/stats if needed)
end