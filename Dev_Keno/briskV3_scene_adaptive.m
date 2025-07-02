function briskV3_scene_adaptive()
    close all;
    tStart = tic;
    
    %% 1. Select image folder
    imageFolder = uigetdir(pwd, 'Select satellite image folder');
    if imageFolder == 0
        disp('No folder selected. Aborting.');
        return;
    end

    %% 2. Load and sort images
    imageFiles = [dir(fullfile(imageFolder, '*.jpg')); 
                  dir(fullfile(imageFolder, '*.png')); 
                  dir(fullfile(imageFolder, '*.tif'))];
    
    if length(imageFiles) < 2
        error('At least two images required.');
    end
    
    imageFiles = sort_nat({imageFiles.name});
    numImages = length(imageFiles);
    fprintf('Found %d images for processing.\n', numImages);

    %% 3. Preprocess ALL images once and cache them
    fprintf('\n=== Preprocessing all images ===\n');
    colorImages = cell(numImages, 1);
    grayImages = cell(numImages, 1);
    
    for i = 1:numImages
        fprintf('Preprocessing image %d/%d: %s\n', i, numImages, imageFiles{i});
        
        % Load color image
        colorImages{i} = imread(fullfile(imageFolder, imageFiles{i}));
        
        % Preprocess to grayscale (only show debug for first image)
        grayImages{i} = preprocessImage(colorImages{i}, i == 1);
        % grayImages{i} = detectRivers(colorImages{i});
    end

    %% 4. Find optimal reference using cached preprocessed images
    fprintf('\n=== Finding optimal reference image ===\n');
    refIdx = findOptimalReference(grayImages, imageFiles);
    fprintf('Selected reference image: %s (index %d)\n', imageFiles{refIdx}, refIdx);
    
    % Get reference images
    refImg = colorImages{refIdx};
    refGray = grayImages{refIdx};
    refSize = size(refGray);

    % Initialize cumulative difference
    cumulativeDiff = zeros(refSize, 'single');
    totalMask = true(refSize);

    %% 5. Process all other images against the optimal reference
    fprintf('\n=== Processing images against optimal reference ===\n');
    
    % Create single output directory
    [~, folderName] = fileparts(imageFolder);
    outputDir = fullfile(pwd, 'Registration_Results', folderName);
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    skippedImages = {};
    processedCount = 0;
    
    % Storage for registered color images
    registeredColorImages = cell(numImages, 1);
    validImageIndices = [];
    
    for i = 1:numImages
        if i == refIdx
            % Store reference image
            registeredColorImages{i} = refImg;
            validImageIndices = [validImageIndices, i];
            
            % Save reference image data
            [~, imgBaseName] = fileparts(imageFiles{i});
            saveImageResults(outputDir, refGray, refImg, true(size(refGray)), imgBaseName, 'REFERENCE');
            
            continue; % Skip the reference image itself
        end
        
        fprintf('Processing image %d/%d: %s\n', i, numImages, imageFiles{i});
        
        % Use cached preprocessed images
        img = colorImages{i};
        gray = grayImages{i};
        
        % Registration with color image support
        [registered, validMask, registeredColor] = registerImages(refGray, gray, imageFiles{i}, true, refImg, img);
    
        % Get image base name
        [~, imgBaseName] = fileparts(imageFiles{i});
        
        if isempty(registered)
            skippedImages{end+1} = imageFiles{i};
            fprintf('  -> Skipped due to registration failure\n');
            
            % Save failure marker
            failurePath = fullfile(outputDir, sprintf('%s_FAILED.txt', imgBaseName));
            fid = fopen(failurePath, 'w');
            if fid > 0
                fprintf(fid, 'Registration failed for: %s\n', imageFiles{i});
                fclose(fid);
            end
            continue;
        end
    
        % Store the registered color image
        registeredColorImages{i} = registeredColor;
        validImageIndices = [validImageIndices, i];
    
        % Calculate difference
        diffImage = imabsdiff(refGray, registered);
        diffImage(~validMask) = 0;
    
        % Accumulate
        cumulativeDiff = cumulativeDiff + single(diffImage);
        totalMask = totalMask & validMask;
        processedCount = processedCount + 1;
        
        % Save registration results in single folder
        saveImageResults(outputDir, registered, registeredColor, validMask, imgBaseName, 'REGISTERED');
        
        fprintf('  -> Successfully registered and processed\n');
    end
    
    fprintf('\nAll results saved to: %s\n', outputDir);

    fprintf('\n=== Processing Summary ===\n');
    fprintf('Reference image: %s\n', imageFiles{refIdx});
    fprintf('Successfully processed: %d/%d images (%.1f%%)\n', ...
        processedCount, numImages-1, 100*processedCount/(numImages-1));
    
    if ~isempty(skippedImages)
        fprintf('Skipped images:\n');
        for i = 1:length(skippedImages)
            fprintf('  - %s\n', skippedImages{i});
        end
    end

    %% 5. Visualize results
    cumulativeDiff(~totalMask) = 0;
    visualizeDifferenceHeatmap(cumulativeDiff, totalMask);

    displayRegisteredColorImages(registeredColorImages, validImageIndices, imageFiles, refIdx);
    
    fprintf('\nProcessing complete!\n');

    elapsedTime = toc(tStart);  % Time in seconds
    fprintf('Elapsed time: %.4f seconds\n', elapsedTime);
end

%% -- detectRivers --
function connectedEdges = detectRivers(imgPath)
    rgbImg = imgPath;
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

%% --- Save image results to single folder ---
function saveImageResults(outputDir, registeredGray, registeredColor, validMask, baseName, suffix)
    try
        % Save registered grayscale image
        if ~isempty(registeredGray)
            grayPath = fullfile(outputDir, sprintf('%s_%s_gray.mat', baseName, suffix));
            registered_gray = registeredGray; %#ok<NASGU>
            save(grayPath, 'registered_gray', '-v7.3');
        end
        
        % Save registered color image
        if ~isempty(registeredColor)
            colorPath = fullfile(outputDir, sprintf('%s_%s_color.mat', baseName, suffix));
            registered_color = registeredColor; %#ok<NASGU>
            save(colorPath, 'registered_color', '-v7.3');
        end
        
        % Save valid mask
        if ~isempty(validMask)
            maskPath = fullfile(outputDir, sprintf('%s_%s_mask.mat', baseName, suffix));
            valid_mask = validMask; %#ok<NASGU>
            save(maskPath, 'valid_mask', '-v7.3');
        end
        
    catch ME
        warning('Failed to save results for %s: %s', baseName, ME.message);
    end
end

%% --- Find optimal reference image based on feature matching success ---
function optimalIdx = findOptimalReference(preprocessedImages, imageFiles)
    numImages = length(preprocessedImages);
    
    % For large datasets, sample a subset for efficiency
    if numImages > 20
        sampleIndices = round(linspace(1, numImages, min(15, numImages)));
        fprintf('Sampling %d images for reference selection\n', length(sampleIndices));
    else
        sampleIndices = 1:numImages;
    end
    
    % Use cached preprocessed images
    fprintf('Using cached preprocessed images for reference selection...\n');
    images = cell(length(sampleIndices), 1);
    for i = 1:length(sampleIndices)
        idx = sampleIndices(i);
        images{i} = preprocessedImages{idx};
    end
    
    % Calculate pairwise registration success matrix
    fprintf('Calculating pairwise registration success...\n');
    successMatrix = zeros(length(sampleIndices), length(sampleIndices));
    featureCountMatrix = zeros(length(sampleIndices), length(sampleIndices));
    
    for i = 1:length(sampleIndices)
        for j = i+1:length(sampleIndices)
            [success, featureCount] = testRegistration(images{i}, images{j});
            successMatrix(i, j) = success;
            successMatrix(j, i) = success;
            featureCountMatrix(i, j) = featureCount;
            featureCountMatrix(j, i) = featureCount;
        end
        if mod(i, 3) == 0
            fprintf('  Completed %d/%d comparisons\n', i, length(sampleIndices));
        end
    end
    
    % Calculate reference quality scores
    referenceScores = zeros(length(sampleIndices), 1);
    
    for i = 1:length(sampleIndices)
        % Success rate as reference
        successRate = mean(successMatrix(i, :));
        
        % Average feature count when used as reference
        avgFeatureCount = mean(featureCountMatrix(i, :));
        
        % Bonus for images in the middle of the time series
        temporalPosition = sampleIndices(i) / numImages;
        temporalBonus = 1 - abs(temporalPosition - 0.5); % Peak at 0.5
        
        % Combined score
        referenceScores(i) = successRate * 0.6 + ...
                           (avgFeatureCount / 100) * 0.3 + ...
                           temporalBonus * 0.1;
        
        fprintf('Image %d (%s): Success=%.2f, Features=%.1f, Score=%.3f\n', ...
            sampleIndices(i), imageFiles{sampleIndices(i)}, ...
            successRate, avgFeatureCount, referenceScores(i));
    end
    
    % Select the best reference
    [~, bestIdx] = max(referenceScores);
    optimalIdx = sampleIndices(bestIdx);
    
    fprintf('\nReference selection results:\n');
    fprintf('Best reference: %s (score: %.3f)\n', ...
        imageFiles{optimalIdx}, referenceScores(bestIdx));
end

%% --- Test registration between two images ---
function [success, featureCount] = testRegistration(gray1, gray2)
    success = 0;
    featureCount = 0;
    
    try
        % Crop images to remove scale/logo area
        cropRatio = 0.07;
        gray1Cropped = cropImageForFeatureDetection(gray1, cropRatio);
        gray2Cropped = cropImageForFeatureDetection(gray2, cropRatio);

        % Quick feature detection and matching
        pts1 = detectSURFFeatures(gray1Cropped, 'MetricThreshold', 250);
        pts2 = detectSURFFeatures(gray2Cropped, 'MetricThreshold', 250);
        
        if pts1.Count < 10 || pts2.Count < 10
            return;
        end
        
        [f1, vpts1] = extractFeatures(gray1Cropped, pts1);
        [f2, vpts2] = extractFeatures(gray2Cropped, pts2);
        
        indexPairs = matchFeatures(f1, f2, 'Unique', true, 'MaxRatio', 0.7);
        
        if size(indexPairs, 1) < 4
            return;
        end
        
        featureCount = size(indexPairs, 1);
        
        % Quick transformation test
        matched1 = vpts1(indexPairs(:,1)).Location;
        matched2 = vpts2(indexPairs(:,2)).Location;
        
        [~, inlierIdx] = estgeotform2d(matched2, matched1, 'similarity', ...
            'MaxNumTrials', 1000, 'Confidence', 85);
        
        inlierRatio = sum(inlierIdx) / length(inlierIdx);
        
        if sum(inlierIdx) >= 4 && inlierRatio >= 0.2
            success = 1;
        end
        
    catch
        % Registration failed
        success = 0;
    end
end

%% --- Create cropped images for feature detection
function croppedImage = cropImageForFeatureDetection(image, cropRatio)
    % Crop the bottom portion of the image to remove scale/logo
    % cropRatio: fraction to remove from bottom (e.g., 0.1 = remove bottom 10%)
    if nargin < 2
        cropRatio = 0.08; % Default: remove bottom 8%
    end
    
    [height, ~, ~] = size(image);
    cropHeight = round(height * (1 - cropRatio));
    
    if length(size(image)) == 3
        croppedImage = image(1:cropHeight, :, :);
    else
        croppedImage = image(1:cropHeight, :);
    end
end

%% --- Adaptive Feature Detection Based on Scene Type ---
function [matched1, matched2, featureStats] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped, showDebug)
    % Analyze scene content to determine optimal parameters
    sceneType1 = 'city';
    sceneType2 = 'city';
    
    % Use the more challenging scene type for parameter selection
    if strcmp(sceneType1, 'water') || strcmp(sceneType2, 'water')
        sceneType = 'water';
    elseif strcmp(sceneType1, 'nature') || strcmp(sceneType2, 'nature')
        sceneType = 'nature';
    else
        sceneType = 'city';
    end
    
    if showDebug
        fprintf('  Scene types: %s + %s -> Using %s parameters\n', sceneType1, sceneType2, sceneType);
    end
    
    % Get adaptive parameters
    [surfParams, briskParams] = getSceneParameters(sceneType);
    
    % SURF feature detection with adaptive parameters
    pts1SURF = detectSURFFeatures(gray1Cropped, ...
        'MetricThreshold', surfParams.threshold, ...
        'NumOctaves', surfParams.octaves);
    pts2SURF = detectSURFFeatures(gray2Cropped, ...
        'MetricThreshold', surfParams.threshold, ...
        'NumOctaves', surfParams.octaves);

    [f1SURF, vpts1SURF] = extractFeatures(gray1Cropped, pts1SURF);
    [f2SURF, vpts2SURF] = extractFeatures(gray2Cropped, pts2SURF);

    indexPairsSURF = matchFeatures(f1SURF, f2SURF, ...
        'Unique', true, 'MaxRatio', surfParams.maxRatio);

    % BRISK feature detection with adaptive parameters
    pts1BRISK = detectBRISKFeatures(gray1Cropped, ...
        'MinContrast', briskParams.contrast, ...
        'NumOctaves', briskParams.octaves);
    pts2BRISK = detectBRISKFeatures(gray2Cropped, ...
        'MinContrast', briskParams.contrast, ...
        'NumOctaves', briskParams.octaves);

    [f1BRISK, vpts1BRISK] = extractFeatures(gray1Cropped, pts1BRISK, 'Method', 'BRISK');
    [f2BRISK, vpts2BRISK] = extractFeatures(gray2Cropped, pts2BRISK, 'Method', 'BRISK');
    
    indexPairsBRISK = matchFeatures(f1BRISK, f2BRISK, ...
        'MatchThreshold', briskParams.matchThreshold, ...
        'MaxRatio', briskParams.maxRatio, 'Unique', true);

    % Combine matches
    matched1 = [];
    matched2 = [];
    surfMatches = 0;
    briskMatches = 0;
    
    if ~isempty(indexPairsSURF)
        matched1SURF = vpts1SURF(indexPairsSURF(:,1)).Location;
        matched2SURF = vpts2SURF(indexPairsSURF(:,2)).Location;
        matched1 = [matched1; matched1SURF];
        matched2 = [matched2; matched2SURF];
        surfMatches = size(matched1SURF, 1);
        if showDebug
            fprintf('  SURF matches (%s params): %d\n', sceneType, surfMatches);
        end
    end
    
    if ~isempty(indexPairsBRISK)
        matched1BRISK = vpts1BRISK(indexPairsBRISK(:,1)).Location;
        matched2BRISK = vpts2BRISK(indexPairsBRISK(:,2)).Location;
        matched1 = [matched1; matched1BRISK];
        matched2 = [matched2; matched2BRISK];
        briskMatches = size(matched1BRISK, 1);
        if showDebug
            fprintf('  BRISK matches (%s params): %d\n', sceneType, briskMatches);
        end
    end
    
    % Return feature statistics for debug output
    featureStats = struct();
    featureStats.surfMatches = surfMatches;
    featureStats.briskMatches = briskMatches;
    featureStats.sceneType = sceneType;
end

%% --- Scene-Specific Parameter Sets ---
function [surfParams, briskParams] = getSceneParameters(sceneType)
    switch lower(sceneType)
        case 'water'   % low-texture, almost flat
            surfParams = struct( ...
                'threshold',        120 , ...  % ↓ metric threshold – keep faint blobs
                'octaves',            5 , ...
                'maxRatio',        0.90);      % more lenient matching later
            briskParams = struct( ...
                'contrast',       0.004 , ...  % ↓ min contrast
                'octaves',           5 , ...
                'matchThreshold',    25 , ...  % a bit stricter than 30
                'maxRatio',       0.90);
    
        case 'city'    % high-texture, lots of strong corners
            surfParams = struct( ...
                'threshold',        1000 , ...  % ↑ only strongest survive
                'octaves',            3 , ...
                'maxRatio',        0.50);      % be picky – plenty of good matches
            briskParams = struct( ...
                'contrast',       0.035 , ...
                'octaves',           3 , ...
                'matchThreshold',     8 , ...
                'maxRatio',       0.55);
    
        case 'nature'  % mixed structure
            surfParams = struct( ...
                'threshold',        350 , ...
                'octaves',            4 , ...
                'maxRatio',        0.80);
            briskParams = struct( ...
                'contrast',       0.015 , ...
                'octaves',           4 , ...
                'matchThreshold',    15 , ...
                'maxRatio',       0.80);
    
        otherwise
            error('Unknown sceneType "%s". Use water | city | nature.', sceneType);
    end
end

%% --- Enhanced registration function with comprehensive debug output ---
function [registered2, validMask, registeredColor] = registerImages(gray1, gray2, imageName, showDebug, colorImg1, colorImg2)
    if nargin < 3
        imageName = 'Current Image';
    end
    if nargin < 4
        showDebug = false;
    end
    if nargin < 5
        colorImg1 = [];
    end
    if nargin < 6
        colorImg2 = [];
    end

    registeredColor = []; % Initialize output
    
    try
        % Crop images for feature detection
        cropRatio = 0.07; % Remove bottom 7% of image
        gray1Cropped = cropImageForFeatureDetection(gray1, cropRatio);
        gray2Cropped = cropImageForFeatureDetection(gray2, cropRatio);
        
        if showDebug
            fprintf('  Original size: %dx%d, Cropped size: %dx%d\n', ...
                size(gray1,1), size(gray1,2), size(gray1Cropped,1), size(gray1Cropped,2));
        end

        % Replace the fixed parameter feature detection with adaptive approach
        [matched1, matched2, featureStats] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped, showDebug);

        if size(matched1, 1) < 4
            if showDebug
                warning('Insufficient matches (%d) for %s', size(matched1, 1), imageName);
            end
            registered2 = [];
            validMask = [];
            registeredColor = [];
            return;
        end

        if showDebug
            fprintf('  Total adaptive matches: %d\n', size(matched1, 1));
        end

        % Robust transformation estimation
        [tform, inlierIdx] = estgeotform2d(matched2, matched1, 'similarity', ...
            'MaxNumTrials', 3000, 'Confidence', 90);
        
        inlierCount = sum(inlierIdx);
        inlierRatio = inlierCount / length(inlierIdx);
        
        if showDebug
            fprintf('  Inliers: %d/%d (%.1f%%)\n', inlierCount, length(inlierIdx), inlierRatio*100);
        end 

        if inlierCount < 1
            if showDebug
                warning('Too few inliers (%d) for %s', inlierCount, imageName);
            end
            registered2 = [];
            validMask = [];
            registeredColor = [];
            return;
        end

        % Apply transformation to grayscale image
        outputRef = imref2d(size(gray1));
        registered2 = imwarp(gray2, tform, 'OutputView', outputRef, ...
            'Interp', 'linear', 'FillValues', 0);

        % Apply same transformation to color image if provided
        if ~isempty(colorImg2)
            if size(colorImg2, 3) == 3
                % Register each color channel separately
                registeredColor = zeros([size(gray1), 3], 'like', colorImg2);
                for ch = 1:3
                    registeredColor(:,:,ch) = imwarp(colorImg2(:,:,ch), tform, ...
                        'OutputView', outputRef, 'Interp', 'linear', 'FillValues', 0);
                end
            else
                % Single channel color image
                registeredColor = imwarp(colorImg2, tform, 'OutputView', outputRef, ...
                    'Interp', 'linear', 'FillValues', 0);
            end
        end

        % Create valid mask
        mask = ones(size(gray2));
        warpedMask = imwarp(mask, tform, 'OutputView', outputRef);
        validMask = warpedMask > 0.5;

        %% COMPREHENSIVE DEBUG OUTPUT
        if showDebug
            % Create main debug figure
            figName = sprintf('Registration Debug: %s', imageName);
            fig = figure('Name', figName, 'NumberTitle', 'off', 'Position', [100, 100, 1400, 900]);
            
            % 1. Feature matches visualization
            subplot(3, 4, 1);
            showMatchedFeatures(gray1, gray2, matched1(inlierIdx,:), matched2(inlierIdx,:), 'montage');
            title(sprintf('Inlier Matches: %d (%.1f%%)', inlierCount, inlierRatio*100));
            
            % 2. All feature matches (including outliers)
            subplot(3, 4, 2);
            showMatchedFeatures(gray1, gray2, matched1, matched2, 'montage');
            title(sprintf('All Matches: %d', size(matched1, 1)));
            
            % 3. Reference image
            subplot(3, 4, 3);
            imshow(gray1);
            title('Reference Image');
            
            % 4. Input image to be registered
            subplot(3, 4, 4);
            imshow(gray2);
            title('Input Image');
            
            % 5. Registered result
            subplot(3, 4, 5);
            imshow(registered2);
            title('Registered Image');
            
            % 6. Overlay - Reference and Registered
            subplot(3, 4, 6);
            imshowpair(gray1, registered2, 'falsecolor');
            title('False Color Overlay');
            
            % 7. Checkerboard overlay
            subplot(3, 4, 7);
            imshowpair(gray1, registered2, 'Scaling', 'joint');
            title('Checkerboard Overlay');
            
            % 8. Difference image
            subplot(3, 4, 8);
            diffImg = imabsdiff(gray1, registered2);
            diffImg(~validMask) = 0;
            imagesc(diffImg);
            axis image off;
            colormap(gca, hot);
            colorbar;
            title('Difference Image');
            
            % 9. Valid mask
            subplot(3, 4, 9);
            imshow(validMask);
            title(sprintf('Valid Mask (%.1f%% valid)', 100*sum(validMask(:))/numel(validMask)));
            
            % 10. Transformation visualization
            subplot(3, 4, 10);
            % Show transformation effects on corner points
            corners = [1, 1; size(gray2, 2), 1; size(gray2, 2), size(gray2, 1); 1, size(gray2, 1)];
            transformedCorners = transformPointsForward(tform, corners);
            
            hold on;
            plot(corners([1:end, 1], 1), corners([1:end, 1], 2), 'b-', 'LineWidth', 2);
            plot(transformedCorners([1:end, 1], 1), transformedCorners([1:end, 1], 2), 'r-', 'LineWidth', 2);
            legend('Original', 'Transformed', 'Location', 'best');
            axis equal;
            grid on;
            title('Transformation Effect');
            
            % 11. Feature distribution in reference
            subplot(3, 4, 11);
            imshow(gray1); hold on;
            plot(matched1(inlierIdx, 1), matched1(inlierIdx, 2), 'g+', 'MarkerSize', 8, 'LineWidth', 2);
            plot(matched1(~inlierIdx, 1), matched1(~inlierIdx, 2), 'r+', 'MarkerSize', 6, 'LineWidth', 1);
            title('Features on Reference');
            legend('Inliers', 'Outliers', 'Location', 'best');
            
            % 12. Quality metrics - FIXED FOR COMPATIBILITY
            subplot(3, 4, 12);
            axis off;
            
            % Calculate additional quality metrics
            validArea = sum(validMask(:)) / numel(validMask);
            avgDiff = mean(diffImg(validMask));
            stdDiff = std(diffImg(validMask));
            
            % Display metrics with compatible text properties - FIXED
            metrics = {
                sprintf('Total Matches: %d', size(matched1, 1));
                sprintf('Inliers: %d (%.1f%%)', inlierCount, inlierRatio*100);
                sprintf('Valid Area: %.1f%%', validArea*100);
                sprintf('Avg Difference: %.4f', avgDiff);
                sprintf('Std Difference: %.4f', stdDiff);
                sprintf('SURF Features: %d', featureStats.surfMatches);
                sprintf('BRISK Features: %d', featureStats.briskMatches);
                sprintf('Scene Type: %s', featureStats.sceneType);
            };
            
            % Use compatible text properties (removed FontFamily)
            text(0.1, 0.9, metrics, 'Units', 'normalized', 'FontSize', 10, ...
                'VerticalAlignment', 'top');
            title('Registration Metrics');
            
            % Add overall title
            try
                % Try sgtitle (newer MATLAB versions)
                sgtitle(sprintf('Registration Analysis: %s', imageName), 'FontSize', 14, 'FontWeight', 'bold');
            catch
                % Fallback for older MATLAB versions
                annotation('textbox', [0 0.95 1 0.05], 'String', sprintf('Registration Analysis: %s', imageName), ...
                    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
            end
            
        end

    catch ME
        if showDebug
            warning('Registration failed for %s: %s', imageName, ME.message);
        end
        registered2 = [];
        validMask = [];
        registeredColor = [];
    end
end

%% --- Enhanced preprocessing ---
function grayImage = preprocessImage(img, showFigure)
    if nargin < 2
        showFigure = false;
    end

    if showFigure
        figure('Name', 'Image Preprocessing', 'NumberTitle', 'off');
        subplot(2, 3, 1);
        imshow(img);
        title('Original');
    end

    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    img = im2double(img);

    if showFigure
        subplot(2, 3, 2);
        imshow(img);
        title('Grayscale');
    end

    img = imgaussfilt(img, 0.3);

    if showFigure
        subplot(2, 3, 3);
        imshow(img);
        title('Noise Reduced');
    end

    img = adapthisteq(img, 'ClipLimit', 0.01, 'NumTiles', [6 6]);

    if showFigure
        subplot(2, 3, 4);
        imshow(img);
        title('Histogram Equalized');
    end

    img = mat2gray(img);

    if showFigure
        subplot(2, 3, 5);
        imshow(img);
        title('Normalized');
        subplot(2, 3, 6);
        imshow(img);
        title('Final Result');
    end

    grayImage = img;
end

% function grayImage = preprocessImage(img, showFigure)
%     if nargin < 2
%         showFigure = false;
%     end
% 
%     % Convert to grayscale
%     if size(img, 3) == 3
%         grayImage = rgb2gray(img);
%     else
%         grayImage = img;
%     end
% 
%     % Convert to double
%     grayImage = im2double(grayImage);
% 
%     % Analyze image characteristics
%     imgMean = mean(grayImage(:));
%     imgStd = std(grayImage(:));
%     imgRange = max(grayImage(:)) - min(grayImage(:));
% 
%     % Conditional preprocessing based on image characteristics
%     if imgRange < 0.3  % Low contrast image
%         if showFigure
%             fprintf('  Applying contrast enhancement (range=%.3f)\n', imgRange);
%         end
%         % Gentle contrast enhancement
%         grayImage = adapthisteq(grayImage, 'ClipLimit', 0.005, 'NumTiles', [8 8]);
%     end
% 
%     if imgMean < 0.3 || imgMean > 0.7  % Very dark or very bright
%         if showFigure
%             fprintf('  Applying brightness normalization (mean=%.3f)\n', imgMean);
%         end
%         % Gentle brightness normalization
%         grayImage = imadjust(grayImage, [0.1 0.9], []);
%     end
% 
%     if imgStd < 0.05  % Very flat image (possibly hazy)
%         if showFigure
%             fprintf('  Applying gentle sharpening (std=%.3f)\n', imgStd);
%         end
%         % Very light unsharp masking
%         grayImage = imsharpen(grayImage, 'Radius', 1, 'Amount', 0.3);
%     end
% 
%     % Always apply very light denoising for satellite images
%     grayImage = imgaussfilt(grayImage, 0.2);  % Reduced from 0.3
% 
%     if showFigure
%         fprintf('  Final stats: mean=%.3f, std=%.3f, range=%.3f\n', ...
%             mean(grayImage(:)), std(grayImage(:)), max(grayImage(:)) - min(grayImage(:)));
%     end
% end

%% --- Enhanced visualization with statistics ---
function visualizeDifferenceHeatmap(diffImage, validMask)
    validDiff = diffImage(validMask);
    meanDiff = mean(validDiff);
    stdDiff = std(validDiff);
    maxDiff = max(validDiff);
    
    fprintf('\nDifference Statistics:\n');
    fprintf('  Mean: %.4f, Std: %.4f, Max: %.4f\n', meanDiff, stdDiff, maxDiff);
    
    figure('Name', 'Change Detection Results', 'NumberTitle', 'off');
    
    subplot(2,2,1);
    imagesc(diffImage);
    axis image off;
    colormap(gca, jet);
    colorbar;
    title('Raw Difference Heatmap');
    
    subplot(2,2,2);
    threshold = meanDiff + 2*stdDiff;
    thresholdedDiff = diffImage;
    thresholdedDiff(diffImage < threshold) = 0;
    imagesc(thresholdedDiff);
    axis image off;
    colormap(gca, hot);
    colorbar;
    title(sprintf('Significant Changes (>%.3f)', threshold));
    
    subplot(2,2,3);
    histogram(validDiff, 50, 'Normalization', 'probability');
    hold on;
    xline(meanDiff, 'r--', 'LineWidth', 2);
    xline(threshold, 'g--', 'LineWidth', 2);
    xlabel('Difference Value');
    ylabel('Probability');
    title('Distribution of Differences');
    grid on;
    
    subplot(2,2,4);
    imshow(validMask);
    title('Valid Registration Area');
    
    validPixels = sum(validMask(:));
    totalPixels = numel(validMask);
    validPercent = 100 * validPixels / totalPixels;
    
    sgtitle(sprintf('Change Detection Results - %.1f%% valid area', validPercent));
end

%% --- Natural sorting function ---
function sorted = sort_nat(filenames)
    expr = '\d{4}';
    years = regexp(filenames, expr, 'match', 'once');
    validYears = ~cellfun(@isempty, years);
    if sum(validYears) == 0
        sorted = sort(filenames);
        return;
    end
    years = cellfun(@str2double, years);
    [~, idx] = sort(years);
    sorted = filenames(idx);
end

%% --- Display all registered color images in a grid ---
function displayRegisteredColorImages(registeredColorImages, validIndices, imageFiles, refIdx)
    % Remove empty cells and get valid images
    validImages = registeredColorImages(validIndices);
    validFiles = imageFiles(validIndices);
    
    numImages = length(validImages);
    if numImages == 0
        fprintf('No valid registered images to display.\n');
        return;
    end
    
    % Calculate grid dimensions
    cols = ceil(sqrt(numImages));
    rows = ceil(numImages / cols);
    
    % Create figure
    figure('Name', 'All Registered Color Images', 'NumberTitle', 'off', ...
           'Position', [50, 50, min(1800, cols*300), min(1200, rows*250)]);
    
    for i = 1:numImages
        subplot(rows, cols, i);
        
        if ~isempty(validImages{i})
            imshow(validImages{i});
            
            % Add title with special marking for reference
            if validIndices(i) == refIdx
                title(sprintf('%d: %s (REF)', validIndices(i), validFiles{i}), ...
                      'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            else
                title(sprintf('%d: %s', validIndices(i), validFiles{i}), ...
                      'FontSize', 9);
            end
        else
            % Show placeholder for failed registration
            axis off;
            text(0.5, 0.5, 'Registration Failed', 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'red');
            title(sprintf('%d: %s (FAILED)', validIndices(i), validFiles{i}), ...
                  'FontSize', 9, 'Color', 'red');
        end
        
        axis off;
    end
    
    % Add overall title
    try
        sgtitle(sprintf('Registered Color Images (%d/%d successful)', ...
                numImages, length(registeredColorImages)), ...
                'FontSize', 14, 'FontWeight', 'bold');
    catch
        % Fallback for older MATLAB versions
        annotation('textbox', [0 0.95 1 0.05], ...
                   'String', sprintf('Registered Color Images (%d/%d successful)', ...
                                   numImages, length(registeredColorImages)), ...
                   'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
                   'FontSize', 14, 'FontWeight', 'bold');
    end
    
    fprintf('\nDisplayed %d registered color images in grid format.\n', numImages);
end