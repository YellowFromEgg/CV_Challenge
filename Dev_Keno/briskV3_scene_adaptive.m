function briskV3_scene_adaptive()
    close all;
    tStart = tic;
    
    % Initialize global flag for skipping manual registration
    global skipAllManualRegistration;
    skipAllManualRegistration = false;
    
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
        [matched1, matched2, featureStats] = adaptiveFeatureDetectionQualityFocused(gray1Cropped, gray2Cropped, false);
        
        featureCount = featureStats.totalMatches;
        
        if featureCount >= 4
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
        cropRatio = 0.07; % Default: remove bottom 7%
    end
    
    [height, ~, ~] = size(image);
    cropHeight = round(height * (1 - cropRatio));
    
    if length(size(image)) == 3
        croppedImage = image(1:cropHeight, :, :);
    else
        croppedImage = image(1:cropHeight, :);
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

        MaxDistance_list = [1.5, 5, 10, 50, 100, 500];
        Confidence = [90, 90, 90, 93, 95, 97];
        inlierCount = 0;
        i = 1;
        
        while inlierCount < 5 && i <= length(Confidence)
            % Robust transformation estimation
            [tform, inlierIdx] = estgeotform2d(matched2, matched1, 'similarity', ...
                'MaxNumTrials', 3000, 'Confidence', 93, 'MaxDistance', MaxDistance_list(i));
            
            inlierCount = sum(inlierIdx);
            inlierRatio = inlierCount / length(inlierIdx);
        
            i = i + 1;
        end
        
        % safety check based on resulting image size
        scl = hypot(tform.T(1,1), tform.T(2,1));     % isotropic scale
        rot = atan2d(tform.T(2,1), tform.T(1,1));    % rotation in degrees
        tx  = tform.T(3,1);  ty = tform.T(3,2);      % translation (pixels)
        
        % % project-dependent limits – massively scaled images are skipped
        % if  scl < 0.75 || scl > 1.75 || ...          % ≤ ±75 % zoom allowed
        %     abs(tx) > 0.8*size(gray1,2) || ...       % ≤ 80 % width translation
        %     abs(ty) > 0.8*size(gray1,1)
        %     warning('Rejected registration: scale=%.3f  rot=%.1f°  Δx=%d  Δy=%d', ...
        %             scl, rot, round(tx), round(ty));
        %     registered2   = [];       %   ➜ treat as failure
        %     validMask     = [];
        %     registeredColor = [];
        %     return
        % end
        
        
        if showDebug
            fprintf('  Inliers: %d/%d (%.1f%%)\n', inlierCount, length(inlierIdx), inlierRatio*100);
        end 

        if inlierCount < 2
            if showDebug
                warning('Too few inliers (%d) for %s', inlierCount, imageName);
            end
            
            % Ask user if they want to manually select points
            answer = questdlg(['Automatic registration failed for ' imageName '. Would you like to manually select corresponding points?'], ...
                'Manual Registration', 'Yes', 'No', 'Skip All Manual', 'Yes');
            
            switch answer
                case 'Yes'
                    [manualMatched1, manualMatched2] = manualPointSelection(gray1, gray2, imageName, showDebug);
                    
                    if size(manualMatched1, 1) >= 4
                        % Use manually selected points
                        matched1 = manualMatched1;
                        matched2 = manualMatched2;
                        
                        % Recompute transformation with manual points
                        [tform, inlierIdx] = estgeotform2d(matched2, matched1, 'similarity', ...
                            'MaxNumTrials', 1000, 'Confidence', 95);
                        
                        inlierCount = sum(inlierIdx);
                        inlierRatio = inlierCount / length(inlierIdx);
                        
                        if showDebug
                            fprintf('  Manual registration: %d inliers from %d manual points (%.1f%%)\n', ...
                                inlierCount, size(matched1, 1), inlierRatio*100);
                        end
                        
                        % Continue with registration process
                        % (don't return here - let the code continue)
                    else
                        % Not enough manual points selected
                        if showDebug
                            fprintf('  Manual point selection cancelled or insufficient points\n');
                        end
                        registered2 = [];
                        validMask = [];
                        registeredColor = [];
                        return;
                    end
                    
                case 'Skip All Manual'
                    % Set global flag to skip all future manual selections
                    global skipAllManualRegistration;
                    skipAllManualRegistration = true;
                    registered2 = [];
                    validMask = [];
                    registeredColor = [];
                    return;
                    
                otherwise % 'No' or cancelled
                    registered2 = [];
                    validMask = [];
                    registeredColor = [];
                    return;
            end
        else
            % Sufficient automatic inliers found - continue normally
            if showDebug
                fprintf('  Automatic registration successful: %d inliers (%.1f%%)\n', inlierCount, inlierRatio*100);
            end
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
            
            % % 7. Checkerboard overlay
            % subplot(3, 4, 7);
            % imshowpair(gray1, registered2, 'Scaling', 'joint');
            % title('Checkerboard Overlay');
            % 
            % % 8. Difference image
            % subplot(3, 4, 8);
            % diffImg = imabsdiff(gray1, registered2);
            % diffImg(~validMask) = 0;
            % imagesc(diffImg);
            % axis image off;
            % colormap(gca, hot);
            % colorbar;
            % title('Difference Image');
            % 
            % % 9. Valid mask
            % subplot(3, 4, 9);
            % imshow(validMask);
            % title(sprintf('Valid Mask (%.1f%% valid)', 100*sum(validMask(:))/numel(validMask)));
            % 
            % % 10. Transformation visualization
            % subplot(3, 4, 10);
            % % Show transformation effects on corner points
            % corners = [1, 1; size(gray2, 2), 1; size(gray2, 2), size(gray2, 1); 1, size(gray2, 1)];
            % transformedCorners = transformPointsForward(tform, corners);
            % 
            % hold on;
            % plot(corners([1:end, 1], 1), corners([1:end, 1], 2), 'b-', 'LineWidth', 2);
            % plot(transformedCorners([1:end, 1], 1), transformedCorners([1:end, 1], 2), 'r-', 'LineWidth', 2);
            % legend('Original', 'Transformed', 'Location', 'best');
            % axis equal;
            % grid on;
            % title('Transformation Effect');
            % 
            % % 11. Feature distribution in reference
            % subplot(3, 4, 11);
            % imshow(gray1); hold on;
            % plot(matched1(inlierIdx, 1), matched1(inlierIdx, 2), 'g+', 'MarkerSize', 8, 'LineWidth', 2);
            % plot(matched1(~inlierIdx, 1), matched1(~inlierIdx, 2), 'r+', 'MarkerSize', 6, 'LineWidth', 1);
            % title('Features on Reference');
            % legend('Inliers', 'Outliers', 'Location', 'best');
            % 
            % % 12. Quality metrics - FIXED FOR COMPATIBILITY
            % subplot(3, 4, 12);
            % axis off;
            
            % % Calculate additional quality metrics
            % validArea = sum(validMask(:)) / numel(validMask);
            % avgDiff = mean(diffImg(validMask));
            % stdDiff = std(diffImg(validMask));
            
            % Display metrics with compatible text properties - FIXED
            % metrics = {
            %     sprintf('Total Matches: %d', size(matched1, 1));
            %     sprintf('Inliers: %d (%.1f%%)', inlierCount, inlierRatio*100);
            %     sprintf('Valid Area: %.1f%%', validArea*100);
            %     sprintf('Avg Difference: %.4f', avgDiff);
            %     sprintf('Std Difference: %.4f', stdDiff);
            %     sprintf('SURF Features: %d', featureStats.surfMatches);
            %     sprintf('BRISK Features: %d', featureStats.briskMatches);
            %     sprintf('Scene Type: %s', featureStats.sceneType);
            % };
            
            % Use compatible text properties (removed FontFamily)
            % text(0.1, 0.9, metrics, 'Units', 'normalized', 'FontSize', 10, ...
            %     'VerticalAlignment', 'top');
            % title('Registration Metrics');
            % 
            % % Add overall title
            % try
            %     % Try sgtitle (newer MATLAB versions)
            %     sgtitle(sprintf('Registration Analysis: %s', imageName), 'FontSize', 14, 'FontWeight', 'bold');
            % catch
            %     % Fallback for older MATLAB versions
            %     annotation('textbox', [0 0.95 1 0.05], 'String', sprintf('Registration Analysis: %s', imageName), ...
            %         'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
            % end
            
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

%% --- Enhanced Manual Point Selection with Zoom and Alternating Selection ---
function [manualMatched1, manualMatched2] = manualPointSelection(gray1, gray2, imageName, showDebug)
    % Initialize outputs
    manualMatched1 = [];
    manualMatched2 = [];
    
    % Check global skip flag
    global skipAllManualRegistration;
    if ~isempty(skipAllManualRegistration) && skipAllManualRegistration
        return;
    end
    
    try
        % Create figure for manual point selection
        figHandle = figure('Name', ['Manual Point Selection: ' imageName], ...
            'NumberTitle', 'off', 'Position', [50, 50, 1500, 800]);
        
        % Display both images side by side
        subplot(1, 2, 1);
        imshow(gray1);
        title('Reference Image', 'FontSize', 12, 'FontWeight', 'bold');
        xlabel('Select point #1 here', 'FontSize', 10, 'Color', 'red');
        
        subplot(1, 2, 2);
        imshow(gray2);
        title(['Input Image: ' imageName], 'FontSize', 12, 'FontWeight', 'bold');
        xlabel('Then select corresponding point here', 'FontSize', 10, 'Color', 'blue');
        
        % FIXED: Enable zoom without mouse wheel callback
        zoom(figHandle, 'on');
        
        % Add instructions
        sgtitle('Manual Point Selection - Select 4 point pairs (zoom enabled)', ...
            'FontSize', 14, 'FontWeight', 'bold');
        
        % Enhanced instructions with zoom info
        annotation('textbox', [0.02, 0.02, 0.96, 0.15], ...
            'String', ['ZOOM CONTROLS: Zoom icon in toolbar, Click+drag to zoom area, Double-click = zoom to fit' char(10) ...
                      'POINT SELECTION: 1) Click point in LEFT image, then corresponding point in RIGHT image' char(10) ...
                      '2) Repeat for 4 point pairs (auto-stops after 4 pairs)' char(10) ...
                      '3) Press ESC to cancel selection'], ...
            'EdgeColor', 'black', 'BackgroundColor', 'yellow', ...
            'FontSize', 10, 'HorizontalAlignment', 'center');
        
        % Alternating point selection with zoom support
        [x1, y1, x2, y2] = alternatingPointSelectionWithZoomFixed(gray1, gray2, figHandle, showDebug);
        
        if length(x1) < 4
            if showDebug
                fprintf('  Insufficient points selected (%d < 4)\n', length(x1));
            end
            close(figHandle);
            return;
        end
        
        % Prepare output
        manualMatched1 = [x1(:), y1(:)];
        manualMatched2 = [x2(:), y2(:)];
        
        % Show final result
        showFinalPointSelectionWithZoomFixed(gray1, gray2, x1, y1, x2, y2, figHandle);
        
        if showDebug
            fprintf('  Manual point selection completed: %d point pairs selected\n', length(x1));
        end
        
        % Keep figure open for a moment to show result
        pause(3);
        close(figHandle);
        
    catch ME
        if showDebug
            warning('Manual point selection failed: %s', ME.message);
        end
        if exist('figHandle', 'var') && isvalid(figHandle)
            close(figHandle);
        end
    end
end

%% --- Alternating Point Selection with Zoom Support (FIXED) ---
function [x1, y1, x2, y2] = alternatingPointSelectionWithZoomFixed(gray1, gray2, figHandle, showDebug)
    x1 = [];
    y1 = [];
    x2 = [];
    y2 = [];
    
    % FIXED: Use only supported color names
    colors = {'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white'};
    
    % Target number of point pairs
    targetPairs = 4;
    
    % Instructions for user
    fprintf('\n=== MANUAL POINT SELECTION ===\n');
    fprintf('Use zoom toolbar button to enable zoom, click+drag to zoom area\n');
    fprintf('Double-click to zoom to fit\n');
    fprintf('Select %d point pairs by alternating between images\n\n', targetPairs);
    
    try
        for pairIdx = 1:targetPairs
            if showDebug
                fprintf('Selecting point pair %d/%d...\n', pairIdx, targetPairs);
            end
            
            % Get color for this pair
            currentColor = colors{mod(pairIdx-1, length(colors)) + 1};
            
            % =================== SELECT POINT ON REFERENCE IMAGE ===================
            figure(figHandle);
            subplot(1, 2, 1);
            xlabel(sprintf('ZOOM enabled: Click point #%d here (pair %d/%d)', pairIdx, pairIdx, targetPairs), ...
                   'FontSize', 11, 'Color', currentColor, 'FontWeight', 'bold');
            
            % Ensure we're in the correct subplot and wait for click
            subplot(1, 2, 1);
            
            % Wait for user click with enhanced feedback
            fprintf('  -> Click on point #%d in the LEFT (reference) image...\n', pairIdx);
            
            % Custom point selection that works with zoom
            [xi1, yi1, success1] = selectPointWithZoomFixed(currentColor, pairIdx, 'reference');
            
            if ~success1
                if showDebug
                    fprintf('  Point selection cancelled on reference image\n');
                end
                return;
            end
            
            % Show selected point on reference image
            hold on;
            plot(xi1, yi1, 'o', 'Color', currentColor, 'MarkerSize', 14, 'LineWidth', 3);
            plot(xi1, yi1, '+', 'Color', 'white', 'MarkerSize', 12, 'LineWidth', 2);
            text(xi1+8, yi1-8, num2str(pairIdx), 'Color', currentColor, ...
                 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');
            
            % =================== SELECT CORRESPONDING POINT ON INPUT IMAGE ===================
            subplot(1, 2, 2);
            xlabel(sprintf('ZOOM enabled: Click corresponding point #%d here', pairIdx), ...
                   'FontSize', 11, 'Color', currentColor, 'FontWeight', 'bold');
            
            % Ensure we're in the correct subplot
            subplot(1, 2, 2);
            
            fprintf('  -> Click on corresponding point #%d in the RIGHT (input) image...\n', pairIdx);
            
            % Custom point selection that works with zoom
            [xi2, yi2, success2] = selectPointWithZoomFixed(currentColor, pairIdx, 'input');
            
            if ~success2
                if showDebug
                    fprintf('  Point selection cancelled on input image\n');
                end
                return;
            end
            
            % Show selected point on input image
            hold on;
            plot(xi2, yi2, 'o', 'Color', currentColor, 'MarkerSize', 14, 'LineWidth', 3);
            plot(xi2, yi2, '+', 'Color', 'white', 'MarkerSize', 12, 'LineWidth', 2);
            text(xi2+8, yi2-8, num2str(pairIdx), 'Color', currentColor, ...
                 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');
            
            % Store the point pair
            x1(pairIdx) = xi1;
            y1(pairIdx) = yi1;
            x2(pairIdx) = xi2;
            y2(pairIdx) = yi2;
            
            % Update progress
            if pairIdx < targetPairs
                fprintf('  ✓ Point pair %d/%d completed. Continue with next pair...\n', pairIdx, targetPairs);
                
                % Update instructions
                subplot(1, 2, 1);
                xlabel(sprintf('Point pair %d/%d completed. Next: click point #%d', ...
                       pairIdx, targetPairs, pairIdx+1), ...
                       'FontSize', 10, 'Color', 'green');
                subplot(1, 2, 2);
                xlabel('Wait for next point selection...', 'FontSize', 10, 'Color', 'black');
            else
                fprintf('  ✓ All %d point pairs completed successfully!\n', targetPairs);
                
                % Final success message
                subplot(1, 2, 1);
                xlabel('✓ All 4 point pairs completed!', 'FontSize', 11, 'Color', 'green', 'FontWeight', 'bold');
                subplot(1, 2, 2);
                xlabel('✓ All 4 point pairs completed!', 'FontSize', 11, 'Color', 'green', 'FontWeight', 'bold');
            end
            
            % Update main title
            sgtitle(sprintf('Manual Point Selection - %d/%d point pairs completed (zoom enabled)', ...
                    pairIdx, targetPairs), 'FontSize', 14, 'FontWeight', 'bold');
            
            drawnow;
            pause(0.5); % Brief pause for visual feedback
        end
        
        fprintf('\n✓ Successfully selected %d point pairs for registration!\n', targetPairs);
        
    catch ME
        if showDebug
            warning('Alternating point selection failed: %s', ME.message);
        end
        x1 = [];
        y1 = [];
        x2 = [];
        y2 = [];
    end
end

%% --- Enhanced Point Selection with Zoom Support (FIXED) ---
function [x, y, success] = selectPointWithZoomFixed(color, pointNum, imageType)
    x = [];
    y = [];
    success = false;
    
    try
        % Wait for user click
        [x, y, button] = ginput(1);
        
        % Check if user cancelled (ESC, close window, etc.)
        if isempty(x) || isempty(y)
            return;
        end
        
        % Check if it's a valid left click
        if button == 1  % Left mouse button
            success = true;
            fprintf('    Point #%d selected at (%.1f, %.1f) on %s image\n', ...
                    pointNum, x, y, imageType);
        else
            fprintf('    Invalid click (use left mouse button)\n');
        end
        
    catch ME
        fprintf('    Point selection cancelled or failed: %s\n', ME.message);
        success = false;
    end
end

%% --- Show Final Point Selection with Zoom Maintained (FIXED) ---
function showFinalPointSelectionWithZoomFixed(gray1, gray2, x1, y1, x2, y2, figHandle)
    % Update the display to show final results while maintaining zoom
    figure(figHandle);
    
    % FIXED: Use only supported color names
    colors = {'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white'};
    
    % Reference image with final points
    subplot(1, 2, 1);
    % Don't reset the image - just update the display
    hold on;
    
    % Clear any previous point markers and redraw all points
    for i = 1:length(x1)
        currentColor = colors{mod(i-1, length(colors)) + 1};
        plot(x1(i), y1(i), 'o', 'Color', currentColor, 'MarkerSize', 14, 'LineWidth', 3);
        plot(x1(i), y1(i), '+', 'Color', 'white', 'MarkerSize', 12, 'LineWidth', 2);
        text(x1(i)+10, y1(i)-10, num2str(i), 'Color', currentColor, ...
             'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');
    end
    title('Reference Image - Final Points', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('✓ All points selected (zoom still active)', 'FontSize', 10, 'Color', 'green', 'FontWeight', 'bold');
    
    % Input image with final points
    subplot(1, 2, 2);
    hold on;
    
    for i = 1:length(x2)
        currentColor = colors{mod(i-1, length(colors)) + 1};
        plot(x2(i), y2(i), 'o', 'Color', currentColor, 'MarkerSize', 14, 'LineWidth', 3);
        plot(x2(i), y2(i), '+', 'Color', 'white', 'MarkerSize', 12, 'LineWidth', 2);
        text(x2(i)+10, y2(i)-10, num2str(i), 'Color', currentColor, ...
             'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');
    end
    title('Input Image - Final Points', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('✓ All points selected (zoom still active)', 'FontSize', 10, 'Color', 'green', 'FontWeight', 'bold');
    
    % Update main title
    sgtitle('✓ Manual Point Selection Complete - 4 point pairs selected', ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', 'green');
    
    % Add enhanced success message
    annotation('textbox', [0.02, 0.02, 0.96, 0.15], ...
        'String', ['✓ SUCCESS: 4 corresponding point pairs selected and will be used for registration' char(10) ...
                  'Point pairs are color-coded and numbered for verification' char(10) ...
                  'Zoom functionality remains active for verification'], ...
        'EdgeColor', 'green', 'BackgroundColor', 'white', ...
        'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    
    % Print final summary
    fprintf('\n=== FINAL POINT SELECTION SUMMARY ===\n');
    fprintf('Successfully selected 4 point pairs:\n');
    for i = 1:length(x1)
        fprintf('  Pair %d: (%.1f, %.1f) <-> (%.1f, %.1f)\n', ...
                i, x1(i), y1(i), x2(i), y2(i));
    end
    fprintf('Points are ready for registration!\n');
end

%% --- Alternating Point Selection with Zoom Support ---
function [x1, y1, x2, y2] = alternatingPointSelectionWithZoom(gray1, gray2, figHandle, showDebug)
    x1 = [];
    y1 = [];
    x2 = [];
    y2 = [];
    
    % Colors for different point pairs
    colors = {'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black'};
    
    % Target number of point pairs
    targetPairs = 4;
    
    % Instructions for user
    fprintf('\n=== MANUAL POINT SELECTION ===\n');
    fprintf('Use mouse wheel to zoom in/out, click+drag to pan\n');
    fprintf('Double-click to zoom to fit\n');
    fprintf('Select %d point pairs by alternating between images\n\n', targetPairs);
    
    try
        for pairIdx = 1:targetPairs
            if showDebug
                fprintf('Selecting point pair %d/%d...\n', pairIdx, targetPairs);
            end
            
            % Get color for this pair
            currentColor = colors{mod(pairIdx-1, length(colors)) + 1};
            
            % =================== SELECT POINT ON REFERENCE IMAGE ===================
            figure(figHandle);
            subplot(1, 2, 1);
            xlabel(sprintf('ZOOM enabled: Click point #%d here (pair %d/%d)', pairIdx, pairIdx, targetPairs), ...
                   'FontSize', 11, 'Color', currentColor, 'FontWeight', 'bold');
            
            % Ensure we're in the correct subplot and wait for click
            subplot(1, 2, 1);
            
            % Wait for user click with enhanced feedback
            fprintf('  -> Click on point #%d in the LEFT (reference) image...\n', pairIdx);
            
            % Custom point selection that works with zoom
            [xi1, yi1, success1] = selectPointWithZoom(currentColor, pairIdx, 'reference');
            
            if ~success1
                if showDebug
                    fprintf('  Point selection cancelled on reference image\n');
                end
                return;
            end
            
            % Show selected point on reference image
            hold on;
            plot(xi1, yi1, 'o', 'Color', currentColor, 'MarkerSize', 14, 'LineWidth', 3);
            plot(xi1, yi1, '+', 'Color', 'white', 'MarkerSize', 12, 'LineWidth', 2);
            text(xi1+8, yi1-8, num2str(pairIdx), 'Color', currentColor, ...
                 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');
            
            % =================== SELECT CORRESPONDING POINT ON INPUT IMAGE ===================
            subplot(1, 2, 2);
            xlabel(sprintf('ZOOM enabled: Click corresponding point #%d here', pairIdx), ...
                   'FontSize', 11, 'Color', currentColor, 'FontWeight', 'bold');
            
            % Ensure we're in the correct subplot
            subplot(1, 2, 2);
            
            fprintf('  -> Click on corresponding point #%d in the RIGHT (input) image...\n', pairIdx);
            
            % Custom point selection that works with zoom
            [xi2, yi2, success2] = selectPointWithZoom(currentColor, pairIdx, 'input');
            
            if ~success2
                if showDebug
                    fprintf('  Point selection cancelled on input image\n');
                end
                return;
            end
            
            % Show selected point on input image
            hold on;
            plot(xi2, yi2, 'o', 'Color', currentColor, 'MarkerSize', 14, 'LineWidth', 3);
            plot(xi2, yi2, '+', 'Color', 'white', 'MarkerSize', 12, 'LineWidth', 2);
            text(xi2+8, yi2-8, num2str(pairIdx), 'Color', currentColor, ...
                 'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');
            
            % Store the point pair
            x1(pairIdx) = xi1;
            y1(pairIdx) = yi1;
            x2(pairIdx) = xi2;
            y2(pairIdx) = yi2;
            
            % Update progress
            if pairIdx < targetPairs
                fprintf('  ✓ Point pair %d/%d completed. Continue with next pair...\n', pairIdx, targetPairs);
                
                % Update instructions
                subplot(1, 2, 1);
                xlabel(sprintf('Point pair %d/%d completed. Next: click point #%d', ...
                       pairIdx, targetPairs, pairIdx+1), ...
                       'FontSize', 10, 'Color', 'green');
                subplot(1, 2, 2);
                xlabel('Wait for next point selection...', 'FontSize', 10, 'Color', 'gray');
            else
                fprintf('  ✓ All %d point pairs completed successfully!\n', targetPairs);
                
                % Final success message
                subplot(1, 2, 1);
                xlabel('✓ All 4 point pairs completed!', 'FontSize', 11, 'Color', 'green', 'FontWeight', 'bold');
                subplot(1, 2, 2);
                xlabel('✓ All 4 point pairs completed!', 'FontSize', 11, 'Color', 'green', 'FontWeight', 'bold');
            end
            
            % Update main title
            sgtitle(sprintf('Manual Point Selection - %d/%d point pairs completed (zoom enabled)', ...
                    pairIdx, targetPairs), 'FontSize', 14, 'FontWeight', 'bold');
            
            drawnow;
            pause(0.5); % Brief pause for visual feedback
        end
        
        fprintf('\n✓ Successfully selected %d point pairs for registration!\n', targetPairs);
        
    catch ME
        if showDebug
            warning('Alternating point selection failed: %s', ME.message);
        end
        x1 = [];
        y1 = [];
        x2 = [];
        y2 = [];
    end
end

%% --- Enhanced Point Selection with Zoom Support ---
function [x, y, success] = selectPointWithZoom(color, pointNum, imageType)
    x = [];
    y = [];
    success = false;
    
    try
        % Wait for user click
        [x, y, button] = ginput(1);
        
        % Check if user cancelled (ESC, close window, etc.)
        if isempty(x) || isempty(y)
            return;
        end
        
        % Check if it's a valid left click
        if button == 1  % Left mouse button
            success = true;
            fprintf('    Point #%d selected at (%.1f, %.1f) on %s image\n', ...
                    pointNum, x, y, imageType);
        else
            fprintf('    Invalid click (use left mouse button)\n');
        end
        
    catch ME
        fprintf('    Point selection cancelled or failed: %s\n', ME.message);
        success = false;
    end
end

%% --- Show Final Point Selection with Zoom Maintained ---
function showFinalPointSelectionWithZoom(gray1, gray2, x1, y1, x2, y2, figHandle)
    % Update the display to show final results while maintaining zoom
    figure(figHandle);
    
    % Colors for different points
    colors = {'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black'};
    
    % Reference image with final points
    subplot(1, 2, 1);
    % Don't reset the image - just update the display
    hold on;
    
    % Clear any previous point markers and redraw all points
    for i = 1:length(x1)
        currentColor = colors{mod(i-1, length(colors)) + 1};
        plot(x1(i), y1(i), 'o', 'Color', currentColor, 'MarkerSize', 14, 'LineWidth', 3);
        plot(x1(i), y1(i), '+', 'Color', 'white', 'MarkerSize', 12, 'LineWidth', 2);
        text(x1(i)+10, y1(i)-10, num2str(i), 'Color', currentColor, ...
             'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');
    end
    title('Reference Image - Final Points', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('✓ All points selected (zoom still active)', 'FontSize', 10, 'Color', 'green', 'FontWeight', 'bold');
    
    % Input image with final points
    subplot(1, 2, 2);
    hold on;
    
    for i = 1:length(x2)
        currentColor = colors{mod(i-1, length(colors)) + 1};
        plot(x2(i), y2(i), 'o', 'Color', currentColor, 'MarkerSize', 14, 'LineWidth', 3);
        plot(x2(i), y2(i), '+', 'Color', 'white', 'MarkerSize', 12, 'LineWidth', 2);
        text(x2(i)+10, y2(i)-10, num2str(i), 'Color', currentColor, ...
             'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'white');
    end
    title('Input Image - Final Points', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('✓ All points selected (zoom still active)', 'FontSize', 10, 'Color', 'green', 'FontWeight', 'bold');
    
    % Update main title
    sgtitle('✓ Manual Point Selection Complete - 4 point pairs selected', ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', 'green');
    
    % Add enhanced success message
    annotation('textbox', [0.02, 0.02, 0.96, 0.15], ...
        'String', ['✓ SUCCESS: 4 corresponding point pairs selected and will be used for registration' char(10) ...
                  'Point pairs are color-coded and numbered for verification' char(10) ...
                  'Zoom functionality remains active for verification'], ...
        'EdgeColor', 'green', 'BackgroundColor', 'white', ...
        'FontSize', 11, 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    
    % Print final summary
    fprintf('\n=== FINAL POINT SELECTION SUMMARY ===\n');
    fprintf('Successfully selected 4 point pairs:\n');
    for i = 1:length(x1)
        fprintf('  Pair %d: (%.1f, %.1f) <-> (%.1f, %.1f)\n', ...
                i, x1(i), y1(i), x2(i), y2(i));
    end
    fprintf('Points are ready for registration!\n');
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

    % img = imgaussfilt(img, 0.1); % 0.3
    img = imbilatfilt(img);

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
    end

    % img = imsharpen(img, 'Radius', 1, 'Amount', 0.2);
    % 
    % if showFigure
    %     subplot(2, 3, 6);
    %     imshow(img);
    %     title('Integral');
    % end

    grayImage = img;
end

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

%% Copilot
%% --- Helper function to filter points near edges ---
function filteredPts = filterPointsNearEdges(pts, edgeImg, maxDistance)
    if pts.Count == 0
        filteredPts = pts;
        return;
    end
    
    % Find edge pixels
    [edgeRows, edgeCols] = find(edgeImg);
    
    % For each point, find distance to nearest edge
    validIdx = false(pts.Count, 1);
    
    for i = 1:pts.Count
        pointLoc = pts.Location(i, :);
        
        % Calculate distances to all edge pixels
        distances = sqrt((edgeRows - pointLoc(2)).^2 + (edgeCols - pointLoc(1)).^2);
        
        % Keep point if it's close to an edge
        if min(distances) <= maxDistance
            validIdx(i) = true;
        end
    end
    
    % Create filtered cornerPoints object
    if any(validIdx)
        filteredPts = cornerPoints(pts.Location(validIdx, :), ...
            'Metric', pts.Metric(validIdx));
    else
        filteredPts = cornerPoints.empty();
    end
end

%% --- Fixed parameter function with odd FilterSize values ---
function [surfParams, cornerParams] = getSceneParametersWithCorners(sceneType)
    % Get SURF parameters
    surfParams = struct();
    
    % Get corner detection parameters
    cornerParams = struct();
    
    switch lower(sceneType)
        case 'city'
            % SURF parameters for city scenes
            surfParams.threshold = 800;
            surfParams.octaves = 4;
            surfParams.maxRatio = 0.7;
            
            % Corner parameters for city scenes (good for buildings)
            cornerParams.cornerQuality = 0.015;
            cornerParams.cornerFilterSize = 5;
            cornerParams.maxRatio = 0.7;
            
        case 'water'
            % SURF parameters for water scenes
            surfParams.threshold = 300;
            surfParams.octaves = 5;
            surfParams.maxRatio = 0.8;
            
            % Corner parameters for water scenes (more sensitive)
            cornerParams.cornerQuality = 0.001;
            cornerParams.cornerFilterSize = 1;
            cornerParams.maxRatio = 0.8;
            
        case 'nature'
            % SURF parameters for nature scenes
            surfParams.threshold = 500;
            surfParams.octaves = 4;
            surfParams.maxRatio = 0.75;
            
            % Corner parameters for nature scenes (balanced)
            cornerParams.cornerQuality = 0.005;
            cornerParams.cornerFilterSize = 3;
            cornerParams.maxRatio = 0.75;
            
        otherwise
            % Default parameters
            surfParams.threshold = 600;
            surfParams.octaves = 4;
            surfParams.maxRatio = 0.7;
            
            cornerParams.cornerQuality = 0.015;
            cornerParams.cornerFilterSize = 5; 
    end
end

%% --- Robust feature detection with error handling ---
function [matched1, matched2, featureStats] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped, showDebug)
    % Initialize outputs
    matched1 = [];
    matched2 = [];
    featureStats = struct();
    
    try
        % Analyze scene content to determine optimal parameters
        sceneType1 = 'nature';
        sceneType2 = 'nature';
        
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
        
        % Get adaptive parameters with validation
        [surfParams, cornerParams] = getSceneParametersWithCorners(sceneType);
        
        % SURF detection with integral image optimization
        if showDebug
            fprintf('  Computing SURF features with integral image optimization...\n');
        end
        
        integral1 = integralImage(gray1Cropped);
        integral2 = integralImage(gray2Cropped);

        pts1SURF = detectSURFFeatures(gray1Cropped, ...
            'MetricThreshold', surfParams.threshold, ...
            'NumOctaves', surfParams.octaves);
        pts2SURF = detectSURFFeatures(gray2Cropped, ...
            'MetricThreshold', surfParams.threshold, ...
            'NumOctaves', surfParams.octaves);
        
        pts1Corner = detectHarrisFeatures(gray1Cropped, ...
            'MinQuality', cornerParams.cornerQuality, ...
            'FilterSize', cornerParams.cornerFilterSize);
        pts2Corner = detectHarrisFeatures(gray2Cropped, ...
            'MinQuality', cornerParams.cornerQuality, ...
            'FilterSize', cornerParams.cornerFilterSize);


        % Extract features from all methods
        [f1SURF, vpts1SURF] = extractFeatures(gray1Cropped, pts1SURF);
        [f2SURF, vpts2SURF] = extractFeatures(gray2Cropped, pts2SURF);

        [f1Corner, vpts1Corner] = extractFeatures(gray1Cropped, pts1Corner);
        [f2Corner, vpts2Corner] = extractFeatures(gray2Cropped, pts2Corner);

        % Match features from each method
        indexPairsSURF = matchFeatures(f1SURF, f2SURF, ...
            'Unique', true, 'MaxRatio', surfParams.maxRatio);
        
        indexPairsCorner = matchFeatures(f1Corner, f2Corner, ...
            'Unique', true, 'MaxRatio', cornerParams.maxRatio);

        % Combine matches from all methods
        if ~isempty(indexPairsSURF)
            matched1 = [matched1; vpts1SURF(indexPairsSURF(:,1)).Location];
            matched2 = [matched2; vpts2SURF(indexPairsSURF(:,2)).Location];
            surfMatches = size(indexPairsSURF, 1);
        end
        
        if ~isempty(indexPairsCorner)
            matched1 = [matched1; vpts1Corner(indexPairsCorner(:,1)).Location];
            matched2 = [matched2; vpts2Corner(indexPairsCorner(:,2)).Location];
            cornerMatches = size(indexPairsCorner, 1);
        end
        
    catch ME
        if showDebug
            warning('Feature detection failed: %s', ME.message);
        end
        
        % Fallback to simple SURF detection
        try
            pts1 = detectSURFFeatures(gray1Cropped, 'MetricThreshold', 500);
            pts2 = detectSURFFeatures(gray2Cropped, 'MetricThreshold', 500);
            
            [f1, vpts1] = extractFeatures(gray1Cropped, pts1);
            [f2, vpts2] = extractFeatures(gray2Cropped, pts2);
            
            indexPairs = matchFeatures(f1, f2, 'Unique', true, 'MaxRatio', 0.7);
            
            if ~isempty(indexPairs)
                matched1 = vpts1(indexPairs(:,1)).Location;
                matched2 = vpts2(indexPairs(:,2)).Location;
            end
            
            featureStats.fallback = true;
            
        catch ME2
            if showDebug
                warning('Fallback feature detection also failed: %s', ME2.message);
            end
        end
    end
end