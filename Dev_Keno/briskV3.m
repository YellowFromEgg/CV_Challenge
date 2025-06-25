function briskV3()
    close all;
    
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

    %% 3. Find optimal reference image
    fprintf('\n=== Finding optimal reference image ===\n');
    refIdx = findOptimalReference(imageFolder, imageFiles);
    fprintf('Selected reference image: %s (index %d)\n', imageFiles{refIdx}, refIdx);
    
    % Load and preprocess reference image
    refImg = imread(fullfile(imageFolder, imageFiles{refIdx}));
    refGray = preprocessImage(refImg, true);
    refSize = size(refGray);

    % Initialize cumulative difference
    cumulativeDiff = zeros(refSize, 'single');
    totalMask = true(refSize);

    %% 4. Process all other images against the optimal reference
    fprintf('\n=== Processing images against optimal reference ===\n');
    skippedImages = {};
    processedCount = 0;
    
    for i = 1:numImages
        if i == refIdx
            continue; % Skip the reference image itself
        end
        
        fprintf('Processing image %d/%d: %s\n', i, numImages, imageFiles{i});
        
        img = imread(fullfile(imageFolder, imageFiles{i}));
        gray = preprocessImage(img, false);
        
        % Registration with debug output
        [registered, validMask] = registerImages(refGray, gray, imageFiles{i}, true); % Enable debug

        if isempty(registered)
            skippedImages{end+1} = imageFiles{i};
            fprintf('  -> Skipped due to registration failure\n');
            continue;
        end

        % Calculate difference
        diffImage = imabsdiff(refGray, registered);
        diffImage(~validMask) = 0;

        % Accumulate
        cumulativeDiff = cumulativeDiff + single(diffImage);
        totalMask = totalMask & validMask;
        processedCount = processedCount + 1;
        
        fprintf('  -> Successfully registered and processed\n');
    end

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
    
    fprintf('\nProcessing complete!\n');
end

%% --- Find optimal reference image based on feature matching success ---
function optimalIdx = findOptimalReference(imageFolder, imageFiles)
    numImages = length(imageFiles);
    
    % For large datasets, sample a subset for efficiency
    if numImages > 20
        sampleIndices = round(linspace(1, numImages, min(15, numImages)));
        fprintf('Sampling %d images for reference selection\n', length(sampleIndices));
    else
        sampleIndices = 1:numImages;
    end
    
    % Preprocess all sampled images
    fprintf('Preprocessing images for reference selection...\n');
    images = cell(length(sampleIndices), 1);
    for i = 1:length(sampleIndices)
        idx = sampleIndices(i);
        img = imread(fullfile(imageFolder, imageFiles{idx}));
        images{i} = preprocessImage(img, false);
        if mod(i, 5) == 0
            fprintf('  Processed %d/%d images\n', i, length(sampleIndices));
        end
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
        
        % Bonus for images in the middle of the time series (often more stable)
        temporalPosition = sampleIndices(i) / numImages;
        temporalBonus = 1 - abs(temporalPosition - 0.5); % Peak at 0.5 (middle)
        
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
        % Quick feature detection and matching
        pts1 = detectSURFFeatures(gray1, 'MetricThreshold', 1000);
        pts2 = detectSURFFeatures(gray2, 'MetricThreshold', 1000);
        
        if pts1.Count < 10 || pts2.Count < 10
            return;
        end
        
        [f1, vpts1] = extractFeatures(gray1, pts1);
        [f2, vpts2] = extractFeatures(gray2, pts2);
        
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

%% --- Enhanced registration function with comprehensive debug output ---
function [registered2, validMask] = registerImages(gray1, gray2, imageName, showDebug)
    if nargin < 3
        imageName = 'Current Image';
    end
    if nargin < 4
        showDebug = false;
    end
    
    try
        % Feature detection with multiple methods
        pts1SURF = detectSURFFeatures(gray1, 'MetricThreshold', 500, 'NumOctaves', 6);
        pts2SURF = detectSURFFeatures(gray2, 'MetricThreshold', 500, 'NumOctaves', 6);

        [f1SURF, vpts1SURF] = extractFeatures(gray1, pts1SURF);
        [f2SURF, vpts2SURF] = extractFeatures(gray2, pts2SURF);

        indexPairsSURF = matchFeatures(f1SURF, f2SURF, 'Unique', true, 'MaxRatio', 0.8);

        % Add BRISK for robustness
        pts1BRISK = detectBRISKFeatures(gray1, 'MinContrast', 0.02, 'NumOctaves', 5);
        pts2BRISK = detectBRISKFeatures(gray2, 'MinContrast', 0.02, 'NumOctaves', 5);

        [f1BRISK, vpts1BRISK] = extractFeatures(gray1, pts1BRISK, 'Method', 'BRISK');
        [f2BRISK, vpts2BRISK] = extractFeatures(gray2, pts2BRISK, 'Method', 'BRISK');
        
        indexPairsBRISK = matchFeatures(f1BRISK, f2BRISK, ...
            'MatchThreshold', 15, 'MaxRatio', 0.8, 'Unique', true);

        % Combine matches
        matched1 = [];
        matched2 = [];
        
        if ~isempty(indexPairsSURF)
            matched1SURF = vpts1SURF(indexPairsSURF(:,1)).Location;
            matched2SURF = vpts2SURF(indexPairsSURF(:,2)).Location;
            matched1 = [matched1; matched1SURF];
            matched2 = [matched2; matched2SURF];
            fprintf('  SURF matches: %d\n', size(matched1SURF, 1));
        end
        
        if ~isempty(indexPairsBRISK)
            matched1BRISK = vpts1BRISK(indexPairsBRISK(:,1)).Location;
            matched2BRISK = vpts2BRISK(indexPairsBRISK(:,2)).Location;
            matched1 = [matched1; matched1BRISK];
            matched2 = [matched2; matched2BRISK];
            fprintf('  BRISK matches: %d\n', size(matched1BRISK, 1));
        end

        if size(matched1, 1) < 4
            warning('Insufficient matches (%d) for %s', size(matched1, 1), imageName);
            registered2 = [];
            validMask = [];
            return;
        end

        fprintf('  Total matches: %d\n', size(matched1, 1));

        % Robust transformation estimation
        [tform, inlierIdx] = estgeotform2d(matched2, matched1, 'similarity', ...
            'MaxNumTrials', 3000, 'Confidence', 90);
        
        inlierCount = sum(inlierIdx);
        inlierRatio = inlierCount / length(inlierIdx);
        
        fprintf('  Inliers: %d/%d (%.1f%%)\n', inlierCount, length(inlierIdx), inlierRatio*100);
        
        if inlierCount < 4
            warning('Too few inliers (%d) for %s', inlierCount, imageName);
            registered2 = [];
            validMask = [];
            return;
        end

        % Apply transformation
        outputRef = imref2d(size(gray1));
        registered2 = imwarp(gray2, tform, 'OutputView', outputRef, ...
            'Interp', 'linear', 'FillValues', 0);

        mask = ones(size(gray2));
        warpedMask = imwarp(mask, tform, 'OutputView', outputRef);
        validMask = warpedMask > 0.5;

        %% COMPREHENSIVE DEBUG OUTPUT - FIXED FOR COMPATIBILITY
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
            
            % Display metrics with compatible text properties
            metrics = {
                sprintf('Total Matches: %d', size(matched1, 1));
                sprintf('Inliers: %d (%.1f%%)', inlierCount, inlierRatio*100);
                sprintf('Valid Area: %.1f%%', validArea*100);
                sprintf('Avg Difference: %.4f', avgDiff);
                sprintf('Std Difference: %.4f', stdDiff);
                sprintf('SURF Features: %d', length(indexPairsSURF));
                sprintf('BRISK Features: %d', length(indexPairsBRISK));
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
            
            % Optional: Save debug figure
            % saveas(fig, fullfile(pwd, sprintf('debug_%s.png', regexprep(imageName, '[^\w]', '_'))));
        end

    catch ME
        warning('Registration failed for %s: %s', imageName, ME.message);
        registered2 = [];
        validMask = [];
    end
end

%% --- Enhanced preprocessing (same as before) ---
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