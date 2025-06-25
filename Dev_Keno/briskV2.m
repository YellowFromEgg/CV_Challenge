function briskV2()
    close all;
    
    % Enable parallel processing if available
    if ~isempty(gcp('nocreate'))
        useParallel = true;
        fprintf('Parallel processing enabled.\n');
    else
        useParallel = false;
        fprintf('Sequential processing mode.\n');
    end
    
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

    %% 3. Prepare reference image
    fprintf('Processing reference image: %s\n', imageFiles{1});
    refImg = imread(fullfile(imageFolder, imageFiles{1}));
    refGray = preprocessImage(refImg, true); % Show preprocessing for reference
    refSize = size(refGray);

    % Initialize cumulative difference with single precision for memory efficiency
    cumulativeDiff = zeros(refSize, 'single');
    totalMask = true(refSize);

    %% 4. Process remaining images
    skippedImages = {};
    
    for i = 2:numImages
        fprintf('Processing image %d/%d: %s\n', i, numImages, imageFiles{i});
        
        img = imread(fullfile(imageFolder, imageFiles{i}));
        gray = preprocessImage(img, false); % Don't show preprocessing for each image
        
        % Registration
        [registered, validMask] = registerImages(refGray, gray, imageFiles{i});

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
        
        fprintf('  -> Successfully registered and processed\n');
    end

    if ~isempty(skippedImages)
        fprintf('\nSkipped images due to registration failures:\n');
        for i = 1:length(skippedImages)
            fprintf('  - %s\n', skippedImages{i});
        end
    end

    %% 5. Visualize cumulative difference
    cumulativeDiff(~totalMask) = 0;
    visualizeDifferenceHeatmap(cumulativeDiff, totalMask);
    
    fprintf('\nProcessing complete!\n');
end

%% --- Enhanced Preprocessing: Noise reduction, normalization, atmospheric correction ---
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
    
    % Convert to grayscale if needed
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Convert to double early for better precision
    img = im2double(img);
    
    if showFigure
        subplot(2, 3, 2);
        imshow(img);
        title('Grayscale');
    end
    
    % 1. Gentle noise reduction (reduced from 0.5)
    img = imgaussfilt(img, 0.3);
    
    if showFigure
        subplot(2, 3, 3);
        imshow(img);
        title('Noise Reduced');
    end
    
    % 2. More gentle histogram equalization
    img = adapthisteq(img, 'ClipLimit', 0.01, 'NumTiles', [6 6]);  % Reduced aggressiveness
    
    if showFigure
        subplot(2, 3, 4);
        imshow(img);
        title('Histogram Equalized');
    end
    
    % 3. Simpler normalization to preserve more original characteristics
    img = mat2gray(img);  % Simple min-max normalization
    
    if showFigure
        subplot(2, 3, 5);
        imshow(img);
        title('Normalized');
    end
    
    % 4. Skip aggressive atmospheric correction for better feature preservation
    % img = imadjust(img, [0.1 0.9], [0 1], 0.8);  % Commented out
    
    if showFigure
        subplot(2, 3, 6);
        imshow(img);
        title('Final Result');
    end
    
    grayImage = img;
end

%% --- Enhanced Registration with multiple feature types and robust estimation ---
function [registered2, validMask] = registerImages(gray1, gray2, imageName)
    if nargin < 3
        imageName = 'Current Image';
    end
    
    try
        % More lenient BRISK detection
        pts1BRISK = detectBRISKFeatures(gray1, ...
            'MinContrast', 0.02, ...
            'MinQuality', 0.05, ...
            'NumOctaves', 5);
        pts2BRISK = detectBRISKFeatures(gray2, ...
            'MinContrast', 0.02, ...
            'MinQuality', 0.05, ...
            'NumOctaves', 5);

        % Extract BRISK features
        [f1BRISK, vpts1BRISK] = extractFeatures(gray1, pts1BRISK, ...
            'Method', 'BRISK', 'Upright', false);
        [f2BRISK, vpts2BRISK] = extractFeatures(gray2, pts2BRISK, ...
            'Method', 'BRISK', 'Upright', false);
        
        % More lenient BRISK matching
        indexPairsBRISK = matchFeatures(f1BRISK, f2BRISK, ...
            'MatchThreshold', 15, ...
            'MaxRatio', 0.8, ...
            'Unique', true);

        % More lenient KAZE detection
        pts1KAZE = detectKAZEFeatures(gray1, 'Threshold', 0.001);
        pts2KAZE = detectKAZEFeatures(gray2, 'Threshold', 0.001);
        
        [f1KAZE, vpts1KAZE] = extractFeatures(gray1, pts1KAZE);
        [f2KAZE, vpts2KAZE] = extractFeatures(gray2, pts2KAZE);
        
        indexPairsKAZE = matchFeatures(f1KAZE, f2KAZE, ...
            'MatchThreshold', 15, ...
            'MaxRatio', 0.8, ...
            'Unique', true);

        % Keep original SURF parameters (they worked well)
        pts1SURF = detectSURFFeatures(gray1, 'MetricThreshold', 500, 'NumOctaves', 6);
        pts2SURF = detectSURFFeatures(gray2, 'MetricThreshold', 500, 'NumOctaves', 6);

        [f1SURF, vpts1SURF] = extractFeatures(gray1, pts1SURF);
        [f2SURF, vpts2SURF] = extractFeatures(gray2, pts2SURF);

        indexPairsSURF = matchFeatures(f1SURF, f2SURF, 'Unique', true, 'MaxRatio', 0.8);

        % Collect all matches
        matched1 = [];
        matched2 = [];
        
        % Prioritize SURF features (your original successful approach)
        if ~isempty(indexPairsSURF)
            matched1SURF = vpts1SURF(indexPairsSURF(:,1)).Location;
            matched2SURF = vpts2SURF(indexPairsSURF(:,2)).Location;
            matched1 = [matched1; matched1SURF];
            matched2 = [matched2; matched2SURF];
            fprintf('  SURF matches: %d\n', size(matched1SURF, 1));
        end
        
        % Add BRISK matches
        if ~isempty(indexPairsBRISK)
            matched1BRISK = vpts1BRISK(indexPairsBRISK(:,1)).Location;
            matched2BRISK = vpts2BRISK(indexPairsBRISK(:,2)).Location;
            matched1 = [matched1; matched1BRISK];
            matched2 = [matched2; matched2BRISK];
            fprintf('  BRISK matches: %d\n', size(matched1BRISK, 1));
        end
        
        % Add KAZE matches if we need more
        if ~isempty(indexPairsKAZE)
            matched1KAZE = vpts1KAZE(indexPairsKAZE(:,1)).Location;
            matched2KAZE = vpts2KAZE(indexPairsKAZE(:,2)).Location;
            matched1 = [matched1; matched1KAZE];
            matched2 = [matched2; matched2KAZE];
            fprintf('  KAZE matches: %d\n', size(matched1KAZE, 1));
        end

        if size(matched1, 1) < 3
            warning('Insufficient feature matches for registration (%d found).', size(matched1, 1));
            registered2 = [];
            validMask = [];
            return;
        end

        fprintf('  Total matches: %d\n', size(matched1, 1));

        % Enhanced RANSAC with progressive relaxation
        tform = [];
        inlierIdx = [];
        bestInlierCount = 0;
        bestTransformType = '';
        
        % Define transformation strategies with increasing flexibility
        strategies = {
            struct('type', 'similarity', 'trials', 5000, 'confidence', 99, 'distance', 1.5), ...
            struct('type', 'similarity', 'trials', 8000, 'confidence', 95, 'distance', 2.0), ...
            struct('type', 'similarity', 'trials', 10000, 'confidence', 90, 'distance', 3.0), ...
            struct('type', 'affine', 'trials', 8000, 'confidence', 95, 'distance', 2.0), ...
            struct('type', 'affine', 'trials', 10000, 'confidence', 90, 'distance', 3.0), ...
            struct('type', 'projective', 'trials', 8000, 'confidence', 90, 'distance', 3.0)
        };
        
        for s = 1:length(strategies)
            strategy = strategies{s};
            
            try
                % Use distance threshold if available (newer MATLAB versions)
                if exist('estgeotform2d', 'file') == 2
                    try
                        % Try with MaxDistance parameter (newer versions)
                        [tempTform, tempInlierIdx] = estgeotform2d(matched2, matched1, ...
                            strategy.type, ...
                            'MaxNumTrials', strategy.trials, ...
                            'Confidence', strategy.confidence, ...
                            'MaxDistance', strategy.distance);
                    catch
                        % Fallback without MaxDistance parameter
                        [tempTform, tempInlierIdx] = estgeotform2d(matched2, matched1, ...
                            strategy.type, ...
                            'MaxNumTrials', strategy.trials, ...
                            'Confidence', strategy.confidence);
                    end
                else
                    % Fallback to older function
                    [tempTform, tempInlierIdx] = estimateGeometricTransform(matched2, matched1, ...
                        strategy.type, ...
                        'MaxNumTrials', strategy.trials, ...
                        'Confidence', strategy.confidence);
                end
                
                inlierCount = sum(tempInlierIdx);
                inlierRatio = inlierCount / length(tempInlierIdx);
                
                fprintf('  %s: %d inliers (%.1f%%)\n', strategy.type, inlierCount, inlierRatio*100);
                
                % Accept if we have enough inliers
                if inlierCount >= 3 && inlierCount > bestInlierCount
                    tform = tempTform;
                    inlierIdx = tempInlierIdx;
                    bestInlierCount = inlierCount;
                    bestTransformType = strategy.type;
                    
                    % If we have a good solution, don't try more complex transforms
                    if inlierCount >= 6 && inlierRatio >= 0.3
                        fprintf('  Good solution found, stopping search\n');
                        break;
                    end
                end
                
            catch ME
                fprintf('  %s failed: %s\n', strategy.type, ME.message);
                continue;
            end
        end
        
        if isempty(tform)
            % Last resort: try with original approach (your working version)
            try
                fprintf('  Trying fallback to original approach...\n');
                if ~isempty(indexPairsSURF) && size(indexPairsSURF, 1) >= 3
                    [tform, inlierIdx] = estgeotform2d(...
                        vpts2SURF(indexPairsSURF(:,2)).Location, ...
                        vpts1SURF(indexPairsSURF(:,1)).Location, ...
                        'similarity', 'MaxNumTrials', 2000, 'Confidence', 85);
                    matched1 = vpts1SURF(indexPairsSURF(:,1)).Location;
                    matched2 = vpts2SURF(indexPairsSURF(:,2)).Location;
                    bestTransformType = 'similarity (fallback)';
                    fprintf('  Fallback successful: %d inliers\n', sum(inlierIdx));
                end
            catch
                warning('All registration approaches failed.');
                registered2 = [];
                validMask = [];
                return;
            end
        end
        
        if isempty(tform)
            warning('No valid transformation found after all attempts.');
            registered2 = [];
            validMask = [];
            return;
        end
        
        % Final validation
        finalInlierCount = sum(inlierIdx);
        finalInlierRatio = finalInlierCount / length(inlierIdx);
        
        fprintf('  Final result (%s): %d inliers (%.1f%%)\n', ...
            bestTransformType, finalInlierCount, finalInlierRatio*100);
        
        % Very lenient acceptance criteria
        if finalInlierCount < 3
            warning('Insufficient final inliers (%d).', finalInlierCount);
            registered2 = [];
            validMask = [];
            return;
        end

        % Apply transformation with error handling
        try
            outputRef = imref2d(size(gray1));
            registered2 = imwarp(gray2, tform, 'OutputView', outputRef, ...
                'Interp', 'linear', 'FillValues', 0);

            % Create validity mask
            mask = ones(size(gray2));
            warpedMask = imwarp(mask, tform, 'OutputView', outputRef);
            validMask = warpedMask > 0.3;  % Very lenient threshold

            % Basic sanity check
            if sum(validMask(:)) < 0.1 * numel(validMask)
                warning('Registration resulted in very small valid area.');
                registered2 = [];
                validMask = [];
                return;
            end

        catch ME
            warning('Image warping failed: %s', ME.message);
            registered2 = [];
            validMask = [];
            return;
        end

        % Only show debug for very poor registrations
        if finalInlierCount < 5 || finalInlierRatio < 0.2
            figure('Name', ['Registration Debug: ', imageName], 'NumberTitle', 'off');
            subplot(2,2,1);
            showMatchedFeatures(gray1, gray2, matched1(inlierIdx,:), matched2(inlierIdx,:), 'montage');
            title(sprintf('Inliers: %d (%.1f%%)', finalInlierCount, finalInlierRatio*100));
            
            subplot(2,2,2); imshow(gray1); title('Reference');
            subplot(2,2,3); imshow(registered2); title('Registered');
            subplot(2,2,4); imshowpair(gray1, registered2); title('Overlay');
        end

    catch ME
        warning('Registration completely failed for %s: %s', imageName, ME.message);
        registered2 = [];
        validMask = [];
    end
end

%% --- Enhanced visualization with statistics ---
function visualizeDifferenceHeatmap(diffImage, validMask)
    % Calculate statistics
    validDiff = diffImage(validMask);
    meanDiff = mean(validDiff);
    stdDiff = std(validDiff);
    maxDiff = max(validDiff);
    
    fprintf('\nDifference Statistics:\n');
    fprintf('  Mean: %.4f\n', meanDiff);
    fprintf('  Std:  %.4f\n', stdDiff);
    fprintf('  Max:  %.4f\n', maxDiff);
    
    % Create enhanced visualization
    figure('Name', 'Change Detection Results', 'NumberTitle', 'off');
    
    % Original heatmap
    subplot(2,2,1);
    imagesc(diffImage);
    axis image off;
    colormap(gca, jet);
    colorbar;
    title('Raw Difference Heatmap');
    
    % Thresholded version
    subplot(2,2,2);
    threshold = meanDiff + 2*stdDiff;
    thresholdedDiff = diffImage;
    thresholdedDiff(diffImage < threshold) = 0;
    imagesc(thresholdedDiff);
    axis image off;
    colormap(gca, hot);
    colorbar;
    title(sprintf('Significant Changes (>%.3f)', threshold));
    
    % Histogram of differences
    subplot(2,2,3);
    histogram(validDiff, 50, 'Normalization', 'probability');
    hold on;
    xline(meanDiff, 'r--', 'LineWidth', 2, 'Label', 'Mean');
    xline(threshold, 'g--', 'LineWidth', 2, 'Label', 'Threshold');
    xlabel('Difference Value');
    ylabel('Probability');
    title('Distribution of Differences');
    grid on;
    
    % Valid area mask
    subplot(2,2,4);
    imshow(validMask);
    title('Valid Registration Area');
    
    % Overall statistics
    validPixels = sum(validMask(:));
    totalPixels = numel(validMask);
    validPercent = 100 * validPixels / totalPixels;
    
    sgtitle(sprintf('Change Detection Results - %.1f%% valid area', validPercent));
end

%% --- Natural sorting with improved year extraction ---
function sorted = sort_nat(filenames)
    % Extract years using more flexible pattern
    expr = '\d{4}';
    years = regexp(filenames, expr, 'match', 'once');
    
    % Handle files without year information
    validYears = ~cellfun(@isempty, years);
    if sum(validYears) == 0
        % Fallback to alphabetical sorting
        sorted = sort(filenames);
        return;
    end
    
    % Convert to numbers and sort
    years = cellfun(@str2double, years);
    [~, idx] = sort(years);
    sorted = filenames(idx);
end