function [successfulImages, validImageIndices, refIdx, tformList] = Register_Color_Images(colorImages, numImages, scene)
    close all; % Close all figure windows
    warning('off', 'vision:ransac:maxTrialsReached'); % disable warnings that come from RANSAC

    %% 1. Preprocess ALL images once and cache them
    grayImages = cell(numImages, 1); % Initialize cell array to store grayscale versions
    
    for i = 1:numImages
        % Preprocess to grayscale (only show debug for first image)
        grayImages{i} = preprocessImage(colorImages{i}, scene); % Convert each color image to enhanced grayscale
    end

    %% 2. Find optimal reference using cached preprocessed images
    refIdx = findOptimalReference(grayImages, scene); % Determine which image works best as reference
    
    % Get reference images
    refImg = colorImages{refIdx}; % Original color reference image
    refGray = grayImages{refIdx}; % Preprocessed grayscale reference image

    %% 3. Process all other images against the optimal reference
    % Storage for registered color images
    registeredColorImages = cell(numImages, 1); % Store final registered color images
    tformList = cell(numImages, 1); % Store transformation matrices for each image
    validImageIndices = []; % Track which images were successfully registered
    
    for i = 1:numImages
        if i == refIdx
            % Store reference image
            registeredColorImages{i} = refImg; % Reference image doesn't need transformation
            validImageIndices = [validImageIndices, i]; % Add reference to valid list

            continue; % Skip the reference image itself
        end
        
        % Use cached preprocessed images
        img = colorImages{i}; % Current color image to register
        gray = grayImages{i}; % Current preprocessed grayscale image
        
        % Registration with color image support - also support matrices
        [registeredColor, tform] = registerImages(refGray, gray, img, scene); % Register current image to reference
        tformList{i} = tform; % Store transformation matrix        
        if isempty(registeredColor) % If registration failed
            continue; % Skip this image
        end
    
        % Calculate percentage of valid (non-black) content
        if size(registeredColor, 3) == 3
            % RGB: pixel is valid if any channel > 0
            validMask = any(registeredColor > 0, 3);
        else
            % Grayscale: pixel is valid if > 0
            validMask = registeredColor > 0;
        end
        
        validContentPercent = sum(validMask(:)) / numel(validMask) * 100;
        minValidContentThreshold = 30; % Minimum 30% valid content required
        
        if validContentPercent < minValidContentThreshold
            fprintf('Image %d: insufficient valid content (%.1f%% < %.1f%%) - skipping\n', ...
                    i, validContentPercent, minValidContentThreshold);
            continue; % Skip this image
        end
    
        % Store the registered color image
        registeredColorImages{i} = registeredColor; % Store successful registration
        validImageIndices = [validImageIndices, i]; % Add to valid list
    end

    % Create compact array with only successful images
    successfulImages = cell(length(validImageIndices), 1);
    for i = 1:length(validImageIndices)
        idx = validImageIndices(i);
        successfulImages{i} = registeredColorImages{idx};
    end

    % Apply common content mask to all registered images
    for i = 1:length(validImageIndices)
        successfulImages{i} = im2double(successfulImages{i});
    end
    successfulImages = applyCommonContentMask(successfulImages, validImageIndices);
    
    successfulImages = matchBrightnessAcrossImages(successfulImages);
end

%% -- Apply mask to keep only common content --
function maskedImages = applyCommonContentMask(registeredImages, validIndices)
    if isempty(validIndices) || length(validIndices) < 2
        maskedImages = registeredImages;
        return;
    end
    
    % Get dimensions
    [h, w, ~] = size(registeredImages{1});
    
    % Initialize common content mask (start with all pixels as valid)
    commonMask = true(h, w);
    
    % For each valid registered image, find areas with actual content
    for i = 1:length(validIndices)
        img = registeredImages{i};
        
        if isempty(img)
            continue;
        end
        
        % Create mask for valid pixels (not black/empty)
        if size(img, 3) == 3
            % RGB: pixel is valid if any channel > 0
            validPixelMask = any(img > 0, 3);
        else
            % Grayscale: pixel is valid if > 0
            validPixelMask = img > 0;
        end
        % Keep only areas that are valid in ALL images
        commonMask = commonMask & validPixelMask;
    end
    
    % Apply common mask to all registered images
    maskedImages = registeredImages;
    
    for i = 1:length(validIndices)
        img = registeredImages{i};
        
        if isempty(img)
            continue;
        end
        
        % Apply mask to image
        if size(img, 3) == 3
            % RGB image
            for c = 1:3
                channel = img(:,:,c);
                channel(~commonMask) = 0; % Set non-common areas to black
                img(:,:,c) = channel;
            end
        else
            % Grayscale image
            img(~commonMask) = 0; % Set non-common areas to black
        end
        
        maskedImages{i} = img;
    end
    
    % Calculate and display statistics
    commonPercent = sum(commonMask(:)) / numel(commonMask) * 100;
end

%% --- Enhanced preprocessing ---
function grayImage = preprocessImage(img, scene)
    
    if size(img, 3) == 3 % Check if image has 3 channels (RGB)
        img = rgb2gray(img); % Convert RGB to grayscale
    end
    
    if strcmp(scene, 'city') % only apply mask when scene is city
        img = applyEntropyMask(img); % Apply same entropy-based mask to enhance features
    end
    
    img = im2double(img); % Convert to double precision for processing

    img = imbilatfilt(img); % Apply bilateral filter for noise reduction while preserving edges

    if strcmp(scene, 'nature') % only equalize when scene is nature
        img = adapthisteq(img, 'ClipLimit', 0.01, 'NumTiles', [6 6]); % Adaptive histogram equalization
    end

    img = mat2gray(img); % Normalize image to [0,1] range

    grayImage = img; % Return processed image
end

%% --- Normalize Registered Colored images ---
function matchedImages = matchBrightnessAcrossImages(colorImages)
    numImages = numel(colorImages);
    matchedImages = cell(size(colorImages));

    % Convert all to HSV
    hsvImages = cell(size(colorImages));
    for i = 1:numImages
        hsvImages{i} = rgb2hsv(colorImages{i});
    end

    % Choose a reference (e.g., middle image)
    refIdx2 = round(numImages / 2);
    refV = hsvImages{refIdx2}(:,:,3);  % Reference brightness channel

    % Apply histogram matching
    for i = 1:numImages
        hsv = hsvImages{i};
        hsv(:,:,3) = imhistmatch(hsv(:,:,3), refV);  % Match brightness
        matchedImages{i} = hsv2rgb(hsv);             % Convert back to RGB
    end
end

%% Create Mask
function maskedImage = applyEntropyMask(grayImg)
    entropyMap = entropyfilt(grayImg); % Calculate local entropy for each pixel
    entropyNorm = mat2gray(entropyMap); % Normalize entropy map to [0,1]

    threshold = graythresh(entropyNorm); % Calculate optimal threshold using Otsu's method
    entropyMask = entropyNorm > threshold; % Create binary mask of high-entropy regions

    se = strel('disk', 5); % Create disk-shaped structuring element
    dilatedMask = imdilate(entropyMask, se); % Dilate mask to fill small gaps

    if ndims(dilatedMask) > 2 % Check if mask has more than 2 dimensions
        dilatedMask = dilatedMask(:,:,1); % Take only first channel
    end

    if size(grayImg, 3) == 3 % Check if input is color image
        maskedImage = zeros(size(grayImg), 'like', grayImg); % Initialize output with same type as input
        for c = 1:3 % Process each color channel
            maskedImage(:,:,c) = grayImg(:,:,c) .* cast(dilatedMask, 'like', grayImg); % Apply mask to each channel
        end
    else
        maskedImage = grayImg .* cast(dilatedMask, 'like', grayImg); % Apply mask to grayscale image
    end
end

%% --- Find optimal reference image based on feature matching success ---
function optimalIdx = findOptimalReference(preprocessedImages, scene)
    numImages = length(preprocessedImages); % Get total number of images
    
    % For large datasets, sample a subset for efficiency
    if numImages > 20 % If dataset is large
        sampleIndices = round(linspace(1, numImages, min(15, numImages))); % Sample up to 15 images
        fprintf('Sampling %d images for reference selection\n', length(sampleIndices));
    else
        sampleIndices = 1:numImages; % Use all images if dataset is small
    end
    
    % Use cached preprocessed images
    images = cell(length(sampleIndices), 1); % Create cell array for sampled images
    for i = 1:length(sampleIndices)
        idx = sampleIndices(i); % Get original image index
        images{i} = preprocessedImages{idx}; % Copy preprocessed image to samples
    end
    
    % Calculate pairwise registration success matrix
    successMatrix = zeros(length(sampleIndices), length(sampleIndices)); % Track registration success between image pairs
    featureCountMatrix = zeros(length(sampleIndices), length(sampleIndices)); % Track feature counts between image pairs
    
    for i = 1:length(sampleIndices)
        for j = i+1:length(sampleIndices) % Only test upper triangle (symmetric matrix)
            [success, featureCount] = testRegistration(images{i}, images{j}, scene); % Test registration between image pair
            successMatrix(i, j) = success; % Store success result
            successMatrix(j, i) = success; % Mirror to lower triangle
            featureCountMatrix(i, j) = featureCount; % Store feature count
            featureCountMatrix(j, i) = featureCount; % Mirror to lower triangle
        end
    end
    
    % Calculate reference quality scores
    referenceScores = zeros(length(sampleIndices), 1); % Initialize scores for each candidate reference
    
    for i = 1:length(sampleIndices)
        % Success rate as reference
        successRate = mean(successMatrix(i, :)); % Average success rate when this image is used as reference
        
        % Average feature count when used as reference
        avgFeatureCount = mean(featureCountMatrix(i, :)); % Average number of features matched
        
        % Bonus for images in the middle of the time series
        temporalPosition = sampleIndices(i) / numImages; % Normalize position in sequence
        temporalBonus = 1 - abs(temporalPosition - 0.5); % Peak at 0.5 (middle of sequence)
        
        % Combined score
        referenceScores(i) = successRate * 0.6 + ... % Weight success rate highest
                           (avgFeatureCount / 100) * 0.3 + ... % Weight feature count
                           temporalBonus * 0.1; % Small bonus for temporal position
    end
    
    % Select the best reference
    [~, bestIdx] = max(referenceScores); % Find index of highest scoring reference
    optimalIdx = sampleIndices(bestIdx); % Convert back to original image index
end

%% --- Test registration between two images ---
function [success, featureCount] = testRegistration(gray1, gray2, scene)
    success = 0; % Initialize success flag
    featureCount = 0; % Initialize feature count
    
    try
        % Crop images to remove scale/logo area
        cropRatio = 0.07; % Remove bottom 7% of image
        gray1Cropped = cropImageForFeatureDetection(gray1, cropRatio); % Crop first image
        gray2Cropped = cropImageForFeatureDetection(gray2, cropRatio); % Crop second image

        % Quick feature detection and matching
        [matched1, matched2] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped, scene); % Find matching features

        [~, inlierIdx] = estgeotform2d(matched2, matched1, 'similarity', ... % Estimate transformation
                'MaxNumTrials', 300, 'Confidence', 90, 'MaxDistance', 5.0); % Using RANSAC
            
        featureCount = sum(inlierIdx); % Count inlier features

        if featureCount >= 5 % If enough good matches found
            success = 1; % Mark as successful
        end
        
    catch
        % Registration failed
        success = 0; % Mark as failed
    end
end

%% --- Create cropped images for feature detection
function croppedImage = cropImageForFeatureDetection(image, cropRatio)
    % Crop the bottom portion of the image to remove scale/logo
    % cropRatio: fraction to remove from bottom (e.g., 0.1 = remove bottom 10%)
    if nargin < 2 % If crop ratio not provided
        cropRatio = 0.07; % Default: remove bottom 7%
    end
    
    [height, ~, ~] = size(image); % Get image dimensions
    cropHeight = round(height * (1 - cropRatio)); % Calculate height after cropping
    
    if length(size(image)) == 3 % If image has 3 dimensions (color)
        croppedImage = image(1:cropHeight, :, :); % Crop all channels
    else
        croppedImage = image(1:cropHeight, :); % Crop grayscale image
    end
end

%% --- Enhanced registration function with comprehensive debug output ---
function [registeredColor, tform] = registerImages(gray1, gray2, colorImg2, scene)
    registeredColor = []; % Initialize output
    tform = []; % Initialize transformation matrix
    
    try
        % Crop images for feature detection
        cropRatio = 0.07; % Remove bottom 7% of image
        gray1Cropped = cropImageForFeatureDetection(gray1, cropRatio); % Crop reference image
        gray2Cropped = cropImageForFeatureDetection(gray2, cropRatio); % Crop target image

        % feature detection with adaptive approach
        [matched1, matched2] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped, scene); % Find matching features

        if size(matched1, 1) < 4 % If insufficient matches found
            registeredColor = []; % Return empty result
            return;
        end

        MaxDistance_list = [1.5, 5, 10, 50, 100, 500]; % List of distance thresholds to try
        Confidence = [90, 90, 90, 93, 95, 97]; % List of confidence levels (not used)
        inlierCount = 0; % Initialize inlier counter
        i = 1; % Initialize attempt counter
        
        while inlierCount < 5 && i <= length(MaxDistance_list) % Try different distance thresholds
            % Robust transformation estimation
            [tform, inlierIdx] = estgeotform2d(matched2, matched1, 'similarity', ... % Estimate similarity transformation
                'MaxNumTrials', 3000, 'Confidence', 93, 'MaxDistance', MaxDistance_list(i)); % Using RANSAC
            
            inlierCount = sum(inlierIdx); % Count inlier matches
        
            i = i + 1; % Move to next distance threshold
        end

        if inlierCount < 5 % If too few inliers
            registeredColor = []; % Return empty result
            return;
        end

        % Apply transformation to color image if provided
        outputRef = imref2d(size(gray1)); % Create reference coordinate system
        if ~isempty(colorImg2) % If color image provided
            if size(colorImg2, 3) == 3 % If RGB image
                % Register each color channel separately
                registeredColor = zeros([size(gray1), 3], 'like', colorImg2); % Initialize output
                for ch = 1:3 % Process each color channel
                    registeredColor(:,:,ch) = imwarp(colorImg2(:,:,ch), tform, ... % Apply transformation
                        'OutputView', outputRef, 'Interp', 'linear', 'FillValues', 0); % With linear interpolation
                end
            else
                % Single channel color image
                registeredColor = imwarp(colorImg2, tform, 'OutputView', outputRef, ... % Apply transformation
                    'Interp', 'linear', 'FillValues', 0); % With linear interpolation
            end
        end
    catch ME
        warning('Register images: %s', ME.message); % Display error message
        registeredColor = []; % Return empty result
    end
end

%% --- Robust feature detection with error handling ---
function [matched1, matched2] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped, sceneType)
    % Initialize outputs
    matched1 = []; % Initialize matched points from first image
    matched2 = []; % Initialize matched points from second image
    
    % Get adaptive parameters with validation
    [surfParams, cornerParams] = getSceneParametersWithCorners(sceneType); % Get parameters for scene type
    
    % SURF detection with integral image optimization
    integral1 = integralImage(gray1Cropped); % Compute integral image for first image
    integral2 = integralImage(gray2Cropped); % Compute integral image for second image

    pts1SURF = detectSURFFeatures(gray1Cropped, ... % Detect SURF features in first image
        'MetricThreshold', surfParams.threshold, ... % Using adaptive threshold
        'NumOctaves', surfParams.octaves); % Using adaptive octaves
    pts2SURF = detectSURFFeatures(gray2Cropped, ... % Detect SURF features in second image
        'MetricThreshold', surfParams.threshold, ... % Using adaptive threshold
        'NumOctaves', surfParams.octaves); % Using adaptive octaves
    
    pts1Corner = detectHarrisFeatures(gray1Cropped, ... % Detect corner features in first image
        'MinQuality', cornerParams.cornerQuality, ... % Using adaptive quality threshold
        'FilterSize', cornerParams.cornerFilterSize); % Using adaptive filter size
    pts2Corner = detectHarrisFeatures(gray2Cropped, ... % Detect corner features in second image
        'MinQuality', cornerParams.cornerQuality, ... % Using adaptive quality threshold
        'FilterSize', cornerParams.cornerFilterSize); % Using adaptive filter size


    % Extract features from all methods
    [f1SURF, vpts1SURF] = extractFeatures(gray1Cropped, pts1SURF); % Extract SURF descriptors from first image
    [f2SURF, vpts2SURF] = extractFeatures(gray2Cropped, pts2SURF); % Extract SURF descriptors from second image

    [f1Corner, vpts1Corner] = extractFeatures(gray1Cropped, pts1Corner); % Extract corner descriptors from first image
    [f2Corner, vpts2Corner] = extractFeatures(gray2Cropped, pts2Corner); % Extract corner descriptors from second image

    % Match features from each method
    indexPairsSURF = matchFeatures(f1SURF, f2SURF, ... % Match SURF features
        'Unique', true, 'MaxRatio', surfParams.maxRatio); % Using adaptive ratio threshold
    
    indexPairsCorner = matchFeatures(f1Corner, f2Corner, ... % Match corner features
        'Unique', true, 'MaxRatio', cornerParams.maxRatio); % Using adaptive ratio threshold

    % Combine matches from all methods
    if ~isempty(indexPairsSURF) % If SURF matches found
        matched1 = [matched1; vpts1SURF(indexPairsSURF(:,1)).Location]; % Add SURF matches from first image
        matched2 = [matched2; vpts2SURF(indexPairsSURF(:,2)).Location]; % Add SURF matches from second image
    end
    
    if ~isempty(indexPairsCorner) % If corner matches found
        matched1 = [matched1; vpts1Corner(indexPairsCorner(:,1)).Location]; % Add corner matches from first image
        matched2 = [matched2; vpts2Corner(indexPairsCorner(:,2)).Location]; % Add corner matches from second image
    end
end

%% --- Fixed parameter function with odd FilterSize values ---
function [surfParams, cornerParams] = getSceneParametersWithCorners(sceneType)
    % Get SURF parameters
    surfParams = struct(); % Initialize SURF parameters structure
    
    % Get corner detection parameters
    cornerParams = struct(); % Initialize corner parameters structure
    
    switch lower(sceneType) % Switch based on scene type
        case 'city'
            % SURF parameters for city scenes
            surfParams.threshold = 800; % High threshold for strong features
            surfParams.octaves = 4; % Standard number of octaves
            surfParams.maxRatio = 0.7; % Conservative matching ratio
            
            % Corner parameters for city scenes (good for buildings)
            cornerParams.cornerQuality = 0.015; % High quality threshold
            cornerParams.cornerFilterSize = 5; % Standard filter size
            cornerParams.maxRatio = 0.7; % Conservative matching ratio
            
        case 'nature'
            % SURF parameters for nature scenes
            surfParams.threshold = 500; % Medium threshold
            surfParams.octaves = 4; % Standard number of octaves
            surfParams.maxRatio = 0.75; % Balanced matching ratio
            
            % Corner parameters for nature scenes (balanced)
            cornerParams.cornerQuality = 0.005; % Medium quality threshold
            cornerParams.cornerFilterSize = 3; % Small filter size
            cornerParams.maxRatio = 0.75; % Balanced matching ratio
            
        otherwise
            % Default parameters
            surfParams.threshold = 400; % Default threshold
            surfParams.octaves = 3; % Default octaves
            surfParams.maxRatio = 0.8; % Default matching ratio
            
            cornerParams.cornerQuality = 0.015; % Default quality threshold
            cornerParams.cornerFilterSize = 3; % Default filter size
            cornerParams.maxRatio = 0.8;
    end
end
