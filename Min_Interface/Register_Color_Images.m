function [registeredColorImages, validImageIndices, imageFiles, refIdx, tformList] = Register_Color_Images(imageFiles, colorImages, numImages)
    close all;
    tStart = tic;

    %% 3. Preprocess ALL images once and cache them
    grayImages = cell(numImages, 1);
    
    for i = 1:numImages
        % Preprocess to grayscale (only show debug for first image)
        grayImages{i} = preprocessImage(colorImages{i});
    end

    %% 4. Find optimal reference using cached preprocessed images
    refIdx = findOptimalReference(grayImages);
    
    % Get reference images
    refImg = colorImages{refIdx};
    refGray = grayImages{refIdx};

    %% 5. Process all other images against the optimal reference
    % Storage for registered color images
    registeredColorImages = cell(numImages, 1);
    tformList = cell(numImages, 1);
    validImageIndices = [];
    
    for i = 1:numImages
        if i == refIdx
            % Store reference image
            registeredColorImages{i} = refImg;
            validImageIndices = [validImageIndices, i];

            continue; % Skip the reference image itself
        end
        
        % Use cached preprocessed images
        img = colorImages{i};
        gray = grayImages{i};
        
        % Registration with color image support - also support matrices
        [registeredColor, tform] = registerImages(refGray, gray, img);
        tformList{i} = tform;
        
        if isempty(registeredColor)
            continue;
        end
    
        % Store the registered color image
        registeredColorImages{i} = registeredColor;
        validImageIndices = [validImageIndices, i];
    end
    tEnd = toc(tStart);
    disp(tEnd);

end

%% --- Enhanced preprocessing ---
function grayImage = preprocessImage(img)

    % In Graustufen umwandeln, falls RGB
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    % --- Entropie-Maske berechnen und anwenden ---
    maskedImage = applyEntropyMask(img);  % Original + bearbeitetes Graubild
    
    img = im2double(img);

    % Rauschreduzierung
    img = imbilatfilt(img);

    % Histogramm-Anpassung
    img = adapthisteq(img, 'ClipLimit', 0.01, 'NumTiles', [6 6]);

    % Normalisierung
    img = mat2gray(img);

    grayImage = maskedImage;
end

%% Create Mask
function maskedImage = applyEntropyMask(grayImg)
    % --- Entropiekarte berechnen ---
    entropyMap = entropyfilt(grayImg);
    entropyNorm = mat2gray(entropyMap);

    % --- Binarisierung mit Otsu-Schwellenwert ---
    threshold = graythresh(entropyNorm);
    entropyMask = entropyNorm > threshold;

    % --- Maske vergrößern (z. B. um kleine Lücken zu schließen) ---
    se = strel('disk', 5);
    dilatedMask = imdilate(entropyMask, se);

    % --- Sicherstellen, dass die Maske 2D ist ---
    if ndims(dilatedMask) > 2
        dilatedMask = dilatedMask(:,:,1);
    end

    % --- Maske auf Originalbild anwenden ---
    if size(grayImg, 3) == 3
        % Farbige Bilder (RGB)
        maskedImage = zeros(size(grayImg), 'like', grayImg);
        for c = 1:3
            maskedImage(:,:,c) = grayImg(:,:,c) .* cast(dilatedMask, 'like', grayImg);
        end
    else
        % Graustufenbilder
        maskedImage = grayImg .* cast(dilatedMask, 'like', grayImg);
    end
end

%% --- Find optimal reference image based on feature matching success ---
function optimalIdx = findOptimalReference(preprocessedImages)
    numImages = length(preprocessedImages);
    
    % For large datasets, sample a subset for efficiency
    if numImages > 20
        sampleIndices = round(linspace(1, numImages, min(15, numImages)));
        fprintf('Sampling %d images for reference selection\n', length(sampleIndices));
    else
        sampleIndices = 1:numImages;
    end
    
    % Use cached preprocessed images
    images = cell(length(sampleIndices), 1);
    for i = 1:length(sampleIndices)
        idx = sampleIndices(i);
        images{i} = preprocessedImages{idx};
    end
    
    % Calculate pairwise registration success matrix
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
    end
    
    % Select the best reference
    [~, bestIdx] = max(referenceScores);
    optimalIdx = sampleIndices(bestIdx);
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
        [matched1, matched2] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped);

        [~, inlierIdx] = estgeotform2d(matched2, matched1, 'similarity', ...
                'MaxNumTrials', 300, 'Confidence', 90, 'MaxDistance', 5.0);
            
        featureCount = sum(inlierIdx);

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
function [registeredColor, tform] = registerImages(gray1, gray2, colorImg2)
    registeredColor = []; % Initialize output
    tform = [];
    
    try
        % Crop images for feature detection
        cropRatio = 0.07; % Remove bottom 7% of image
        gray1Cropped = cropImageForFeatureDetection(gray1, cropRatio);
        gray2Cropped = cropImageForFeatureDetection(gray2, cropRatio);

        % feature detection with adaptive approach
        [matched1, matched2] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped);

        if size(matched1, 1) < 4
            registeredColor = [];
            return;
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
        
            i = i + 1;
        end
        
        % % safety check based on resulting image size
        % scl = hypot(tform.T(1,1), tform.T(2,1));     % isotropic scale
        % rot = atan2d(tform.T(2,1), tform.T(1,1));    % rotation in degrees
        % tx  = tform.T(3,1);  ty = tform.T(3,2);      % translation (pixels)
        % 
        % % % project-dependent limits – massively scaled images are skipped
        % if  scl < 0.75 || scl > 1.75 || ...          % ≤ ±75 % zoom allowed
        %     abs(tx) > 0.8*size(gray1,2) || ...       % ≤ 80 % width translation
        %     abs(ty) > 0.8*size(gray1,1)
        %     warning('Rejected registration: scale=%.3f  rot=%.1f°  Δx=%d  Δy=%d', ...
        %             scl, rot, round(tx), round(ty));
        %     registeredColor = [];
        %     return
        % end

        if inlierCount < 2
            registeredColor = [];
            return;
        end

        % Apply transformation to color image if provided
        outputRef = imref2d(size(gray1));
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
    catch ME
        warning('Register images: %s', ME.message);
        registeredColor = [];
    end
end

%% --- Robust feature detection with error handling ---
function [matched1, matched2] = adaptiveFeatureDetection(gray1Cropped, gray2Cropped)
    % Initialize outputs
    matched1 = [];
    matched2 = [];
    
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
    
    % Get adaptive parameters with validation
    [surfParams, cornerParams] = getSceneParametersWithCorners(sceneType);
    
    % SURF detection with integral image optimization
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
    end
    
    if ~isempty(indexPairsCorner)
        matched1 = [matched1; vpts1Corner(indexPairsCorner(:,1)).Location];
        matched2 = [matched2; vpts2Corner(indexPairsCorner(:,2)).Location];
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