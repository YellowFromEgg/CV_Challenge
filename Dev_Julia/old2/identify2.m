imgPath = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg";
imgPath1 ="C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg";
imgPath2 = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg";



%% Method 1

evaluateLandCoverMethods(imgPath);

function evaluateLandCoverMethods(imgPath)
    % EVALUATELANDCOVERMETHODS
    % Displays original image and results of three land cover classification methods

    % --- Load image ---
    rgbImg = im2double(imread(imgPath));

    % Prepare figure
    figure('Name', 'Land Cover Classification Evaluation', 'NumberTitle', 'off');
    
    % --- 1. Original Image ---
    subplot(2,2,1);
    imshow(rgbImg);
    title('Original Image');

    % --- 2. findRegionsFromRGBImage ---
    subplot(2,2,2);
    labelImg1 = classify_findRegionsFromRGBImage(rgbImg);
    imshow(labelImg1);
    title('findRegionsFromRGBImage');

    % --- 3. analyzeRGBLandcover ---
    subplot(2,2,3);
    labelImg2 = classifyLandCoverEnhanced(rgbImg);
    imshow(labelImg2);
    title('analyzeRGBLandcover');

    % --- 4. classifyLandCoverColorRiver ---
    subplot(2,2,4);
    labelImg3 = classify_classifyLandCoverColorRiver(rgbImg);
    imshow(labelImg3);
    title('classifyLandCoverColorRiver');
end

function overlay = classify_findRegionsFromRGBImage(rgbImg)
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);
    vegMask = (G > R) & (G > B) & (G > 0.35);
    waterMask = (B > G) & (B > R) & (B > 0.35);
    landMask = ~(vegMask | waterMask);
    labelImg = zeros(size(R));
    labelImg(waterMask) = 1;
    labelImg(vegMask) = 2;
    labelImg(landMask) = 3;
    cmap = [0 0 1; 0 1 0; 0.6 0.3 0];
    overlay = labeloverlay(rgbImg, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
end

%not used
function overlay = classify_analyzeRGBLandcover(rgbImg)
    R = rgbImg(:,:,1); G = rgbImg(:,:,2); B = rgbImg(:,:,3);
    hsv = rgb2hsv(rgbImg); S = hsv(:,:,2); V = hsv(:,:,3);
    ExG = 2*G - R - B;
    vegMask = ExG > 0.05;
    waterMask = (S < 0.3) & (V > 0.4) & (B > R) & (B > G);
    urbanMask = (S < 0.25) & (V > 0.6) & abs(R-G) < 0.1 & abs(R-B) < 0.1;
    landMask = ~(vegMask | waterMask | urbanMask);
    labelImg = zeros(size(R));
    labelImg(waterMask) = 1;
    labelImg(vegMask) = 2;
    labelImg(urbanMask) = 3;
    labelImg(landMask) = 4;
    cmap = [0 0 1; 0 1 0; 1 0 0; 0.6 0.3 0];
    overlay = labeloverlay(rgbImg, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
end

function overlay = classify_classifyLandCoverColorRiver(rgbImg)
    R = rgbImg(:,:,1); G = rgbImg(:,:,2); B = rgbImg(:,:,3);
    h = size(R, 1); w = size(R, 2);
    darkGreen = (G > 0.25) & (R < 0.25) & ((R + G + B)/3 < 0.4);
    forestMask = darkGreen;
    riverMask = (G > R + 0.05) & (G > B + 0.05) & (G > 0.35);
    riverMask = imclose(riverMask, strel('disk', 1));
    riverMask = bwareaopen(riverMask, 50);
    cityMask = ((R + G + B)/3 > 0.3) & ~forestMask & ~riverMask;
    restMask = ~(forestMask | riverMask | cityMask);
    labelImg = zeros(h, w);
    labelImg(forestMask) = 1;
    labelImg(riverMask) = 2;
    labelImg(cityMask) = 3;
    labelImg(restMask) = 4;
    cmap = [0 1 0; 0 0 1; 0.5 0.5 0.5; 0.6 0.4 0.2];
    overlay = labeloverlay(rgbImg, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
end


function overlay = classifyLandCoverEnhanced(rgbImg)
    % Convert to double for processing
    rgbImg = im2double(rgbImg);
    R = rgbImg(:,:,1);
    G = rgbImg(:,:,2);
    B = rgbImg(:,:,3);
    [h, w, ~] = size(rgbImg);
    
    %% Convert to HSV color space
    hsvImg = rgb2hsv(rgbImg);
    H = hsvImg(:,:,1); S = hsvImg(:,:,2); V = hsvImg(:,:,3);

    %% Approximate NDVI (from R and G only)
    ndvi = (G - R) ./ (G + R + eps);

    %% Grayscale and texture (entropy)
    grayImg = rgb2gray(rgbImg);
    entropyMap = entropyfilt(grayImg);

    %% Forest detection using NDVI and texture
    forestMask = (ndvi > 0.1) & (entropyMap > 3);

    %% River detection using HSV (blue hue range)
    riverMask = (H > 0.55 & H < 0.67) & (S > 0.25) & (V > 0.2);
    riverMask = imclose(riverMask, strel('disk', 2));
    riverMask = bwareaopen(riverMask, 50);

    %% City/urban detection: brighter, less green, low texture
    avgBrightness = (R + G + B) / 3;
    cityMask = (avgBrightness > 0.35) & ~forestMask & ~riverMask & (entropyMap < 3);

    %% Rest (uncategorized)
    restMask = ~(forestMask | riverMask | cityMask);

    %% Assign labels
    labelImg = zeros(h, w);
    labelImg(forestMask) = 1;  % Green
    labelImg(riverMask) = 2;   % Blue
    labelImg(cityMask) = 3;    % Gray
    labelImg(restMask) = 4;    % Brown

    %% Visualization colormap
    cmap = [0 1 0;         % Forest - Green
            0 0 1;         % River - Blue
            0.5 0.5 0.5;   % City - Gray
            0.6 0.4 0.2];  % Rest - Brown

    %% Overlay output
    overlay = labeloverlay(rgbImg, labelImg, 'Colormap', cmap, 'Transparency', 0.4);
end
