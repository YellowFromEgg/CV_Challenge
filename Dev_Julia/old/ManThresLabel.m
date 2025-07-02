imgPaths = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2017.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Wiesn\3_2020.jpg"
};

results = [];  % Initialize empty results table

for imgIdx = 1:length(imgPaths)
    imagePath = imgPaths{imgIdx};
    [~, imageName, ~] = fileparts(imagePath);
    I = imread(imagePath);

    % --- Preprocessing ---
    gray = rgb2gray(I);
    gray = im2double(gray);
    stdMap = stdfilt(gray, true(15));
    stdMap = mat2gray(stdMap);
    baseMask = stdMap > 0.15;

    cityMask = bwareaopen(baseMask, 100);
    cityMask = imclose(cityMask, strel('disk', 5));
    cityMask = imfill(cityMask, 'holes');
    CC = bwconncomp(cityMask);
    props = regionprops(CC, 'PixelIdxList', 'BoundingBox', 'Area');

    Ihsv = rgb2hsv(I);
    H = Ihsv(:,:,1); S = Ihsv(:,:,2); V = Ihsv(:,:,3);

    regionData = [];

    for r = 1:CC.NumObjects
        pix = props(r).PixelIdxList;
        if numel(pix) < 500
            continue
        end

        % Extract HSV values
        h = H(pix); s = S(pix); v = V(pix);
        meanSat = mean(s);
        meanVal = mean(v);
        varVal  = var(v);

        % Pixel-type ratios
        isGreen = h > 0.2 & h < 0.45 & s > 0.25;
        isRed   = (h < 0.05 | h > 0.95) & s > 0.3;
        isGray  = s < 0.2 & v > 0.2 & v < 0.8;
        isTan   = h > 0.05 & h < 0.15 & s > 0.2 & v > 0.6;

        % Visualize region
        mask = false(size(gray));
        mask(pix) = true;

        figure(1); clf;
        subplot(1,2,1); imshow(I); title(['Original: ', imageName]);
        subplot(1,2,2); imshow(mask); title(['Region ', num2str(r), ' (area = ', num2str(numel(pix)), ')']);
        sgtitle('Label this region: 1=Forest, 2=City, 3=Land, 4=Others (1,2,..), 5=Background');

        labelInput = input('Enter label: ', 's');
        labelClean = strrep(labelInput, ' ', '');

        if strcmp(labelClean, '5')
            labelFinal = "background";
        else
            labelFinal = string(labelClean);
        end

        % Add to table
        entry = table({imageName}, r, labelFinal, ...
                      sum(isRed), sum(isGray), sum(isTan), sum(isGreen), ...
                      sum(isRed)/numel(pix), sum(isGray)/numel(pix), ...
                      sum(isTan)/numel(pix), sum(isGreen)/numel(pix), ...
                      meanSat, meanVal, varVal, ...
                      'VariableNames', {'PictureName','RegionID','Label', ...
                                        'RedSum','GraySum','TanSum','GreenSum', ...
                                        'RedRatio','GrayRatio','TanRatio','GreenRatio', ...
                                        'MeanSat','MeanVal','VarVal'});
        regionData = [regionData; entry];
    end

    results = [results; regionData];
end

% --- Show and Save ---
disp(results);
writetable(results, 'manual_region_labeling.csv');
