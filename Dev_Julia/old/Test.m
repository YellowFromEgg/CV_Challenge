%bilderOrdner = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai"
bilderOrdner = "C:/Users/julia/OneDrive/Dokumente/1_TUM/Computer Vision/Challenge/CV_Challenge/Datasets/Dubai"; 
bilderOrdner = "C:/Users/julia/OneDrive/Dokumente/1_TUM/Computer Vision/Challenge/CV_Challenge/Datasets/Brazilian Rainforest"; 
bilderOrdner = "C:/Users/julia/OneDrive/Dokumente/1_TUM/Computer Vision/Challenge/CV_Challenge/Datasets/Wiesn"; 
schwelle = 0.2;

registrierteBilder = registriere_satellitenbilder(bilderOrdner);
heatmapNorm = heatmap_veraenderungshauefigkeit(registrierteBilder, schwelle);


%% Ansatz nur über heatmap pixel

heatmapthreshold = 0.03;
connectingPixels = 2;

detect_high_edge_regions_sharper(heatmapNorm, heatmapthreshold, connectingPixels);

%% Ansatz über edge density

edgeDensityThreshold = 0.15;
connectingPixels = 2;

heatmapNorm = heatmapNorm(heatmapNorm > 0.03);

detect_high_edge_regions(heatmapNorm,edgeDensityThreshold,connectingPixels)


%% --- Hilfsfunktion: Bilder registrieren ---

function region_dauer_map = veraenderungsdauer_pro_region(bilderOrdner, schwelle)
    if nargin < 2
        schwelle = 0.2;
    end

    %% 1. Bilder laden & registrieren
    registrierteBilder = registriere_satellitenbilder(bilderOrdner);
    numImages = numel(registrierteBilder);

    if numImages < 2
        error('Mindestens zwei registrierte Bilder erforderlich.');
    end

    %% 2. Veränderungen pro Pixel über Zeit erfassen
    [h, w, ~] = size(registrierteBilder{1});
    veraenderungsMasken = false(h, w, numImages-1);

    for i = 2:numImages
        img1 = im2double(rgb2gray(registrierteBilder{i-1}));
        img2 = im2double(rgb2gray(registrierteBilder{i}));
        diff = abs(img1 - img2);

        veraenderungsMasken(:,:,i-1) = diff > schwelle;
    end

    %% 3. Heatmap der Veränderungshäufigkeit (für Regionserkennung)
    heatmapNorm = sum(veraenderungsMasken, 3) / (numImages - 1);
    highChangeMask = heatmapNorm > 0.2;

    %% 4. Regionen erkennen
    CC = bwconncomp(highChangeMask);
    props = regionprops(CC, 'PixelIdxList');

    %% 5. Dauer der Veränderung pro Region berechnen
    regionDauer = zeros(CC.NumObjects, 1);
    for i = 1:CC.NumObjects
        pixelIdx = props(i).PixelIdxList;
        [rows, cols] = ind2sub([h, w], pixelIdx);

        dauerProPixel = zeros(length(pixelIdx), 1);
        for j = 1:length(pixelIdx)
            r = rows(j); c = cols(j);
            zeitreihe = squeeze(veraenderungsMasken(r, c, :));
            first = find(zeitreihe, 1, 'first');
            last = find(zeitreihe, 1, 'last');
            if ~isempty(first)
                dauerProPixel(j) = last - first + 1;
            end
        end

        regionDauer(i) = mean(dauerProPixel);
    end

    %% 6. Regiondauer in Karte eintragen
    region_dauer_map = zeros(h, w);
    for i = 1:CC.NumObjects
        region_dauer_map(CC.PixelIdxList{i}) = regionDauer(i);
    end

    %% 7. Visualisierung
    figure;
    imagesc(region_dauer_map);
    colormap parula;
    colorbar;
    axis image off;
    title('Veränderungsdauer pro Region (in Zeitschritten)');
end

function registrierteBilder = registriere_satellitenbilder(bilderOrdner)
    %% 1. Bilder laden
    bildDatastore = imageDatastore(bilderOrdner);
    numImages = numel(bildDatastore.Files);

    if numImages < 2
        error('Mindestens zwei Bilder werden benötigt.');
    end

    % Erstes Bild als Referenz
    refImage = readimage(bildDatastore, 1);
    refGray = im2gray(refImage);
    refImage = im2double(refImage);

    % Registrierte Bilder speichern
    registrierteBilder = cell(numImages, 1);
    registrierteBilder{1} = refImage;

    % SURF-Merkmale des Referenzbildes
    refPoints = detectSURFFeatures(refGray);
    [refFeatures, refValidPoints] = extractFeatures(refGray, refPoints);

    % Maske initialisieren mit allen gültigen Pixeln
    gemeinsameMaske = true(size(refGray));

    %% 2. Registrierung aller weiteren Bilder
    for i = 2:numImages
        currImage = readimage(bildDatastore, i);
        currGray = im2gray(currImage);
        currImage = im2double(currImage);

        currPoints = detectSURFFeatures(currGray);
        [currFeatures, currValidPoints] = extractFeatures(currGray, currPoints);

        indexPairs = matchFeatures(refFeatures, currFeatures);
        matchedRef = refValidPoints(indexPairs(:,1));
        matchedCurr = currValidPoints(indexPairs(:,2));

        % Transformation schätzen und anwenden
        tform = estimateGeometricTransform2D(matchedCurr, matchedRef, 'similarity');
        outputView = imref2d(size(refImage));
        currRegistered = imwarp(currImage, tform, 'OutputView', outputView);

        % Maske für gültige Pixel im registrierten Bild
        currMask = imwarp(true(size(currGray)), tform, 'OutputView', outputView);
        gemeinsameMaske = gemeinsameMaske & currMask;

        registrierteBilder{i} = currRegistered;
    end

    %% 3. Gemeinsame Maske auf alle Bilder anwenden
    for i = 1:numImages
        img = registrierteBilder{i};
        maskedImg = img;
        for c = 1:size(img, 3)
            channel = img(:,:,c);
            channel(~gemeinsameMaske) = 0;
            maskedImg(:,:,c) = channel;
        end
        registrierteBilder{i} = maskedImg;
    end

    %% Optional: Registrierte Bilder speichern
    % for i = 1:numImages
    %     imwrite(registrierteBilder{i}, sprintf('registriert_maskiert_%02d.png', i));% end
end

function heatmapNorm = heatmap_veraenderungshauefigkeit(registrierteBilder, schwelle)
    % registrierteBilder: Zellarray mit registrierten RGB-Bildern
    % schwelle: Schwellwert für Veränderung (z. B. 0.2)

    if nargin < 2
        schwelle = 0.2; % Standardwert
    end

    numImages = numel(registrierteBilder);
    if numImages < 2
        error('Mindestens zwei registrierte Bilder erforderlich.');
    end

    % Initialisiere Zählerbild
    [h, w, ~] = size(registrierteBilder{1});
    veraenderungsZaehler = zeros(h, w);

    % Schleife über Bildpaare
    for i = 2:numImages
        img1 = registrierteBilder{i-1};
        img2 = registrierteBilder{i};

        % Differenzbild berechnen
        diff = imabsdiff(img1, img2);
        diffGray = rgb2gray(diff);

        % Binäre Maske der Veränderung
        veraenderungsMaske = diffGray > schwelle;

        % Zähler erhöhen
        veraenderungsZaehler = veraenderungsZaehler + veraenderungsMaske;
    end

    % Normieren auf [0, 1] für Visualisierung
    heatmapNorm = veraenderungsZaehler / (numImages - 1);

    % Anzeige
    % figure;
    % imagesc(heatmapNorm);
    % colormap hot;
    % colorbarHandle = colorbar;
    % axis image off;
    % title('Heatmap der Veränderungshäufigkeit pro Pixel');
    % 
    % Farbskala beschriften
    % colorbarHandle.Label.String = ...
    %     sprintf('Veränderungshäufigkeit\n(0 = keine Veränderung, %d = Veränderung bei jedem Vergleich)', numImages - 1);
    % colorbarHandle.Ticks = linspace(0, 1, numImages);
    % colorbarHandle.TickLabels = arrayfun(@(x) sprintf('%dx', x), 0:(numImages - 1), 'UniformOutput', false);
end

function detect_high_edge_regions(I, edgeDensityThreshold,connectingPixels)
     % Wenn I eine Heatmap ist, ist es bereits grau
    gray = im2double(I);  % KEIN rgb2gray mehr!

    % STEP 2: Edge detection and edge density map
    edges = edge(gray, 'Sobel');  % Optional: 'Canny'
    edgeDensity = conv2(double(edges), ones(15), 'same') / (15^2);

    % Threshold edge density to detect high-density areas
    highEdgeMask = edgeDensity > edgeDensityThreshold;

    % STEP 3: Morphological cleanup
    cleanedMask = bwareaopen(highEdgeMask, 10);
    cleanedMask = imclose(cleanedMask, strel('disk', connectingPixels));
    %cleanedMask = imfill(cleanedMask, 'holes');

    % STEP 4: Label each connected region uniquely
    CC = bwconncomp(cleanedMask);
    labelMap = labelmatrix(CC);

    % Generate a random colormap (exclude 0)
    numRegions = CC.NumObjects;
    cmap = [0 0 0; rand(numRegions, 3)];  % 0 = background (black)

    % STEP 5: Visualization
    figure('Name', 'High Edge Density Regions', 'Position', [100 100 1400 500]);
    subplot(1,2,1); imshow(I); title('Original Image');
    subplot(1,2,2);
    imagesc(labelMap); axis image off;
    title('High Edge Density Regions (Colored)');
    colormap(gca, cmap); caxis([0 numRegions]);
    colorbar;
end

function detect_high_edge_regions_sharper(I, heatmapthreshold, connectingPixels)
     % Wenn I eine Heatmap ist, ist es bereits grau
    gray = im2double(I);  % KEIN rgb2gray mehr!

    highEdgeMask = gray >= heatmapthreshold;

    % STEP 3: Morphological cleanup
    cleanedMask = bwareaopen(highEdgeMask, 10);
    cleanedMask = imclose(cleanedMask, strel('disk', connectingPixels));
    cleanedMask = imfill(cleanedMask, 'holes');

    % STEP 4: Label each connected region uniquely
    CC = bwconncomp(cleanedMask);
    labelMap = labelmatrix(CC);

    % Generate a random colormap (exclude 0)
    numRegions = CC.NumObjects;
    cmap = [0 0 0; rand(numRegions, 3)];  % 0 = background (black)

    % STEP 5: Visualization
    figure('Name', 'High Edge Density Regions', 'Position', [100 100 1400 500]);
    subplot(1,2,1); imshow(I); title('Original Image');
    subplot(1,2,2);
    imagesc(labelMap); axis image off;
    title('High Edge Density Regions (Colored)');
    colormap(gca, cmap); caxis([0 numRegions]);
    colorbar;
end
