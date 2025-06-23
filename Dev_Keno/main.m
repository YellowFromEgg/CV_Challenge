function main()
    %% 1. Ordner mit Bildern wählen
    imageFolder = uigetdir(pwd, 'Wähle Ordner mit Satellitenbildern aus');
    if imageFolder == 0
        disp('Kein Ordner gewählt. Abbruch.');
        return;
    end

    %% 2. Bilder laden und sortieren
    imageFiles = dir(fullfile(imageFolder, '*.jpg'));
    if isempty(imageFiles)
        imageFiles = dir(fullfile(imageFolder, '*.png'));
    end
    if length(imageFiles) < 2
        error('Mindestens zwei Bilder erforderlich.');
    end

    imageFiles = sort_nat({imageFiles.name});
    img1 = imread(fullfile(imageFolder, imageFiles{1}));
    img2 = imread(fullfile(imageFolder, imageFiles{end}));

    %% 3. Preprocessing: Graustufen + Histogrammausgleich
    gray1 = preprocessImage(img1);
    gray2 = preprocessImage(img2);

    % Debug: zeige Vorverarbeitung
    figure, imshow(gray1), title('Bild 1: Vorverarbeitet');
    figure, imshow(gray2), title('Bild 2: Vorverarbeitet');

    %% 4. Featurebasierte Registrierung (inkl. Rotation)
    [registered2, commonMask] = registerImages(gray1, gray2);

    % Debug: zeige registriertes Bild 2
    figure, imshow(registered2), title('Bild 2: Registriert auf Bild 1');

    %% 5. Differenz berechnen
    diffImage = imabsdiff(gray1, registered2);
    diffImage(~commonMask) = 0; % maskiere nicht überlappende Bereiche

    %% 6. Binarisierung zur Änderungserkennung
    changeMap = imbinarize(diffImage, 'adaptive', 'Sensitivity', 0.5);

    %% 7. Visualisierung
    visualizeDifferenceHeatmap(diffImage);
end

%% --- Preprocessing: Grau + Histogrammausgleich ---
function grayImage = preprocessImage(img)
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = adapthisteq(img);         % Kontrast- und Helligkeitsanpassung
    grayImage = im2double(img);     % Normalisierung auf [0,1]
end

%% --- Registrierung basierend auf SURF Features ---
function [registered2, validMask] = registerImages(gray1, gray2)
    pts1 = detectSURFFeatures(gray1);
    pts2 = detectSURFFeatures(gray2);
    [f1, vpts1] = extractFeatures(gray1, pts1);
    [f2, vpts2] = extractFeatures(gray2, pts2);
    indexPairs = matchFeatures(f1, f2, 'Unique', true);
    matched1 = vpts1(indexPairs(:,1));
    matched2 = vpts2(indexPairs(:,2));

    % Transformation berechnen (inkl. Rotation)
    tform = estimateGeometricTransform2D(matched2, matched1, 'similarity');

    % Registrierung
    outputRef = imref2d(size(gray1));
    registered2 = imwarp(gray2, tform, 'OutputView', outputRef);

    % Gültige Pixelmaske berechnen
    mask = ones(size(gray2));
    warpedMask = imwarp(mask, tform, 'OutputView', outputRef);
    validMask = warpedMask > 0;
end

%% --- Heatmap der Differenz anzeigen ---
function visualizeDifferenceHeatmap(diffImage)
    figure;
    imagesc(diffImage);
    axis image off;
    colormap jet;
    colorbar;
    title('Differenz-Heatmap');
end

%% --- Natürliche Sortierung der Bilddateien ---
function sorted = sort_nat(filenames)
    expr = '\d{4}';
    years = regexp(filenames, expr, 'match', 'once');
    years = cellfun(@str2double, years);
    [~, idx] = sort(years);
    sorted = filenames(idx);
end
