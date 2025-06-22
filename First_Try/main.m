function main()
    % Verzeichnis mit Bildern auswählen
    imageFolder = uigetdir(pwd, 'Wähle den Ordner mit Satellitenbildern aus');
    if imageFolder == 0
        disp('Kein Ordner gewählt. Abbruch.');
        return;
    end

    % Lade alle Bilder im Ordner
    imageFiles = dir(fullfile(imageFolder, '*.jpg'));
    if isempty(imageFiles)
        imageFiles = dir(fullfile(imageFolder, '*.png'));
    end
    if length(imageFiles) < 2
        error('Mindestens zwei Bilder erforderlich.');
    end

    % Sortiere nach Jahr (angenommen: YYYY MM.jpg oder YYYY_MM.png)
    imageFiles = sort_nat({imageFiles.name});

    % Lade erstes und letztes Bild zum Vergleich
    img1 = imread(fullfile(imageFolder, imageFiles{1}));
    img2 = imread(fullfile(imageFolder, imageFiles{end}));

    % Rufe Vergleichsfunktion auf
    changeMap = detectChanges(img1, img2);

    % Zeige Veränderung
    visualizeChanges(img1, img2, changeMap);
    
    % Bilder vor Verarbeitung anpassen
    gray1 = preprocessImage(img1);
    gray2 = preprocessImage(img2);
    
    % Bringe beide Bilder auf gleiche Größe
    minRows = min(size(gray1, 1), size(gray2, 1));
    minCols = min(size(gray1, 2), size(gray2, 2));
    gray1_resized = imresize(gray1, [minRows, minCols]);
    gray2_resized = imresize(gray2, [minRows, minCols]);
    
    % Zeige Differenz-Heatmap
    visualizeDifferenceHeatmap(imabsdiff(gray1_resized, gray2_resized));

end

function changeMap = detectChanges(img1, img2)
    % Vorverarbeitung
    gray1 = preprocessImage(img1);
    gray2 = preprocessImage(img2);

    % Bildregistrierung mit SURF Features
    pts1 = detectSURFFeatures(gray1);
    pts2 = detectSURFFeatures(gray2);
    [f1, vpts1] = extractFeatures(gray1, pts1);
    [f2, vpts2] = extractFeatures(gray2, pts2);
    indexPairs = matchFeatures(f1, f2);

    matched1 = vpts1(indexPairs(:,1));
    matched2 = vpts2(indexPairs(:,2));

    % Schätze geometrische Transformation
    tform = estimateGeometricTransform2D(matched2, matched1, 'similarity');

    % Registriere img2 zu img1
    registered2 = imwarp(gray2, tform, 'OutputView', imref2d(size(gray1)));

    % Bilddifferenz berechnen
    diffImage = imabsdiff(gray1, registered2);

    % Schwellwert auf Differenzbild
    changeMap = imbinarize(diffImage, 'adaptive', 'Sensitivity', 0.5);
end

function grayImage = preprocessImage(img)
    % Falls RGB, in Graustufen umwandeln
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    % Normalisiere Helligkeit (optional)
    img = imadjust(img);
    grayImage = im2double(img);
end

function sorted = sort_nat(filenames)
    % Natürliche Sortierung nach Jahreszahlen
    expr = '\d{4}';
    years = regexp(filenames, expr, 'match', 'once');
    years = cellfun(@str2double, years);
    [~, idx] = sort(years);
    sorted = filenames(idx);
end

function visualizeChanges(img1, img2, changeMask)
    % Stelle sicher, dass img1 RGB ist
    if size(img1, 3) == 1
        img1 = cat(3, img1, img1, img1);
    end

    % Maske vergrößern für bessere Sichtbarkeit
    dilatedMask = imdilate(changeMask, strel('disk', 2));

    % Erstelle eine Overlay-Farbe (rot)
    overlay = img1;
    overlay(:,:,1) = uint8(dilatedMask) * 255 + uint8(~dilatedMask) .* img1(:,:,1); % rot
    overlay(:,:,2) = uint8(~dilatedMask) .* img1(:,:,2); % grün bleibt unverändert
    overlay(:,:,3) = uint8(~dilatedMask) .* img1(:,:,3); % blau bleibt unverändert

    % Visualisierung
    figure;
    imshow(overlay);
    title('Veränderungen hervorgehoben (rot)');
end

function visualizeDifferenceHeatmap(diffImage)
    figure;
    imagesc(diffImage);
    axis image off;
    colormap jet;
    colorbar;
    title('Differenz-Heatmap');
end
