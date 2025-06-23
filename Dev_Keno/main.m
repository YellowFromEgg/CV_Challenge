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

    %% 3. Referenzbild vorbereiten
    refImg = imread(fullfile(imageFolder, imageFiles{1}));
    refGray = preprocessImage(refImg);
    refSize = size(refGray);

    % Kumulierte Differenzinitialisierung
    cumulativeDiff = zeros(refSize);
    totalMask = true(refSize);

    %% 4. Schleife über alle weiteren Bilder
    skippedImages = {};

    for i = 2:length(imageFiles)
        img = imread(fullfile(imageFolder, imageFiles{i}));
        gray = preprocessImage(img);

        % Registrierung
        [registered, validMask] = registerImages(refGray, gray);

        if isempty(registered)
            skippedImages{end+1} = imageFiles{i};
            continue;
        end
        
        % Differenzbild berechnen
        diffImage = imabsdiff(refGray, registered);
        diffImage(~validMask) = 0;

        % Akkumulation
        cumulativeDiff = cumulativeDiff + diffImage;
        totalMask = totalMask & validMask; % nur überlappende Bereiche
    end
    
    if ~isempty(skippedImages)
        disp('Folgende Bilder konnten nicht registriert werden und wurden übersprungen:');
        disp(skippedImages');
    end

    %% 5. Visualisierung kumulierter Differenz
    cumulativeDiff(~totalMask) = 0;
    visualizeDifferenceHeatmap(cumulativeDiff);
end


%% --- Preprocessing: Grau + Histogrammausgleich ---
function grayImage = preprocessImage(img)
    figure;
    subplot(1, 2, 1);
    imshow(img);
    title('Orginal');
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = adapthisteq(img);         % Kontrast- und Helligkeitsanpassung
    grayImage = im2double(img);     % Normalisierung auf [0,1
    subplot(1, 2, 2);
    imshow(grayImage);
    title('Verarbeitet');
end

%% --- Registrierung basierend auf SURF Features ---
% Diese Funktion registriert zwei Graustufenbilder (gray2 auf gray1)
% mithilfe von SURF-Merkmalen und einer Ähnlichkeitstransformation.
%
% Eingabeparameter:
%   gray1: Das Referenzbild (Graustufenbild). Das Bild, auf das gray2 ausgerichtet werden soll.
%   gray2: Das zu registrierende Bild (Graustufenbild). Dieses Bild wird transformiert.
%
% Ausgabeparameter:
%   registered2: Das auf gray1 registrierte Bild gray2.
%   validMask: Eine binäre Maske, die anzeigt, welche Pixel in registered2
%              aus gültigen Pixeln von gray2 stammen (d.h., nicht außerhalb des ursprünglichen Bildbereichs).

function [registered2, validMask] = registerImages(gray1, gray2)
    try
        % 1. SURF-Merkmale (Features) in beiden Bildern erkennen
        % detectSURFFeatures findet "Speeded Up Robust Features" (SURF)-Punkte,
        % die invariant gegenüber Skalierung und Rotation sind.
        pts1 = detectSURFFeatures(gray1); % Erkennung von SURF-Punkten im Referenzbild (gray1)
        pts2 = detectSURFFeatures(gray2); % Erkennung von SURF-Punkten im zu registrierenden Bild (gray2)
    
        % 2. Merkmalsdeskriptoren extrahieren
        % extractFeatures berechnet einen Deskriptor (eine Art "Fingerabdruck")
        % für jeden erkannten Merkmalspunkt. Dieser Deskriptor ermöglicht den
        % Vergleich von Merkmalen zwischen Bildern.
        [f1, vpts1] = extractFeatures(gray1, pts1); % Extrahiert Deskriptoren (f1) und gültige Punkte (vpts1) von gray1
        [f2, vpts2] = extractFeatures(gray2, pts2); % Extrahiert Deskriptoren (f2) und gültige Punkte (vpts2) von gray2
    
        % 3. Merkmale zwischen den Bildern abgleichen
        % matchFeatures vergleicht die Deskriptoren von f1 und f2, um Paare von
        % übereinstimmenden Merkmalspunkten zu finden. 'Unique', true stellt sicher,
        % dass jeder Merkmalspunkt in einem Bild nur mit höchstens einem Punkt im anderen Bild übereinstimmt.
        indexPairs = matchFeatures(f1, f2, 'Unique', true);
    
        if size(indexPairs, 1) < 3
            warning('Zu wenige Übereinstimmungen für Registrierung.');
            registered2 = [];
            validMask = [];
            return;
        end
    
        % 4. Korrespondierende Merkmalspunkte abrufen
        % Basierend auf den gefundenen Indexpaaren werden die tatsächlichen
        % Koordinaten der übereinstimmenden Punkte abgerufen.
        matched1 = vpts1(indexPairs(:,1)); % Übereinstimmende Punkte im Referenzbild (gray1)
        matched2 = vpts2(indexPairs(:,2)); % Übereinstimmende Punkte im zu registrierenden Bild (gray2)
    
        % 5. Transformation berechnen (inkl. Rotation, Skalierung und Translation)
        % estimateGeometricTransform2D schätzt die geometrische Transformation,
        % die benötigt wird, um matched2 auf matched1 abzubilden.
        % 'similarity' bedeutet, dass die Transformation Translation, Rotation und Skalierung
        % aber keine Scherung oder ungleichmäßige Skalierung zulässt.
        % Die Reihenfolge ist wichtig: 'matched2' (bewegtes Bild) auf 'matched1' (festes Bild).
        tform = estimateGeometricTransform2D(matched2, matched1, 'similarity'); % Schätzt die Ähnlichkeitstransformation
    
        % 6. Registrierung des zweiten Bildes
        % imref2d erstellt ein Raumbezugsobjekt für das Referenzbild (gray1),
        % um sicherzustellen, dass das transformierte Bild registered2 die gleiche
        % Größe und Ausrichtung wie gray1 hat.
        outputRef = imref2d(size(gray1)); % Definiert den Ausgaberaum für das transformierte Bild
        % imwarp wendet die berechnete Transformation 'tform' auf 'gray2' an.
        % 'OutputView', outputRef sorgt dafür, dass das transformierte Bild
        % korrekt im Referenzraum platziert wird.
        registered2 = imwarp(gray2, tform, 'OutputView', outputRef);
    
        % 7. Gültige Pixelmaske berechnen
        % Da bei der Transformation Pixel außerhalb des ursprünglichen Bildbereichs
        % liegen können oder leere Bereiche entstehen, wird eine Maske erstellt.
        % Diese Maske zeigt, welche Pixel im registrierten Bild tatsächlich
        % aus dem Originalbild stammen und welche "Füllpixel" sind (z.B. Nullen).
        mask = ones(size(gray2)); % Erstellt eine Maske der Größe von gray2, initialisiert mit Einsen
        % Die Maske wird ebenfalls transformiert, um zu sehen, welche Bereiche
        % von gray2 nach der Transformation sichtbar sind.
        warpedMask = imwarp(mask, tform, 'OutputView', outputRef);
        % Pixel, die nach der Transformation einen Wert größer als 0 haben,
        % sind gültige Pixel aus dem ursprünglichen Bild.
        validMask = warpedMask > 0;

    catch ME
        warning(['Registrierung fehlgeschlagen: ', ME.message]);
        registered2 = [];
        validMask = [];
    end
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
