function Veraenderungszeitpunkte(registrierteBilder, imageFiles, thr, targetAxes)
% Veraenderungszeitpunkte – Visualisiert Veränderungspaare in Farbcodierung
%
% registrierteBilder : Cell-Array mit RGB-Bildern
% imageFiles         : Bilddateinamen (für Datumsanzeige)
% thr                : Schwellenwert [0…1]
% targetAxes         : UIAxes zum Zeichnen in App

    if nargin < 3 || isempty(thr), thr = 0.15; end
    if nargin < 4 || isempty(targetAxes), error('Target UIAxes fehlt.'); end

    numImages = numel(registrierteBilder);
    if numImages < 2
        error('Mindestens zwei registrierte Bilder werden benötigt.');
    end

    % Skalierung & Initialisierung
    for k = 1:numImages
        registrierteBilder{k} = im2double(registrierteBilder{k});
    end

    numPairs = numImages - 1;
    farben = hsv(numPairs);
    [rows, cols, ~] = size(registrierteBilder{1});
    veraenderungskarte = zeros(rows, cols, 3);
    legendenTexte = strings(1, numPairs);
    minA = 10;  % Mindestfläche (Pixel)

    % Datumsbeschriftungen aus imageFiles
    dateLabels = strings(1, numImages);
    for i = 1:numImages
        tokens = regexp(imageFiles{i}, '(\d{1,2})_(\d{4})', 'tokens', 'once');
        if isempty(tokens)
            dateLabels(i) = sprintf('Bild %d', i);
        else
            monat = str2double(tokens{1});
            jahr = str2double(tokens{2});
            dateLabels(i) = sprintf('%02d.%d', monat, jahr);
        end
    end

    % Veränderungen berechnen
    for i = 2:numImages
        img1 = registrierteBilder{i-1};
        img2 = registrierteBilder{i};
        diffGray = rgb2gray(imabsdiff(img1, img2));
        mask = imbinarize(diffGray, thr);
        mask = bwareaopen(mask, minA);

        farbe = farben(i-1, :);
        for c = 1:3
            veraenderungskarte(:,:,c) = ...
                veraenderungskarte(:,:,c) + mask * farbe(c);
        end

        legendenTexte(i-1) = sprintf('%s → %s', dateLabels(i-1), dateLabels(i));
    end

    veraenderungskarte = min(veraenderungskarte, 1);
    hintergrund = registrierteBilder{end};
    if size(hintergrund,3) == 1
        hintergrund = repmat(hintergrund, [1 1 3]);
    end

    alpha = 0.6;
    kombibild = (1 - alpha) * hintergrund + alpha * veraenderungskarte;

    % Anzeige in UIAxes
    cla(targetAxes);
    imshow(kombibild, 'Parent', targetAxes);
    colorbar(targetAxes, 'off');  % Keine Colorbar anzeigen
    set(targetAxes, 'XTick', [], 'YTick', []);
    title(targetAxes, sprintf('Veränderungszeitpunkte (thr = %.2f)', thr));

    % Legende (mit Dummy-Markern)
    hold(targetAxes, 'on');
    dummyPlots = gobjects(1, numPairs);
    for j = 1:numPairs
        dummyPlots(j) = plot(targetAxes, NaN, NaN, 's', ...
            'MarkerFaceColor', farben(j,:), ...
            'MarkerEdgeColor', 'k', ...
            'DisplayName', legendenTexte(j));
    end
    legend(targetAxes, dummyPlots, ...
        'Location', 'northeastoutside', ...
        'Orientation', 'vertical', ...
        'Box', 'off');
    hold(targetAxes, 'off');
end
