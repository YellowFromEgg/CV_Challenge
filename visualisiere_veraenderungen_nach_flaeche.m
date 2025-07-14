function bilderOut = visualisiere_veraenderungen_nach_flaeche(registrierteBilder, modus, changeThresh, flaechenSchwelle)
    if nargin < 3 || isempty(changeThresh)
        changeThresh = 0.18;
    end
    if nargin < 4 || isempty(flaechenSchwelle)
        flaechenSchwelle = 200;
    end

    numImages = numel(registrierteBilder);
    if numImages < 2
        error('Mindestens zwei registrierte Bilder werden benötigt.');
    end

    % Bilder auf [0…1] normalisieren
    for k = 1:numImages
        registrierteBilder{k} = im2double(registrierteBilder{k});
    end

    bilderOut = {};  % Rückgabe-Container

    for i = 2:numImages
        img1 = registrierteBilder{i-1};
        img2 = registrierteBilder{i};

        diff = imabsdiff(img1, img2);
        if size(diff,3) == 3
            diffGray = rgb2gray(diff);
        else
            diffGray = diff;
        end

        mask = imbinarize(diffGray, changeThresh);
        mask = bwareaopen(mask, 10);

        cc = bwconncomp(mask);
        stats = regionprops(cc, 'Area', 'PixelIdxList');

        kleineMaske = false(size(diffGray));
        grosseMaske = false(size(diffGray));
        for k = 1:numel(stats)
            if stats(k).Area < flaechenSchwelle
                kleineMaske(stats(k).PixelIdxList) = true;
            else
                grosseMaske(stats(k).PixelIdxList) = true;
            end
        end

        [verMap, alpha] = baueFarbkarte(kleineMaske, grosseMaske, modus);

        % NEU: Nur im Overlaybereich einfärben, ansonsten Original lassen
        overlay = img2;
        for c = 1:3
            overlay(:,:,c) = (1 - alpha) .* overlay(:,:,c) + alpha .* verMap(:,:,c);
        end

        bilderOut{end+1} = overlay;

        % % Debug-Figur anzeigen
        % figure('Name','Veränderungskarte','NumberTitle','off');
        % imshow(img2); hold on
        % h = imshow(verMap); set(h, 'AlphaData', alpha);
        % title(sprintf('Veränderung Bild %d → %d  (Modus=%s, Thr=%.2f)', ...
        %     i-1, i, modus, changeThresh));
    end

    % Vergleich erstes ↔ letztes
    img1 = registrierteBilder{1};
    img2 = registrierteBilder{end};
    diff = imabsdiff(img1, img2);
    if size(diff, 3) == 3
        diffGray = rgb2gray(diff);
    else
        diffGray = diff;
    end
    mask = imbinarize(diffGray, changeThresh);
    mask = bwareaopen(mask, 10);

    cc = bwconncomp(mask);
    stats = regionprops(cc, 'Area', 'PixelIdxList');

    kleineMaske = false(size(diffGray));
    grosseMaske = false(size(diffGray));
    for k = 1:numel(stats)
        if stats(k).Area < flaechenSchwelle
            kleineMaske(stats(k).PixelIdxList) = true;
        else
            grosseMaske(stats(k).PixelIdxList) = true;
        end
    end

    [verMap, alpha] = baueFarbkarte(kleineMaske, grosseMaske, modus);

    overlay = img2;
    for c = 1:3
        overlay(:,:,c) = (1 - alpha) .* overlay(:,:,c) + alpha .* verMap(:,:,c);
    end

    % figure('Name','Veränderungskarte (Erstes vs Letztes)','NumberTitle','off');
    % imshow(img2); hold on
    % h = imshow(verMap); set(h, 'AlphaData', alpha);
    % title(sprintf('Veränderung Bild 1 → %d  (Modus=%s, Thr=%.2f)', ...
    %     numImages, modus, changeThresh));

    bilderOut{end+1} = overlay; % optional hinzufügen
end

% -------------------------------------------------------------------------
function [verMap, alpha] = baueFarbkarte(klein, gross, modus)
    verMap = zeros([size(klein), 3]);
    alpha  = zeros(size(klein));

    switch lower(modus)
        case 'klein'
            verMap(:,:,1:2) = repmat(klein, 1, 1, 2);  % Gelb
            alpha = klein * 0.5;

        case 'groß'
            verMap(:,:,1) = gross;                    % Rot
            alpha = gross * 0.7;

        case 'alle'
            alle = klein | gross;
            verMap(:,:,3) = alle;                     % Blau
            alpha = alle * 0.6;

        otherwise
            error('Ungültiger Modus: %s', modus);
    end
end
