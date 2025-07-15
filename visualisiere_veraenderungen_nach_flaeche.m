function bilderOut = visualisiere_veraenderungen_nach_flaeche(registrierteBilder, modus, changeThresh, flaechenSchwelle)
% VISUALISIERE_VERAENDERUNGEN_NACH_FLAECHE
% -------------------------------------------------------------------------
% Markiert Bildbereiche, die sich zwischen zwei registrierten Bildern
% verändert haben, und färbt sie abhängig von ihrer Fläche ein:
%   • kleine Flächen   → Gelb   (modus = 'klein')
%   • große  Flächen   → Rot    (modus = 'groß')
%   • alle Veränderungen → Blau (modus = 'alle')
%
% Die Färbung wird als halbtransparentes Overlay auf das jeweils spätere
% Bild gelegt. Am Ende wird zusätzlich der Vergleich „erstes ↔ letztes
% Bild“ ausgegeben.
%
% Eingaben
% --------
% registrierteBilder : Cell-Array (≥ 2) mit räumlich bereits registrierten
%                      Bildern (grau oder RGB).
% modus              : 'klein' | 'groß' | 'alle'  → bestimmt, welche
%                      Masken farblich hervorgehoben werden.
% changeThresh       : Schwelle zum Binarisieren der Differenzbilder
%                      (Default 0.18, Werte in [0 … 1]).
% flaechenSchwelle   : Flächen­schwelle in Pixeln – trennt „klein“ vs.
%                      „groß“ (Default 200).
%
% Ausgabe
% -------
% bilderOut : Cell-Array mit farbigen Overlay-Bildern
% -------------------------------------------------------------------------

    %% 0) Standardwerte setzen --------------------------------------------
    if nargin < 3 || isempty(changeThresh)
        changeThresh = 0.18;                     % Default-Schwelle
    end
    if nargin < 4 || isempty(flaechenSchwelle)
        flaechenSchwelle = 200;                  % Default-Flächenlimit
    end

    %% 1) Eingaben prüfen --------------------------------------------------
    numImages = numel(registrierteBilder);
    if numImages < 2
        error('Mindestens zwei registrierte Bilder werden benötigt.');
    end

    %% 2) Bilder in [0 … 1] skalieren --------------------------------------
    for k = 1:numImages
        registrierteBilder{k} = im2double(registrierteBilder{k});
    end

    bilderOut = {};                              % Rückgabe-Container

    %% 3) Paarweiser Vergleich (Bild i-1 ↔ Bild i) -------------------------
    for i = 2:numImages
        img1 = registrierteBilder{i-1};
        img2 = registrierteBilder{i};

        % 3a) Absoluter Differenzwert pro Pixel
        diff = imabsdiff(img1, img2);
        if size(diff,3) == 3                     % RGB? → Graustufen
            diffGray = rgb2gray(diff);
        else
            diffGray = diff;
        end

        % 3b) Binäre Maske der Veränderungen
        mask = imbinarize(diffGray, changeThresh);
        mask = bwareaopen(mask, 10);             % Kleinst­artefakte filtern

        % 3c) Flächen nach Größe trennen
        cc    = bwconncomp(mask);
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

        % 3d) Farbkarte & Alphamaske aufbauen
        [verMap, alpha] = baueFarbkarte(kleineMaske, grosseMaske, modus);

        % 3e) Overlay erzeugen: Nur wo alpha > 0 wird eingefärbt
        overlay = img2;                          % Start = Originalbild
        for c = 1:3
            overlay(:,:,c) = (1 - alpha) .* overlay(:,:,c) + ...
                              alpha .* verMap(:,:,c);
        end

        bilderOut{end+1} = overlay;              % Ergebnis abspeichern

        % -- Debug-Plot (auskommentiert) -----------------------------------
        % figure('Name','Veränderungskarte','NumberTitle','off');
        % imshow(img2); hold on
        % h = imshow(verMap); set(h, 'AlphaData', alpha);
        % title(sprintf('Veränderung Bild %d → %d  (Modus=%s, Thr=%.2f)', ...
        %     i-1, i, modus, changeThresh));
    end

    %% 4) Vergleich erstes ↔ letztes Bild ----------------------------------
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

    cc    = bwconncomp(mask);
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
        overlay(:,:,c) = (1 - alpha) .* overlay(:,:,c) + ...
                          alpha .* verMap(:,:,c);
    end

    % figure('Name','Veränderungskarte (Erstes vs Letztes)','NumberTitle','off');
    % imshow(img2); hold on
    % h = imshow(verMap); set(h, 'AlphaData', alpha);
    % title(sprintf('Veränderung Bild 1 → %d  (Modus=%s, Thr=%.2f)', ...
    %     numImages, modus, changeThresh));

    bilderOut{end+1} = overlay;                  % Letztes Overlay anhängen
end

% ========================================================================
% Hilfsfunktion: baueFarbkarte
% ========================================================================
function [verMap, alpha] = baueFarbkarte(klein, gross, modus)
% Erstellt eine farbige Veränderungskarte (verMap) und eine
% zugehörige Alphamaske (alpha) abhängig vom gewählten Modus:
%   'klein' → Gelb für kleine Flächen
%   'groß'  → Rot  für große  Flächen
%   'alle'  → Blau für alle veränderten Pixel
% ------------------------------------------------------------------------
    verMap = zeros([size(klein), 3]);            % RGB-Array initialisieren
    alpha  = zeros(size(klein));                 % Transparenz­maske

    switch lower(modus)
        case 'klein'                             % Nur kleine Flächen
            verMap(:,:,1:2) = repmat(klein, 1, 1, 2);  % Gelb (R+G)
            alpha = klein * 0.5;                 % 50 % Deckkraft

        case 'groß'                              % Nur große Flächen
            verMap(:,:,1) = gross;               % Rot
            alpha = gross * 0.7;                 % 70 % Deckkraft

        case 'alle'                              % Beide Masken zusammen
            alle = klein | gross;                % Vereinigung
            verMap(:,:,3) = alle;                % Blau
            alpha = alle * 0.6;                  % 60 % Deckkraft

        otherwise
            error('Ungültiger Modus: %s', modus);
    end
end

