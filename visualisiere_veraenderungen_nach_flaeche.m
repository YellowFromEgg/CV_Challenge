function bilderOut = visualisiere_veraenderungen_nach_flaeche(registrierteBilder, modus, changeThresh, flaechenSchwelle)
% VISUALISIERE_VERAENDERUNGEN_NACH_FLAECHE
% -------------------------------------------------------------------------
% Highlights image regions that change between two registered images and
% colors them according to their area:
%   • small regions  → Yellow (modus = 'klein')
%   • large regions  → Red    (modus = 'groß')
%   • all changes    → Blue   (modus = 'alle')
%
% The coloration is applied as a semi-transparent overlay onto the later
% image of the pair. At the end, a comparison “first ↔ last image” is also
% generated.
%
% Inputs
% ------
% registrierteBilder : Cell array (≥ 2) containing spatially registered
%                      images (grayscale or RGB).
% modus              : 'klein' | 'groß' | 'alle' – selects which masks are
%                      highlighted.
% changeThresh       : Threshold for binarising the difference images
%                      (default 0.18, values in [0 … 1]).
% flaechenSchwelle   : Area threshold in pixels separating “small” vs.
%                      “large” (default 200).
%
% Output
% ------
% bilderOut : Cell array with colour-overlay images
% -------------------------------------------------------------------------

    %% 0) Set default parameters ------------------------------------------
    if nargin < 3 || isempty(changeThresh)
        changeThresh = 0.18;                     % Default threshold
    end
    if nargin < 4 || isempty(flaechenSchwelle)
        flaechenSchwelle = 200;                  % Default area limit
    end

    %% 1) Basic input checks ----------------------------------------------
    numImages = numel(registrierteBilder);
    if numImages < 2
        error('Mindestens zwei registrierte Bilder werden benötigt.');
    end

    %% 2) Normalise images to [0 … 1] -------------------------------------
    for k = 1:numImages
        registrierteBilder{k} = im2double(registrierteBilder{k});
    end

    bilderOut = {};                              % Output container

    %% 3) Pairwise comparison (image i-1 ↔ image i) -----------------------
    for i = 2:numImages
        img1 = registrierteBilder{i-1};
        img2 = registrierteBilder{i};

        % 3a) Absolute per-pixel difference
        diff = imabsdiff(img1, img2);
        if size(diff,3) == 3                     % RGB? → convert to gray
            diffGray = rgb2gray(diff);
        else
            diffGray = diff;
        end

        % 3b) Binary change mask
        mask = imbinarize(diffGray, changeThresh);
        mask = bwareaopen(mask, 10);             % Remove tiny artefacts

        % 3c) Separate regions by size
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

        % 3d) Build colour map & alpha mask
        [verMap, alpha] = baueFarbkarte(kleineMaske, grosseMaske, modus);

        % 3e) Create overlay: colour only where alpha > 0
        overlay = img2;                          % Start with original image
        for c = 1:3
            overlay(:,:,c) = (1 - alpha) .* overlay(:,:,c) + ...
                              alpha .* verMap(:,:,c);
        end

        bilderOut{end+1} = overlay;              % Store result

        % -- Optional debug plot (commented) -------------------------------
        % figure('Name','Change map','NumberTitle','off');
        % imshow(img2); hold on
        % h = imshow(verMap); set(h, 'AlphaData', alpha);
        % title(sprintf('Change image %d → %d  (modus=%s, thr=%.2f)', ...
        %     i-1, i, modus, changeThresh));
    end

    %% 4) Compare first ↔ last image --------------------------------------
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

    % Optional debug plot for first ↔ last comparison (commented)
    % figure('Name','Change map (first vs last)','NumberTitle','off');
    % imshow(img2); hold on
    % h = imshow(verMap); set(h, 'AlphaData', alpha);
    % title(sprintf('Change image 1 → %d  (modus=%s, thr=%.2f)', ...
    %     numImages, modus, changeThresh));

    bilderOut{end+1} = overlay;                  % Append final overlay
end

% ========================================================================
% Helper function: baueFarbkarte
% ========================================================================
function [verMap, alpha] = baueFarbkarte(klein, gross, modus)
% Creates a coloured change map (verMap) and an alpha mask (alpha) depending
% on the chosen modus:
%   'klein' → Yellow for small regions
%   'groß'  → Red    for large regions
%   'alle'  → Blue   for all changed pixels
% ------------------------------------------------------------------------
    verMap = zeros([size(klein), 3]);            % RGB map initialisation
    alpha  = zeros(size(klein));                 % Transparency mask

    switch lower(modus)
        case 'klein'                             % Only small regions
            verMap(:,:,1:2) = repmat(klein, 1, 1, 2);  % Yellow (R+G)
            alpha = klein * 0.5;                 % 50 % opacity

        case 'groß'                              % Only large regions
            verMap(:,:,1) = gross;               % Red
            alpha = gross * 0.7;                 % 70 % opacity

        case 'alle'                              % All regions together
            alle = klein | gross;                % Union of both masks
            verMap(:,:,3) = alle;                % Blue
            alpha = alle * 0.6;                  % 60 % opacity

        otherwise
            error('Ungültiger Modus: %s', modus);
    end
end


