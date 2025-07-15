function Veraenderungszeitpunkte(registrierteBilder, imageFiles, thr, targetAxes)
% Visualises change pairs using colour coding.
%
% registrierteBilder : cell array with RGB images (already registered)
% imageFiles         : corresponding file names (used for date labels)
% thr                : threshold in the range [0 … 1]
% targetAxes         : UIAxes in which to draw (e.g. inside an App)

    % --- defaults --------------------------------------------------------
    if nargin < 3 || isempty(thr), thr = 0.15; end
    if nargin < 4 || isempty(targetAxes), error('Target UIAxes missing.'); end

    % --- sanity check ----------------------------------------------------
    numImages = numel(registrierteBilder);
    if numImages < 2
        error('At least two registered images are required.');
    end

    % --- convert images to double in [0,1] -------------------------------
    for k = 1:numImages
        registrierteBilder{k} = im2double(registrierteBilder{k});
    end

    % --- colour map & buffers -------------------------------------------
    numPairs  = numImages - 1;
    farben    = hsv(numPairs);                    % unique colour per pair
    [rows, cols, ~] = size(registrierteBilder{1});
    veraenderungskarte = zeros(rows, cols, 3);    % RGB overlay accumulator
    legendenTexte      = strings(1, numPairs);    % legend labels
    minA = 10;                                    % min region size (px)

    % --- extract date labels from file names -----------------------------
    dateLabels = strings(1, numImages);
    for i = 1:numImages
        tokens = regexp(imageFiles{i}, '(\d{1,2})_(\d{4})', 'tokens', 'once');
        if isempty(tokens)
            dateLabels(i) = sprintf('Image %d', i);
        else
            monat = str2double(tokens{1});
            jahr  = str2double(tokens{2});
            dateLabels(i) = sprintf('%02d.%d', monat, jahr);
        end
    end

    % --- compute change masks for every neighbouring pair ----------------
    for i = 2:numImages
        img1 = registrierteBilder{i-1};
        img2 = registrierteBilder{i};
        diffGray = rgb2gray(imabsdiff(img1, img2));
        mask = imbinarize(diffGray, thr);
        mask = bwareaopen(mask, minA);            % remove tiny blobs

        farbe = farben(i-1, :);                   % colour of this pair
        for c = 1:3
            veraenderungskarte(:,:,c) = ...
                veraenderungskarte(:,:,c) + mask * farbe(c);
        end

        legendenTexte(i-1) = sprintf('%s → %s', dateLabels(i-1), dateLabels(i));
    end

    veraenderungskarte = min(veraenderungskarte, 1);  % clamp to [0,1]

    % --- blend overlay with last frame for context -----------------------
    hintergrund = registrierteBilder{end};
    if size(hintergrund,3) == 1                   % ensure RGB background
        hintergrund = repmat(hintergrund, [1 1 3]);
    end
    alpha = 0.6;                                  % overlay opacity
    kombibild = (1 - alpha) * hintergrund + alpha * veraenderungskarte;

    % --- display in UIAxes ----------------------------------------------
    cla(targetAxes);
    imshow(kombibild, 'Parent', targetAxes);
    colorbar(targetAxes, 'off');                  % no colour-bar here
    set(targetAxes, 'XTick', [], 'YTick', []);
    title(targetAxes, sprintf('Change time points (thr = %.2f)', thr));

    % --- build legend using dummy markers --------------------------------
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

