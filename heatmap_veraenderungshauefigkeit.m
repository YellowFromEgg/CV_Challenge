function heatmap_veraenderungshauefigkeit(registrierteBilder, schwelle, axesHandle)
% Heat-map of per-pixel change frequency across all registered image pairs

    % --- default threshold ------------------------------------------------
    if nargin < 2 || isempty(schwelle)
        schwelle = 0.2;
    end

    % --- handle optional target axes -------------------------------------
    if nargin < 3 || isempty(axesHandle)
        figure;                          % create new figure + axes
        useUIAxes = false;
    else
        cla(axesHandle);                 % clear previous image
        useUIAxes = true;
    end

    % --- basic input check -----------------------------------------------
    numImages = numel(registrierteBilder);
    if numImages < 2
        error('Mindestens zwei registrierte Bilder erforderlich.'); % need ≥ 2 frames
    end

    % --- allocate counter for each pixel ---------------------------------
    [h, w, ~] = size(registrierteBilder{1});
    veraenderungsZaehler = zeros(h, w);  % change count per pixel

    % --- accumulate changes over all consecutive pairs -------------------
    for i = 2:numImages
        img1 = im2double(registrierteBilder{i-1});
        img2 = im2double(registrierteBilder{i});
        diffGray = rgb2gray(abs(img1 - img2));    % absolute difference (gray)
        veraenderungsMaske = diffGray > schwelle; % binary change mask
        veraenderungsZaehler = veraenderungsZaehler + veraenderungsMaske;
    end

    heatmapNorm = veraenderungsZaehler / (numImages - 1);  % scale to [0,1]

    % 🔍 display -----------------------------------------------------------
    if useUIAxes
        imagesc(axesHandle, heatmapNorm);
        colormap(axesHandle, hot);
        axis(axesHandle, 'image');
        axis(axesHandle, 'off');
        title(axesHandle, ...
              'Heat-map of per-pixel change frequency', ...
              'FontWeight', 'bold');

        % add color-bar
        cb = colorbar(axesHandle, 'Location', 'eastoutside');
    else
        imagesc(heatmapNorm);
        colormap hot;
        axis image off;
        title('Heat-map of per-pixel change frequency', 'FontWeight', 'bold');
        cb = colorbar;
    end

    %  format color-bar ticks -------------------------------------------
    cb.Label.String = sprintf(['Change frequency\n' ...
                               '0 = never changed, %d = changed every pair'], ...
                               numImages - 1);
    cb.Ticks = linspace(0, 1, numImages);
    cb.TickLabels = arrayfun(@(x) sprintf('%dx', x), 0:numImages - 1, ...
                             'UniformOutput', false);
end

