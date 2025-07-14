function heatmap_veraenderungshauefigkeit(registrierteBilder, schwelle, axesHandle)
% Heatmap der Veränderungshäufigkeit pro Pixel über alle Bildpaare

    if nargin < 2 || isempty(schwelle)
        schwelle = 0.2;
    end

    if nargin < 3 || isempty(axesHandle)
        figure;
        useUIAxes = false;
    else
        cla(axesHandle);  % Vorheriges Bild löschen
        useUIAxes = true;
    end

    numImages = numel(registrierteBilder);
    if numImages < 2
        error('Mindestens zwei registrierte Bilder erforderlich.');
    end

    [h, w, ~] = size(registrierteBilder{1});
    veraenderungsZaehler = zeros(h, w);

    for i = 2:numImages
        img1 = im2double(registrierteBilder{i-1});
        img2 = im2double(registrierteBilder{i});
        diffGray = rgb2gray(abs(img1 - img2));
        veraenderungsMaske = diffGray > schwelle;
        veraenderungsZaehler = veraenderungsZaehler + veraenderungsMaske;
    end

    heatmapNorm = veraenderungsZaehler / (numImages - 1);  % Skaliert auf [0,1]

    % 🔍 Darstellung
    if useUIAxes
        imagesc(axesHandle, heatmapNorm);
        colormap(axesHandle, hot);
        axis(axesHandle, 'image');
        axis(axesHandle, 'off');
        title(axesHandle, 'Heatmap der Veränderungshäufigkeit pro Pixel', 'FontWeight', 'bold');

        % Colorbar hinzufügen
        cb = colorbar(axesHandle, 'Location', 'eastoutside');
    else
        imagesc(heatmapNorm);
        colormap hot;
        axis image off;
        title('Heatmap der Veränderungshäufigkeit pro Pixel', 'FontWeight', 'bold');
        cb = colorbar;
    end

    % 🟡 Colorbar-Ticks formatieren
    cb.Label.String = sprintf('Veränderungshäufigkeit\n0 = keine Veränderung, %d = Veränderung bei jedem Vergleich', numImages - 1);
    cb.Ticks = linspace(0, 1, numImages);
    cb.TickLabels = arrayfun(@(x) sprintf('%dx', x), 0:numImages - 1, 'UniformOutput', false);
end
