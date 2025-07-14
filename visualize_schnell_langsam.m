function visualize_schnell_langsam(registrierteBilder, targetAxes)
% Visualisiert Veränderungen (langsam → schnell) mit farblicher Legende direkt daneben

    %% 1. Vorverarbeitung
    N = numel(registrierteBilder);
    if N < 2, error('Mindestens zwei Bilder nötig.'); end
    cla(targetAxes);

    for k = 1:N
        I = im2double(registrierteBilder{k});
        if size(I,3) == 3, I = rgb2gray(I); end
        if k == 1, [r,c] = size(I); imgStack = zeros(r,c,N); end
        imgStack(:,:,k) = I;
    end

    %% 2. Veränderungsanalyse
    diffs     = abs(diff(imgStack,1,3));
    slow_norm = mat2gray(mean(diffs,3));
    fast_norm = mat2gray(max(diffs,[],3));
    composite = cat(3, fast_norm, zeros(size(slow_norm)), slow_norm);  % Rot = schnell, Blau = langsam

    %% 3. Bild anzeigen
    imshow(composite, 'Parent', targetAxes, 'InitialMagnification','fit');
    title(targetAxes, 'Veränderungen: schnell (rot) vs. langsam (blau)', 'FontWeight','bold');

    %% 4. Farb-Legende (rechts daneben, in App) erzeugen
    fig = ancestor(targetAxes, 'figure');
    delete(findall(fig, 'Tag', 'customLegend'));  % Alte entfernen

    % Position der Achse holen (in Pixeln)
    axPosNorm = get(targetAxes, 'Position');
    axUnitsOld = get(targetAxes, 'Units');
    set(targetAxes, 'Units', 'pixels');
    axPos = get(targetAxes, 'Position');
    set(targetAxes, 'Units', axUnitsOld);  % zurückstellen

    % Legendenposition daneben (Pixelbasiert auf Höhe des Bildes)
    legWidth = 25;
    legLeft = axPos(1) + axPos(3) + 10;
    legBottom = axPos(2);
    legHeight = axPos(4);

    % % % Neues UIAxes für Farbbalken
    % % legendAxes = axes('Parent', app.UIFigure, ...
    % %                   'Units', 'pixels', ...
    % %                   'Position', [legLeft, legBottom, legWidth, legHeight], ...
    % %                   'Tag', 'customLegend');
    % % 
    % % % Farbverlauf: Blau → Rot
    % % n = 256;
    % % cmap = [linspace(0,1,n)' zeros(n,1) linspace(1,0,n)'];
    % % image(legendAxes, reshape(cmap, [n 1 3]));
    % % set(legendAxes, 'YDir', 'normal', ...
    % %                 'XTick', [], ...
    % %                 'YTick', [1 n], ...
    % %                 'YTickLabel', {'0','max'}, ...
    % %                 'FontWeight','bold', ...
    % %                 'Box','on');
    % % ylabel(legendAxes, 'Veränderungsgröße', 'FontWeight','bold');
end
