function visualize_schnell_langsam(registrierteBilder, targetAxes)
% Visualisiert Veränderungen (langsam → schnell) in einem
% Bildstapel: schnelle Pixelbewegungen erscheinen rot, langsame blau.
% Alle Ausgaben erfolgen in den übergebenen UIAxes.

    %% 1. Vorverarbeitung -------------------------------------------------
    N = numel(registrierteBilder);                 % Anzahl der Eingabebilder
    if N < 2                                       % Sicherstellen, dass
        error('Mindestens zwei Bilder nötig.');    % ein Bildpaar vorliegt
    end
    cla(targetAxes);                               % Ziel-Achse leeren

    for k = 1:N
        I = im2double(registrierteBilder{k});      % in [0 … 1] skalieren

        if size(I,3) == 3                          % RGB? → Graustufen
            I = rgb2gray(I);
        end

        if k == 1                                  % Stack einmalig anlegen
            [r,c]  = size(I);                      % Bildgröße ermitteln
            imgStack = zeros(r,c,N);               % 3-D-Array vormerken
        end

        imgStack(:,:,k) = I;                       % Slice in Stack schreiben
    end

    %% 2. Veränderungsanalyse ---------------------------------------------
    diffs     = abs(diff(imgStack,1,3));           % |Differenz| jedes Frames
    slow_norm = mat2gray(mean(diffs,3));           % Mittelwert → „langsam“
    fast_norm = mat2gray(max(diffs,[],3));         % Maximum  → „schnell“
    composite = cat(3, fast_norm, ...              % Rot-Kanal  = schnell
                       zeros(size(slow_norm)), ... % Grün-Kanal = 0
                       slow_norm);                 % Blau-Kanal = langsam

    %% 3. Bild anzeigen ----------------------------------------------------
    imshow(composite, ...                          % RGB-Composite darstellen
           'Parent', targetAxes, ...
           'InitialMagnification','fit');
    title(targetAxes, ...
          'Veränderungen: schnell (rot) vs. langsam (blau)', ...
          'FontWeight','bold');

    %% 4. Farb-Legende vorbereiten ----------------------------------------
    fig = ancestor(targetAxes, 'figure');          % übergeordnete Figur
    delete(findall(fig, 'Tag', 'customLegend'));   % evtl. alte Legende löschen

    % Position der Achse (erst in Pixel umschalten, dann zurückstellen)
    axPosNorm  = get(targetAxes, 'Position');      % (hier nicht weiter genutzt)
    axUnitsOld = get(targetAxes, 'Units');         
    set(targetAxes, 'Units', 'pixels');
    axPos = get(targetAxes, 'Position');           % [left bottom width height]
    set(targetAxes, 'Units', axUnitsOld);

    % Zielkoordinaten der Legende (rechts vom Bild, gleiche Höhe)
    legWidth  = 25;                                % Balkenbreite in Pixeln
    legLeft   = axPos(1) + axPos(3) + 10;          % 10 px Abstand zum Bild
    legBottom = axPos(2);
    legHeight = axPos(4);

    % ---------- Beispielcode für eine UI-Achse (auskommentiert) ----------
    %{
    legendAxes = axes('Parent', fig, ...           % Neue Achse für Farbbalken
                      'Units',  'pixels', ...
                      'Position',[legLeft, legBottom, legWidth, legHeight], ...
                      'Tag',    'customLegend');

    n    = 256;                                    % Anzahl Farbstufen
    cmap = [linspace(0,1,n)'  zeros(n,1)  linspace(1,0,n)']; % Blau→Rot
    image(legendAxes, reshape(cmap,[n 1 3]));      % Balken zeichnen
    set(legendAxes, 'YDir', 'normal', ...          % Blau unten, Rot oben
                    'XTick', [], ...
                    'YTick', [1 n], ...
                    'YTickLabel', {'0','max'}, ...
                    'FontWeight', 'bold', ...
                    'Box', 'on');
    ylabel(legendAxes, 'Veränderungsgröße', ...    % Achsenbeschriftung
           'FontWeight', 'bold');
    %}
end
