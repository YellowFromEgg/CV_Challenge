function visualize_schnell_langsam(registrierteBilder, targetAxes)
% Visualizes changes (slow → fast) in an image stack: fast pixel motion
% appears red, slow motion appears blue. All output is rendered in the
% provided UIAxes.

    %% 1. Pre-processing ---------------------------------------------------
    N = numel(registrierteBilder);                 % Number of input images
    if N < 2                                       % Ensure at least one image pair exists
        error('Mindestens zwei Bilder nötig.');    
    end
    cla(targetAxes);                               % Clear target axes

    for k = 1:N
        I = im2double(registrierteBilder{k});      % Normalize to [0 … 1]

        if size(I,3) == 3                          % RGB → convert to grayscale
            I = rgb2gray(I);
        end

        if k == 1                                  % Allocate stack once
            [r,c]  = size(I);                      % Determine image size
            imgStack = zeros(r,c,N);               % Reserve 3-D array
        end

        imgStack(:,:,k) = I;                       % Insert slice into stack
    end

    %% 2. Change analysis --------------------------------------------------
    diffs     = abs(diff(imgStack,1,3));           % |Difference| for every frame pair
    slow_norm = mat2gray(mean(diffs,3));           % Mean    → “slow” map
    fast_norm = mat2gray(max(diffs,[],3));         % Maximum → “fast” map
    composite = cat(3, fast_norm, ...              % Red   channel  = fast
                       zeros(size(slow_norm)), ... % Green channel = 0
                       slow_norm);                 % Blue  channel  = slow

    %% 3. Display composite ------------------------------------------------
    imshow(composite, ...                          % Show RGB composite
           'Parent', targetAxes, ...
           'InitialMagnification','fit');
    title(targetAxes, ...
          'Veränderungen: schnell (rot) vs. langsam (blau)', ...
          'FontWeight','bold');

    %% 4. Prepare color legend --------------------------------------------
    fig = ancestor(targetAxes, 'figure');          % Parent figure handle
    delete(findall(fig, 'Tag', 'customLegend'));   % Remove any existing legend axes

    % Get axis position (temporarily switch to pixel units, then restore)
    axPosNorm  = get(targetAxes, 'Position');      
    axUnitsOld = get(targetAxes, 'Units');         
    set(targetAxes, 'Units', 'pixels');
    axPos = get(targetAxes, 'Position');           % [left bottom width height]
    set(targetAxes, 'Units', axUnitsOld);

    % Legend coordinates (to the right of the image, same height)
    legWidth  = 25;                                % Bar width in pixels
    legLeft   = axPos(1) + axPos(3) + 10;          % 10-px gap from the image
    legBottom = axPos(2);
    legHeight = axPos(4);

   
end

