function timelapseSlider(imgCell)
% TIMELAPSESLIDER  Displays an interactive time-lapse viewer.
%   timelapseSlider(imgCell) expects a cell array {I1,…,IN} containing
%   already registered images of class double, in the range [0 … 1]
%   (grayscale or RGB).

    % --- Basic consistency check ----------------------------------------
    N = numel(imgCell);
    if N < 2
        error('Mindestens zwei Bilder erforderlich.');  % message unchanged
    end

    % --- Make sure every image is double in the range [0 … 1] ------------
    for k = 1:N
        imgCell{k} = im2double(imgCell{k});
    end

    % --- Create main figure and axes ------------------------------------
    hFig = figure('Name','Timelapse Viewer', ...
                  'Color','w', 'NumberTitle','off', 'MenuBar','none');
    ax = axes('Parent', hFig, 'Position', [0.05 0.15 0.9 0.8]);
    imHandle = imshow(imgCell{1}, 'Parent', ax);
    title(ax, sprintf('Frame 1 / %d', N), 'FontSize', 12);

    % --- Slider UI element ----------------------------------------------
    sld = uicontrol('Parent', hFig, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.05], ...
        'Min', 1, 'Max', N, 'Value', 1, ...
        'SliderStep', [1/(N-1) , 5/(N-1)]);

    % --- Attach callback to the slider ----------------------------------
    addlistener(sld, 'Value', 'PostSet', @updateFrame);

    % --------------------------------------------------------------------
    function updateFrame(~,~)
    % Updates the displayed image whenever the slider value changes.
        v      = get(sld, 'Value');      % Current (possibly fractional) value
        i      = floor(v);               % Lower integer frame index
        alpha  = v - i;                  % Fractional part for interpolation

        if i >= N                        % Slider at the very end
            img       = imgCell{N};
            frameText = sprintf('Frame %d / %d', N, N);
        else                             % Blend between frames i and i+1
            A         = imgCell{i};
            B         = imgCell{i+1};
            img       = (1-alpha) * A + alpha * B;
            frameText = sprintf('Frame %.2f / %d', v, N);
        end

        set(imHandle, 'CData', img);     % Update displayed image
        title(ax, frameText, 'FontSize', 12);
        drawnow;
    end
end
