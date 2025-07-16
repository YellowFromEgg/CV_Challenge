function frames = timelapse_bilderfolge(images, showDiff, changeThresh)
% TIMELAPSE_BILDERFOLGE  Prepares either difference-based or original
% frames that can be played back as a time-lapse sequence.
%
%   frames = timelapse_bilderfolge(images, showDiff, changeThresh)
%   - images       : cell array containing RGB images
%   - showDiff     : true  → display change overlay,
%                    false → display original images
%   - changeThresh : threshold used for the difference mask (range 0.0–1.0)
%
%   Output
%   ------
%   frames : cell array of processed frames, ready for playback

    % Set default arguments when they are not provided
    if nargin < 2 || isempty(showDiff),   showDiff     = false;  end
    if nargin < 3 || isempty(changeThresh), changeThresh = 0.15;  end

    % Basic sanity check: at least one image pair is required
    numImages = numel(images);
    if numImages < 2
        error('Mindestens zwei Bilder erforderlich.');
    end

    frames = cell(1, numImages - 1);      % Pre-allocate output container
    farbe  = [1 0 0];                     % Red colour for the overlay
    alpha  = 0.6;                         % Overlay opacity

    % Loop through consecutive image pairs
    for i = 2:numImages
        img1 = im2double(images{i-1});
        img2 = im2double(images{i});

        if showDiff
            % --- Build binary change mask --------------------------------
            diff = imabsdiff(img1, img2);
            if size(diff,3) == 3, diff = rgb2gray(diff); end
            mask = imbinarize(diff, changeThresh);
            mask = bwareaopen(mask, 10);         % Remove tiny artefacts

            % --- Colour overlay (red) where the mask is true -------------
            overlay = zeros(size(img2));
            for c = 1:3
                overlay(:,:,c) = mask * farbe(c);
            end

            % --- Combine original and overlay ----------------------------
            frame = (1 - alpha) * img2 + alpha * overlay;
        else
            % No difference view → pass the original image
            frame = img2;
        end

        frames{i - 1} = frame;                   % Store processed frame
    end
end

%% Local helper: playTimelapseFrames
function playTimelapseFrames(frames, targetAxes)
% Plays the prepared frames as a simple time-lapse animation
    for i = 1:numel(frames)
        imshow(frames{i}, 'Parent', targetAxes);
        title(targetAxes, ...
              sprintf('Frame %d of %d', i+1, numel(frames)+1), ...
              'FontWeight','bold');
        pause(1);                                % Wait 1 s between frames
    end
end
