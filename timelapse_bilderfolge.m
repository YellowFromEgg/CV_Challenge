function frames = timelapse_bilderfolge(images, showDiff, changeThresh)
% TIMELAPSE_BILDERFOLGE  Bereitet Differenz- oder Originalframes für Timelapse vor.
%
%   frames = timelapse_bilderfolge(images, showDiff, changeThresh)
%   - images: Zellarray mit RGB-Bildern
%   - showDiff: true = Differenzansicht, false = Originalbilder
%   - changeThresh: Schwellwert für Differenz (0.0–1.0)
%
%   Rückgabe: Zellarray mit vorbereiteten Frames

    if nargin < 2 || isempty(showDiff), showDiff = false; end
    if nargin < 3 || isempty(changeThresh), changeThresh = 0.15; end

    numImages = numel(images);
    if numImages < 2
        error('Mindestens zwei Bilder erforderlich.');
    end

    frames = cell(1, numImages - 1);
    farbe = [1 0 0]; alpha = 0.6;

    for i = 2:numImages
        img1 = im2double(images{i-1});
        img2 = im2double(images{i});

        if showDiff
            diff = imabsdiff(img1, img2);
            if size(diff,3) == 3, diff = rgb2gray(diff); end
            mask = imbinarize(diff, changeThresh);
            mask = bwareaopen(mask, 10);

            overlay = zeros(size(img2));
            for c = 1:3
                overlay(:,:,c) = mask * farbe(c);
            end

            frame = (1 - alpha) * img2 + alpha * overlay;
        else
            frame = img2;
        end

        frames{i - 1} = frame;
    end
end

%% Lokale Funktion zum Abspielen
function playTimelapseFrames(frames, targetAxes)
% SPIELT vorbereitete Frames als Timelapse ab
    for i = 1:numel(frames)
        imshow(frames{i}, 'Parent', targetAxes);
        title(targetAxes, sprintf('Frame %d von %d', i+1, numel(frames)+1), ...
            'FontWeight','bold');
        pause(1);
    end
end
