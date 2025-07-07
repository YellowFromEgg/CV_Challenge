function segmentedOverlapMasks = Transform_Segmented_Images(segmentedMaps, tformList, refIdx)
% TRANSFORM_SEGMENTED_IMAGES Applies affine transforms to segmented images.
% If a transform is missing, the original image is returned unchanged.
%
% Inputs:
%   segmentedMaps - cell array of original segmented images (label maps)
%   tformList     - cell array of transform objects (affine2d or similitude2d, etc.)
%   refIdx        - index of reference image to determine output size
%
% Output:
%   segmentedOverlapMasks - cell array of transformed segmented images

    numImages = numel(segmentedMaps);
    segmentedOverlapMasks = cell(1, numImages);

    % Use reference size to build spatial reference
    refSize = size(segmentedMaps{refIdx});
    if numel(refSize) == 2
        refSize(3) = 1;
    end
    outputRef = imref2d(refSize(1:2));  % height × width only

    for k = 1:numImages
        segImg = segmentedMaps{k};
        tformRaw = tformList{k};

        if isempty(segImg)
            segmentedOverlapMasks{k} = [];
            continue;
        end

        % Fallback: If no transform, store original
        if isempty(tformRaw)
            segmentedOverlapMasks{k} = segImg;
            continue;
        end

        % Convert to affine2d if needed
        try
            if ~isa(tformRaw, 'affine2d')
                tform = affine2d(tformRaw.T);
            else
                tform = tformRaw;
            end
        catch
            segmentedOverlapMasks{k} = [];
            continue;
        end

        % Apply transformation
        try
            if size(segImg, 3) == 3
                warpedImg = zeros(refSize(1), refSize(2), 3, 'like', segImg);
                for ch = 1:3
                    warpedImg(:,:,ch) = imwarp(segImg(:,:,ch), tform, ...
                        'OutputView', outputRef, ...
                        'InterpolationMethod', 'nearest', ...
                        'FillValues', 0);
                end
            else
                warpedImg = imwarp(segImg, tform, ...
                    'OutputView', outputRef, ...
                    'InterpolationMethod', 'nearest', ...
                    'FillValues', 0);
            end

            segmentedOverlapMasks{k} = warpedImg;
        catch
            segmentedOverlapMasks{k} = [];
        end
    end
end
