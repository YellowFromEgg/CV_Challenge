function transformedSegmented = Transform_Segmented_Images(segmentedMaps, tformList, refSize)
%TRANSFORM_SEGMENTED_IMAGES Apply affine2d transforms to segmented images.
%
% Inputs:
%   segmentedMaps - cell array of segmented images (label maps, RGB, etc.)
%   tformList     - cell array of affine2d transformations
%   refSize       - [height, width] size of the reference frame
%
% Output:
%   transformedSegmented - cell array of transformed segmentation images

    numImages = numel(segmentedMaps);
    transformedSegmented = cell(1, numImages);
    outputRef = imref2d(refSize); % spatial reference frame

    for k = 1:numImages
        segImg = segmentedMaps{k};
        tform = tformList{k};

        if isempty(segImg) || isempty(tform) || ~isa(tform, 'affine2d')
            transformedSegmented{k} = [];
            continue;
        end

        if size(segImg, 3) == 3
            % RGB image: apply transform channel-wise
            warpedImg = zeros(refSize(1), refSize(2), 3, 'like', segImg);
            for ch = 1:3
                warpedImg(:,:,ch) = imwarp(segImg(:,:,ch), tform, ...
                    'OutputView', outputRef, ...
                    'InterpolationMethod', 'nearest', ...
                    'FillValues', 0);
            end
        else
            % Single-channel (label map)
            warpedImg = imwarp(segImg, tform, ...
                'OutputView', outputRef, ...
                'InterpolationMethod', 'nearest', ...
                'FillValues', 0);
        end

        transformedSegmented{k} = warpedImg;
    end
end
