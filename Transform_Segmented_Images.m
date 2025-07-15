function segmentedOverlapMasks = Transform_Segmented_Images(segmentedMaps, tformList, refIdx)
% TRANSFORM_SEGMENTED_IMAGES Applies affine transformations to segmented label maps.
% Each image is warped to align with a common spatial reference (e.g., for mosaicking).
%
% Inputs:
%   segmentedMaps - cell array of segmented label maps (each a 2D or 3D array)
%   tformList     - cell array of geometric transform objects (affine2d, etc.)
%   refIdx        - index of reference image used to define the output canvas size
%
% Output:
%   segmentedOverlapMasks - cell array of transformed segmented images

    numImages = numel(segmentedMaps);  % Number of input images
    segmentedOverlapMasks = cell(1, numImages);  % Initialize output cell array

    % Step 1: Determine spatial reference from reference image size
    refSize = size(segmentedMaps{refIdx});  % Size of the reference image
    if numel(refSize) == 2
        refSize(3) = 1;  % Ensure consistent handling of 2D/3D (grayscale/RGB)
    end
    outputRef = imref2d(refSize(1:2));  % Only height × width are used for reference

    % Step 2: Loop through each image and apply its corresponding transform
    for k = 1:numImages
        segImg = segmentedMaps{k};     % Current label map
        tformRaw = tformList{k};       % Corresponding transformation

        % Skip if the input segmented image is missing
        if isempty(segImg)
            segmentedOverlapMasks{k} = [];
            continue;
        end

        % If no transformation provided, just return the original image
        if isempty(tformRaw)
            segmentedOverlapMasks{k} = segImg;
            continue;
        end

        % Step 3: Convert to affine2d if needed
        try
            if ~isa(tformRaw, 'affine2d')
                tform = affine2d(tformRaw.T);  % Wrap using transformation matrix
            else
                tform = tformRaw;
            end
        catch
            % Catch any invalid transformation objects
            segmentedOverlapMasks{k} = [];
            continue;
        end

        % Step 4: Apply geometric transformation to the label map
        try
            if size(segImg, 3) == 3
                % For 3D (RGB or multi-channel label maps), warp each channel
                warpedImg = zeros(refSize(1), refSize(2), 3, 'like', segImg);
                for ch = 1:3
                    warpedImg(:,:,ch) = imwarp(segImg(:,:,ch), tform, ...
                        'OutputView', outputRef, ...
                        'InterpolationMethod', 'nearest', ...  % Nearest for label integrity
                        'FillValues', 0);  % Fill outside bounds with 0
                end
            else
                % For 2D label maps
                warpedImg = imwarp(segImg, tform, ...
                    'OutputView', outputRef, ...
                    'InterpolationMethod', 'nearest', ...
                    'FillValues', 0);
            end

            segmentedOverlapMasks{k} = warpedImg;  % Store transformed label map
        catch
            % Handle rare warping failures gracefully
            segmentedOverlapMasks{k} = [];
        end
    end
end
