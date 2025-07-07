function result = transform_and_intersect(seg_img, R, t)
% Applies rotation and translation to a labeled segmentation image,
% and returns only the intersecting labeled area.
%
% seg_img - input labeled image (values from 1 to N)
% R       - 2x2 rotation matrix
% t       - 2x1 translation vector

    % Ensure image is 2D
    if ndims(seg_img) > 2
        seg_img = rgb2gray(seg_img);
    end
    seg_img = double(seg_img);

    % Create affine2d object from R and t
    T = affine2d([R, t; 0 0 1]);

    % Transform the image
    output_ref = imref2d(size(seg_img));
    transformed_img = imwarp(seg_img, T, ...
        'OutputView', output_ref, ...
        'InterpolationMethod', 'nearest', ...
        'FillValues', 0);

    % Keep only the intersection (non-zero in both original and transformed)
    intersection_mask = (seg_img > 0) & (transformed_img > 0);
    result = zeros(size(seg_img));
    result(intersection_mask) = transformed_img(intersection_mask);
end
