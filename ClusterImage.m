
imgPath = "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg";
K = 8;  % Number of color clusters (try 5–8)

[clusterLabels, clusterRGBs] = clusterImageByColor(imgPath, K);
classifyClustersByColor(imgPath, clusterLabels, clusterRGBs);


function [clusterLabels, clusterRGBs] = clusterImageByColor(imgPath, K)
% CLUSTERIMAGEBYCOLOR Clusters image pixels by color using K-means.
% Returns cluster label image and cluster mean RGBs.

    % Read and normalize image
    rgbImg = im2double(imread(imgPath));
    [h, w, ~] = size(rgbImg);
    reshaped = reshape(rgbImg, [], 3);

    % K-means clustering in RGB space
    [labels, C] = kmeans(reshaped, K, 'MaxIter', 500, 'Replicates', 3);

    % Reshape labels into image form
    clusterLabels = reshape(labels, h, w);
    clusterRGBs = C;  % Cluster centroids (mean RGBs)

    % Debug visualization
    debugImg = zeros(h, w, 3);
    for k = 1:K
        mask = clusterLabels == k;
        for c = 1:3
            channel = debugImg(:,:,c);
            channel(mask) = C(k, c);
            debugImg(:,:,c) = channel;
        end
    end

    figure;
    imshow(debugImg)
    title("Color Cluster Visualization")

end

function classifyClustersByColor(imgPath, clusterLabels, clusterRGBs)
% CLASSIFYCLUSTERSBYCOLOR Assigns land cover class to each cluster by mean color.

    [h, w] = size(clusterLabels);
    labelImg = zeros(h, w);

    % Loop over clusters
    for k = 1:size(clusterRGBs, 1)
        rgb = clusterRGBs(k, :);
        R = rgb(1); G = rgb(2); B = rgb(3);

        % --- Heuristic color classification ---
        if (G > R + 0.05) && (G > B + 0.05) && (G > 0.35)
            class = 1;  % Forest
        elseif (B > 0.5 && G > 0.5 && R > 0.4) && (B >= G) && (G >= R) || ...
               (abs(R - G) < 0.08 && abs(G - B) < 0.08 && mean(rgb) > 0.5)
            class = 2;  % River (white/cyan)
        elseif mean(rgb) > 0.3 && G < R && G < B
            class = 3;  % City (non-green, bright)
        else
            class = 4;  % Unlabeled
        end

        % Assign class to all pixels in this cluster
        labelImg(clusterLabels == k) = class;
    end

    % --- Display ---
    cmap = [0 1 0;       % Forest - green
            0 0 1;       % River - blue
            0.5 0.5 0.5; % City - gray
            0.6 0.4 0.2];% Unlabeled - brown

    overlay = labeloverlay(im2double(imread(imgPath)), labelImg, ...
        'Colormap', cmap, 'Transparency', 0.4);

    figure;
    imshow(overlay);
    title("Land Cover Classification via Clustering")

    % Legend
    annotation("textbox", [0.75 0.60 0.2 0.05], "String", "Forest", ...
        "BackgroundColor", [0 1 0], "Color", [0 0 0]);
    annotation("textbox", [0.75 0.53 0.2 0.05], "String", "River", ...
        "BackgroundColor", [0 0 1], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.46 0.2 0.05], "String", "City", ...
        "BackgroundColor", [0.5 0.5 0.5], "Color", [1 1 1]);
    annotation("textbox", [0.75 0.39 0.2 0.05], "String", "Unlabeled", ...
        "BackgroundColor", [0.6 0.4 0.2], "Color", [1 1 1]);
end
