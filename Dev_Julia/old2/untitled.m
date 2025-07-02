imgPaths = {
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_2020.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Brazilian Rainforest\12_1985.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2014.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Columbia Glacier\12_2000.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Dubai\12_1995.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Frauenkirche\2012_08.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Kuwait\2_2017.jpg"
    "C:\Users\julia\OneDrive\Dokumente\1_TUM\Computer Vision\Challenge\CV_Challenge\Datasets\Wiesn\3_2020.jpg"
};

for i = 1:length(imgPaths)
    imgPath = imgPaths{i};
    img = imread(imgPath);
    mask = true(size(img,1), size(img,2));  % full image mask

    fprintf('Processing: %s\n', imgPath);
    testRiverDetectionStages(img, mask);

end

function testRiverDetectionStages(rgbImg, mask)
    % Run detection and capture intermediate outputs
    [riverMask, keepAll, validWhite, mergedRiverMask, riverCandidates] = ...
        classifyRiversThinCurves_Debug(rgbImg, mask);

    % Plot all stages
    figure('Name','River Detection Stages','NumberTitle','off');

    subplot(2,3,1);
    imshow(rgbImg); title('1. Original RGB Image');

    subplot(2,3,2);
    imshow(riverCandidates); title('2. Initial River Candidates');

    subplot(2,3,3);
    imshow(validWhite); title('3. Valid Whiteish Shapes');

    subplot(2,3,4);
    imshow(mergedRiverMask); title('4. Merged Candidate Mask');

    subplot(2,3,5);
    imshow(keepAll); title('5. After Generous Connection');

    subplot(2,3,6);
    imshow(riverMask); title('6. Final Filtered River Mask');
end

function [riverMask, keepAll, validWhite, mergedRiverMask, riverCandidates] = classifyRiversThinCurves_Debug(rgbImg, mask)
    R = double(rgbImg(:,:,1)) / 255;
    G = double(rgbImg(:,:,2)) / 255;
    B = double(rgbImg(:,:,3)) / 255;

    greenish = (G > R + 0.03) & (G > B + 0.02) & (G > 0.35);
    whiteish = (R > 0.75) & (G > 0.75) & (B > 0.75);
    riverCandidates = (greenish | whiteish) & mask;

    riverCandidates = bwareaopen(riverCandidates, 20);
    whiteMask = whiteish & mask;
    whiteMask = bwareaopen(whiteMask, 20);
    CC_white = bwconncomp(whiteMask);
    stats_white = regionprops(CC_white, 'Area', 'Eccentricity', 'PixelIdxList');

    validWhite = false(size(R));
    for i = 1:CC_white.NumObjects
        if stats_white(i).Eccentricity > 0.85 && stats_white(i).Area < 300
            validWhite(stats_white(i).PixelIdxList) = true;
        end
    end

    mergedRiverMask = (greenish & mask) | validWhite;

    % --- Generous connection stage ---
    skel_all = bwskel(mergedRiverMask, 'MinBranchLength', 5);
    thickened = imdilate(skel_all, strel('disk', 3));
    grouped = imclose(thickened, strel('disk', 5));

    CC_all = bwconncomp(grouped);
    minLenAll = 0.0001 * sqrt(size(riverCandidates,1)^2 + size(riverCandidates,2)^2);

    keepAll = false(size(R));
    statsAll = regionprops(CC_all, 'Area', 'PixelIdxList');
    for k = 1:CC_all.NumObjects
        if statsAll(k).Area >= minLenAll
            keepAll(statsAll(k).PixelIdxList) = true;
        end
    end

    % --- Final strict filtering stage ---
    skel_main = bwskel(keepAll);
    CC_main = bwconncomp(skel_main);
    minLenMain = 0.1 * sqrt(size(R,1)^2 + size(R,2)^2);

    finalKeep = false(size(R));
    statsMain = regionprops(CC_main, 'Area', 'PixelIdxList');
    for k = 1:CC_main.NumObjects
        if statsMain(k).Area >= minLenMain
            finalKeep(statsMain(k).PixelIdxList) = true;
        end
    end

    riverMask = mergedRiverMask & imdilate(finalKeep, strel('disk', 2));
end
