classdef SatelliteImageRegistration < handle
    %SATELLITEIMAGEREGISTRATION  End-to-end workflow for registering
    % true-colour or multispectral satellite imagery stored in a folder.
    %
    %   reg = SatelliteImageRegistration;   % interactive folder picker
    %   reg.run();                          % do everything in one line

    %% Public (user-facing) properties
    properties (Access = public)
        imageFolder            char    % Path chosen by the user
        imageFiles             cell    % Sorted file names
        colorImages            cell    % Unregistered colour images (RGB)
        registeredColorImages  cell    % Output returned by Register_Color_Images
        validIndices           logical % Logical vector of successful registers
        refIdx                 double  % Index of reference frame
        tformList              cell    % Cell array of affine2d objects
        segmentedMaps          cell    % Segmentation output for each image
        segmentedOverlapMasks  cell    % Overlap masks after transformation
    end

    %% =====             Constructor & high-level API              ===== %%
    methods
        function obj = SatelliteImageRegistration(imageFolder)
            % Constructor
            % If imageFolder is omitted we fall back to an interactive
            % folder selection dialog—exactly what the old script did.
            if nargin < 1 || isempty(imageFolder)
                imageFolder = uigetdir(pwd, 'Select satellite image folder');
            end

            if imageFolder == 0
                error('SatelliteImageRegistration:NoFolder', ...
                     'No folder selected.');
            end
            obj.imageFolder = imageFolder;
        end

        function run(obj)
            %RUN  Convenience wrapper: load → register → display.
            %obj.loadImages();
            % obj.segmentImages(); 
            % obj.registerImages('city');   % default preset; change as needed
            obj.displayRegisteredImages();
            % obj.applyTransformsToSegmentations();
            %obj.displaySegmentedImages();           
            %obj.displayTransformedSegmentations();
            % obj.plotClassStatistics();
            obj.plotClassHeatmap();
        end

        function selectFolder(obj, folder)
            %SELECTFOLDER  Change the source image folder *after* construction
            if nargin < 2
                folder = uigetdir(pwd, 'Select satellite image folder');
            end
            if folder == 0,  return,  end
            obj.reset();          % clear cached data from previous run
            obj.imageFolder = folder;
        end
    end

    %% =====                    Pipeline steps                    ===== %%
    methods
        function loadImages(obj)
            %LOADIMAGES  Read all JPG, PNG and TIF files in imageFolder
            imgStruct = [dir(fullfile(obj.imageFolder,'*.jpg')); ...
                         dir(fullfile(obj.imageFolder,'*.png')); ...
                         dir(fullfile(obj.imageFolder,'*.tif'))];

            if numel(imgStruct) < 2
                error('SatelliteImageRegistration:NotEnoughImages', ...
                     'At least two images are required to perform registration.');
            end

            obj.imageFiles  = obj.sortNat({imgStruct.name});
            obj.colorImages = cell(1, numel(obj.imageFiles));

            for k = 1:numel(obj.imageFiles)
                obj.colorImages{k} = imread(fullfile(obj.imageFolder, ...
                                                     obj.imageFiles{k}));
            end
        end

        function segmentImages(obj)
            %SEGMENTIMAGES Segment loaded color images using external function
            if isempty(obj.colorImages)
                error('No images loaded. Call loadImages() first.');
            end
            obj.segmentedMaps = Segmentation(obj.colorImages);
        end

        function registerImages(obj, varargin)
            %REGISTERIMAGES  Wrapper around Register_Color_Images
            %
            %   registerImages(obj, 'city')  % keeps original default

            if isempty(obj.colorImages)
                obj.loadImages();
            end

            numImages = numel(obj.colorImages);
            [obj.registeredColorImages, ...
             obj.validIndices,          ...
             obj.refIdx,                ...
             obj.tformList] = Register_Color_Images( ...
                                   obj.colorImages, numImages, varargin{:});
        end

        function applyTransformsToSegmentations(obj)
        %APPLYTRANSFORMSTOSEGMENTATIONS Wrapper that calls external function
        % Applies transforms and stores results in obj.segmentedOverlapMasks
        
            if isempty(obj.segmentedMaps) || isempty(obj.tformList)
                warning('Segmentations or transformations missing. Skipping transformation.');
                return;
            end
        
            % Call the external function and store results in the class
            obj.segmentedOverlapMasks = Transform_Segmented_Images( ...
                obj.segmentedMaps, obj.tformList, obj.refIdx);
        end


        function applyTransformsToSegmentations2(obj)
            if isempty(obj.segmentedMaps) || isempty(obj.tformList)
                warning('You must run segmentation and registration first.');
                return;
            end
            refSize = size(obj.segmentedMaps{obj.refIdx}); % Höhe × Breite
            obj.segmentedOverlapMasks = Transform_Segmented_Images(obj.segmentedMaps, obj.tformList, refSize(1:2));
        end

        function plotClassStatistics(obj)
            %PLOTCLASSTRENDS Calls external function to show class trends
            if isempty(obj.segmentedOverlapMasks)
                warning('Segmented overlap masks are empty. Run applyTransformsToSegmentations first.');
                return;
            end
            Plot_Class_Percentages_Over_Time(obj.segmentedOverlapMasks, obj.imageFiles);
        end

        function plotClassHeatmap(obj)
            if isempty(obj.segmentedOverlapMasks)
                warning('Segmented overlap masks are empty. Run applyTransformsToSegmentations first.');
                return;
            end
            Plot_Class_Heatmap(obj.segmentedOverlapMasks, obj.imageFiles);
        end
        


        function displayRegisteredImages(obj)
            %DISPLAYREGISTEREDIMAGES  Show results in a tidy grid
            if isempty(obj.registeredColorImages)
                warning('Nothing to display – call registerImages() first.');
                return;
            end
            obj.displayGrid(obj.registeredColorImages, ...
                            obj.validIndices,          ...
                            obj.imageFiles,            ...
                            obj.refIdx);
        end

        function displaySegmentedImages(obj)
            %DISPLAYSEGMENTEDIMAGES Displays the original segmented maps with custom colormap
            if isempty(obj.segmentedMaps)
                warning('No segmented maps available. Run segmentImages() first.');
                return;
            end
        
            % Define custom colormap and labels
            cmap = [
                0.8 0.8 0.8; 0.2 0.55 0.5; 0.6 0.4 0.2; 1 0 0; 1 1 1;
                0 1 1
            ];
            classNames = {'Unclassified','Water/Forest','Land','Urban/Agriculture','Snow','River/Road'};
        
            numImages = numel(obj.segmentedMaps);
            cols = ceil(sqrt(numImages));
            rows = ceil(numImages / cols);
        
            figure('Name','Original Segmented Images', ...
                   'NumberTitle','off', ...
                   'Position',[100, 100, min(1800, cols*300), min(1200, rows*250)]);
        
            for i = 1:numImages
                subplot(rows, cols, i);
                segImg = obj.segmentedMaps{i};
        
                if isempty(segImg)
                    axis off;
                    text(0.5, 0.5, 'Segmentation Failed', ...
                        'HorizontalAlignment','center', ...
                        'VerticalAlignment','middle', ...
                        'FontSize',12,'Color','red');
                else
                    imshow(segImg, cmap);
                    title(sprintf('Segmented %d', i), 'FontSize', 9);
                end
        
                axis off;
            end
        
            sgtitle('Segmented Images (Original)', 'FontSize', 14, 'FontWeight', 'bold');
        
            % Optional: add color legend
            colormap(cmap);
            colorbar('Ticks', 0:numel(classNames)-1, ...
                     'TickLabels', classNames, ...
                     'TickLength', 0, 'Direction', 'reverse');
        end


        function displayTransformedSegmentations(obj)
            %DISPLAYTRANSFORMEDSEGMENTATIONS Displays warped segmented maps with custom colormap
            if isempty(obj.segmentedOverlapMasks)
                warning('No transformed segmentations available. Run applyTransformsToSegmentations() first.');
                return;
            end
        
            % Define custom colormap and labels
            cmap = [
                0.8 0.8 0.8; 0.2 0.55 0.5; 0.6 0.4 0.2; 1 0 0; 1 1 1;
                0 1 1
            ];
            classNames = {'Unclassified','Water/Forest','Land','Urban/Agriculture','Snow','River/Road'};
        
            numImages = numel(obj.segmentedOverlapMasks);
            cols = ceil(sqrt(numImages));
            rows = ceil(numImages / cols);
        
            figure('Name','Transformed Segmentations', ...
                   'NumberTitle','off', ...
                   'Position',[150, 150, min(1800, cols*300), min(1200, rows*250)]);
        
            for i = 1:numImages
                subplot(rows, cols, i);
                segImg = obj.segmentedOverlapMasks{i};
        
                if isempty(segImg)
                    axis off;
                    text(0.5, 0.5, 'Transform Failed', ...
                        'HorizontalAlignment','center', ...
                        'VerticalAlignment','middle', ...
                        'FontSize',12,'Color','red');
                else
                    imshow(segImg, cmap);
                    title(sprintf('Warped %d', i), 'FontSize', 9);
                end
        
                axis off;
            end
        
            sgtitle('Transformed Segmented Images', 'FontSize', 14, 'FontWeight', 'bold');
        
            % Optional: add color legend
            colormap(cmap);
            colorbar('Ticks', 0:numel(classNames)-1, ...
                     'TickLabels', classNames, ...
                     'TickLength', 0, 'Direction', 'reverse');
        end



    end

    %% =====                    House-keeping                     ===== %%
    methods (Access = public)
        function reset(obj)
            %RESET  Clear cached data, keeping only the folder path
            props = properties(obj);
            for p = 1:numel(props)
                if ~strcmp(props{p}, 'imageFolder')
                    obj.(props{p}) = [];
                end
            end
        end
    end

    %% =====             PRIVATE helpers (no API)                ===== %%
    methods (Access = private)
        function sorted = sortNat(~, filenames)
            %SORTNAT  Natural sort by 4-digit year token in the file name
            %
            %   - Looks for a YYYY pattern (e.g. 2022) and sorts numerically
            %   - Falls back to plain alphabetical sort if no pattern found
            expr      = '\d{4}';
            years     = regexp(filenames, expr, 'match', 'once');
            hasYear   = ~cellfun('isempty', years);

            if ~any(hasYear)          % plain old alphabetical sort
                sorted = sort(filenames);
                return
            end

            years   = cellfun(@str2double, years);
            [~, ix] = sort(years);
            sorted  = filenames(ix);
        end
    end

    methods (Static, Access = private)
        function displayGrid(registeredColorImages, validIndices, ...
                             imageFiles, refIdx)
            %DISPLAYGRID  (static)  Port of original displayRegisteredColorImages
            %
            % Making this *static* allows reuse outside the class, e.g.
            % SatelliteImageRegistration.displayGrid(…)
            %
            % -----------------------------------------------------------------
            validImages = registeredColorImages(validIndices);
            validFiles  = imageFiles(validIndices);

            numImages   = numel(validImages);
            if numImages == 0
                fprintf('No valid registered images to display.\n');
                return
            end

            cols = ceil(sqrt(numImages));
            rows = ceil(numImages / cols);

            figure('Name','Registered Colour Images', ...
                   'NumberTitle','off', ...
                   'Position',[50, 50, min(1800, cols*300), min(1200, rows*250)]);

            for i = 1:numImages
                subplot(rows, cols, i);

                if ~isempty(validImages{i})
                    imshow(validImages{i});

                    idx = validIndices(i);
                    if idx == refIdx                         % highlight ref
                        title(sprintf('%d: %s (REF)', idx, validFiles{i}), ...
                              'FontSize',10,'FontWeight','bold','Color','red');
                    else
                        title(sprintf('%d: %s', idx, validFiles{i}), ...
                              'FontSize',9);
                    end
                else                                        % failed case
                    axis off
                    text(0.5,0.5,'Registration Failed', ...
                         'HorizontalAlignment','center', ...
                         'VerticalAlignment','middle', ...
                         'FontSize',12,'Color','red');
                    title(sprintf('%d: %s (FAILED)', validIndices(i), ...
                                  validFiles{i}), 'FontSize',9,'Color','red');
                end

                axis off
            end

            % Overall caption
            try
                sgtitle(sprintf('Registered Colour Images (%d/%d successful)', ...
                        numImages, numel(registeredColorImages)), ...
                        'FontSize',14,'FontWeight','bold');
            catch       % older MATLAB release
                annotation('textbox',[0 0.95 1 0.05], ...
                           'String',sprintf('Registered Colour Images (%d/%d successful)', ...
                                   numImages, numel(registeredColorImages)), ...
                           'EdgeColor','none','HorizontalAlignment','center', ...
                           'FontSize',14,'FontWeight','bold');
            end
        end
    end
end
