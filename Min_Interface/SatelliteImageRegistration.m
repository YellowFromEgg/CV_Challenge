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
            obj.loadImages();
            obj.registerImages('city');   % default preset; change as needed
            obj.displayRegisteredImages();
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
