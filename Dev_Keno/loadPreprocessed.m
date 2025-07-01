function [grayImages, colorImages, validMasks, imageNames, referenceIdx] = loadPreprocessed()
    %% Load registration results from saved .mat files
    
    % Select results folder
    resultsBaseDir = fullfile(pwd, 'Registration_Results');
    if ~exist(resultsBaseDir, 'dir')
        error('Registration_Results directory not found. Run briskV3() first.');
    end
    
    % Let user select which result folder to load
    selectedFolder = uigetdir(resultsBaseDir, 'Select registration results folder');
    if selectedFolder == 0
        error('No folder selected. Aborting.');
    end
    
    fprintf('Loading results from: %s\n', selectedFolder);
    
    % Get all .mat files
    matFiles = dir(fullfile(selectedFolder, '*.mat'));
    
    % Initialize outputs
    grayImages = {};
    colorImages = {};
    validMasks = {};
    imageNames = {};
    referenceIdx = [];
    
    % Parse filenames to group by image
    imageGroups = containers.Map();
    
    for i = 1:length(matFiles)
        filename = matFiles(i).name;
        [~, baseName, ~] = fileparts(filename);
        
        % Parse filename: imagename_TYPE_datatype.mat
        parts = strsplit(baseName, '_');
        if length(parts) >= 3
            imageName = strjoin(parts(1:end-2), '_');
            imageType = parts{end-1}; % REGISTERED or REFERENCE
            dataType = parts{end}; % gray, color, mask
            
            if ~isKey(imageGroups, imageName)
                imageGroups(imageName) = struct('name', imageName, 'type', imageType, ...
                    'gray', '', 'color', '', 'mask', '');
            end
            
            group = imageGroups(imageName);
            group.type = imageType;
            group.(dataType) = fullfile(selectedFolder, filename);
            imageGroups(imageName) = group;
        end
    end
    
    % Load data for each image group
    keys = imageGroups.keys;
    for i = 1:length(keys)
        group = imageGroups(keys{i});
        
        try
            % Load grayscale registered image
            if ~isempty(group.gray) && exist(group.gray, 'file')
                data = load(group.gray);
                grayImages{end+1} = data.registered_gray;
            else
                grayImages{end+1} = [];
            end
            
            % Load color registered image
            if ~isempty(group.color) && exist(group.color, 'file')
                data = load(group.color);
                colorImages{end+1} = data.registered_color;
            else
                colorImages{end+1} = [];
            end
            
            % Load valid mask
            if ~isempty(group.mask) && exist(group.mask, 'file')
                data = load(group.mask);
                validMasks{end+1} = data.valid_mask;
            else
                validMasks{end+1} = [];
            end
            
            imageNames{end+1} = group.name;
            
            % Mark reference image
            if strcmp(group.type, 'REFERENCE')
                referenceIdx = length(imageNames);
            end
            
        catch ME
            warning('Failed to load data for %s: %s', group.name, ME.message);
        end
    end
    
    fprintf('Successfully loaded:\n');
    fprintf('  - %d grayscale images\n', sum(~cellfun(@isempty, grayImages)));
    fprintf('  - %d color images\n', sum(~cellfun(@isempty, colorImages)));
    fprintf('  - %d valid masks\n', sum(~cellfun(@isempty, validMasks)));
    fprintf('  - Reference image index: %d (%s)\n', referenceIdx, imageNames{referenceIdx});
end