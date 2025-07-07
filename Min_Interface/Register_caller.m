function Register_caller()
    %% 1. Select image folder
        imageFolder = uigetdir(pwd, 'Select satellite image folder');
        if imageFolder == 0
            disp('No folder selected. Aborting.');
            return;
        end
    
        %% 2. Load and sort images
        imageFiles = [dir(fullfile(imageFolder, '*.jpg')); 
                      dir(fullfile(imageFolder, '*.png')); 
                      dir(fullfile(imageFolder, '*.tif'))];
        
        if length(imageFiles) < 2
            error('At least two images required.');
        end
        
        imageFiles = sort_nat({imageFiles.name});
        numImages = length(imageFiles);

        for i = 1:numImages
            % Load color image
            colorImages{i} = imread(fullfile(imageFolder, imageFiles{i}));
        end

        [registeredColorImages, validIndices, imageFiles, refIdx, tformList] = Register_Color_Images(imageFiles, colorImages, numImages);

        % for i = 1:numel(tformList)
        %     fprintf('Entry %d:\n', i);
        %     if isempty(tformList{i})
        %         disp('  <empty>');
        %     else
        %         disp(tformList{i}); % Will show object contents or data
        %     end
        % end


        displayRegisteredColorImages(registeredColorImages, validIndices, imageFiles, refIdx)

end

%% --- Natural sorting function ---
function sorted = sort_nat(filenames)
    expr = '\d{4}';
    years = regexp(filenames, expr, 'match', 'once');
    validYears = ~cellfun(@isempty, years);
    if sum(validYears) == 0
        sorted = sort(filenames);
        return;
    end
    years = cellfun(@str2double, years);
    [~, idx] = sort(years);
    sorted = filenames(idx);
end

%% --- Display all registered color images in a grid ---
function displayRegisteredColorImages(registeredColorImages, validIndices, imageFiles, refIdx)
    % Remove empty cells and get valid images
    validImages = registeredColorImages(validIndices);
    validFiles = imageFiles(validIndices);
    
    numImages = length(validImages);
    if numImages == 0
        fprintf('No valid registered images to display.\n');
        return;
    end
    
    % Calculate grid dimensions
    cols = ceil(sqrt(numImages));
    rows = ceil(numImages / cols);
    
    % Create figure
    figure('Name', 'All Registered Color Images', 'NumberTitle', 'off', ...
           'Position', [50, 50, min(1800, cols*300), min(1200, rows*250)]);
    
    for i = 1:numImages
        subplot(rows, cols, i);
        
        if ~isempty(validImages{i})
            imshow(validImages{i});
            
            % Add title with special marking for reference
            if validIndices(i) == refIdx
                title(sprintf('%d: %s (REF)', validIndices(i), validFiles{i}), ...
                      'FontSize', 10, 'FontWeight', 'bold', 'Color', 'red');
            else
                title(sprintf('%d: %s', validIndices(i), validFiles{i}), ...
                      'FontSize', 9);
            end
        else
            % Show placeholder for failed registration
            axis off;
            text(0.5, 0.5, 'Registration Failed', 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'middle', 'FontSize', 12, 'Color', 'red');
            title(sprintf('%d: %s (FAILED)', validIndices(i), validFiles{i}), ...
                  'FontSize', 9, 'Color', 'red');
        end
        
        axis off;
    end
    
    % Add overall title
    try
        sgtitle(sprintf('Registered Color Images (%d/%d successful)', ...
                numImages, length(registeredColorImages)), ...
                'FontSize', 14, 'FontWeight', 'bold');
    catch
        % Fallback for older MATLAB versions
        annotation('textbox', [0 0.95 1 0.05], ...
                   'String', sprintf('Registered Color Images (%d/%d successful)', ...
                                   numImages, length(registeredColorImages)), ...
                   'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
                   'FontSize', 14, 'FontWeight', 'bold');
    end
    
    fprintf('\nDisplayed %d registered color images in grid format.\n', numImages);
end