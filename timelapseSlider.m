function timelapseSlider(imgCell)
% timelapseViewer zeigt ein interaktives Timelapse
%   timelapseViewer(imgCell) erwartet ein Cell-Array {I1,…,IN} mit
%   bereits registrierten double-Bildern (Grau- oder RGB).

    N = numel(imgCell);
    if N < 2
        error('Mindestens zwei Bilder erforderlich.');
    end

    % Stelle sicher, dass alle Bilder double im [0,1]-Bereich sind
    for k = 1:N
        imgCell{k} = im2double(imgCell{k});
    end

    % Figure anlegen
    hFig = figure('Name','Timelapse Viewer','Color','w', ...
                  'NumberTitle','off','MenuBar','none');
    ax = axes('Parent',hFig,'Position',[0.05 0.15 0.9 0.8]);
    imHandle = imshow(imgCell{1}, 'Parent', ax);
    title(ax, sprintf('Frame 1 / %d', N), 'FontSize',12);

    % Slider
    sld = uicontrol('Parent',hFig, 'Style','slider', ...
        'Units','normalized','Position',[0.05 0.05 0.9 0.05], ...
        'Min',1,'Max',N,'Value',1, ...
        'SliderStep',[1/(N-1) , 5/(N-1)]);

    % Callback
    addlistener(sld, 'Value', 'PostSet', @updateFrame);

    function updateFrame(~,~)
        v = get(sld,'Value');
        i = floor(v);
        alpha = v - i;
        if i >= N
            img = imgCell{N};
            frameText = sprintf('Frame %d / %d', N, N);
        else
            A = imgCell{i};
            B = imgCell{i+1};
            img = (1-alpha)*A + alpha*B;
            frameText = sprintf('Frame %.2f / %d', v, N);
        end
        set(imHandle, 'CData', img);
        title(ax, frameText, 'FontSize',12);
        drawnow;
    end
end