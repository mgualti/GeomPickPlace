function PlotSegmentationPR()

    %% Parameters
    
    resultFiles = {'../segmentation-results-train', '../segmentation-results-test1', '../segmentation-results-test2'};
    resultNames = {'Train', 'Test-1', 'Test-2'};

    %% Load
    figure; hold('on');
    for resultIdx = 1:length(resultFiles)
        
        data = load(resultFiles{resultIdx});
        
        [recall, sortIdx] = sort(data.recall, 'ascend');
        precision = data.precision(sortIdx);
        
        plot(recall, precision, '-x', 'linewidth', 3, 'markersize', 15);
    end
    
    grid('on');
    xlabel('Recall', 'fontsize', 14, 'fontweight', 'bold');
    ylabel('Precision', 'fontsize', 14, 'fontweight', 'bold')
    legend(resultNames, 'fontsize', 14, 'fontweight', 'bold');