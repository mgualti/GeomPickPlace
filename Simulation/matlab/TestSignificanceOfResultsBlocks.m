function TestSignificanceOfResultsBlocks()

    %% Parameters
    
    baselineResultFile = '../results-blocks-isec10-test-gq.mat';
    treatmentResultFile = '../results-blocks-isec10-test-sp.mat';
    
    %% Load Data
    
    B = load(baselineResultFile);
    T = load(treatmentResultFile);
    
    clc; disp('');
    
    %% Student's t-test
    
    % Null hypothesis: The treatment resulted in no improvement over the baseline.
    % Alternative hypothesis: The treatment resulted in a statistically significant improvement.
    
    placeSuccessB = [ones(1, sum(B.nPlaced)), zeros(1, B.nObjects * length(B.nPlaced) - sum(B.nPlaced))];
    placeSuccessT = [ones(1, sum(T.nPlaced)), zeros(1, T.nObjects * length(T.nPlaced) - sum(T.nPlaced))];
    [h, p] = ttest2(placeSuccessB, placeSuccessT, 'tail', 'left');
    if h, isSig = 'is'; else isSig = 'is not'; end
    disp(['Place success rate ' isSig ' significantly greater than baseline (p = ' num2str(p) ').']);
    
    execSuccessB = [ones(1, sum(B.nPlaced)), zeros(1, length(B.planLength) - sum(B.nPlaced))];
    execSuccessT = [ones(1, sum(T.nPlaced)), zeros(1, length(T.planLength) - sum(T.nPlaced))];
    [h, p] = ttest2(execSuccessB, execSuccessT, 'tail', 'left');
    if h, isSig = 'is'; else isSig = 'is not'; end
    disp(['Place execution success rate ' isSig ' significantly greater than baseline (p = ' num2str(p) ').']);
    
    [h, p] = ttest2(B.graspSuccess, T.graspSuccess, 'tail', 'left');
    if h, isSig = 'is'; else isSig = 'is not'; end
    disp(['Grasp success rate ' isSig ' significantly greater than baseline (p = ' num2str(p) ').']);
    
    [h, p] = ttest2(B.graspAntipodal, T.graspAntipodal, 'tail', 'left');
    if h, isSig = 'is'; else isSig = 'is not'; end
    disp(['Grasp antipodal rate ' isSig ' significantly greater than baseline (p = ' num2str(p) ').']);
    
    [h, p] = ttest2(B.tempPlaceStable, T.tempPlaceStable, 'tail', 'left');
    if h, isSig = 'is'; else isSig = 'is not'; end
    disp(['Temporary placement stable rate ' isSig ' significantly greater than baseline (p = ' num2str(p) ').']);