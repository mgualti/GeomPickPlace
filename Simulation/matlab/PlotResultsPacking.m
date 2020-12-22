function PlotResultsPacking()

    %% Parameters
    
    %resultsFileName = '../results-packing-test1-noCost.mat';
    %resultsFileName = '../results-packing-test1-step.mat';
    %resultsFileName = '../results-packing-test1-antipodal.mat';
    %resultsFileName = '../results-packing-test1-contact.mat';
    %resultsFileName = '../results-packing-test1-mc.mat';
    resultsFileName = '../results-packing-isec10-test1-sp.mat';
    
    %resultsFileName = '../results-packing-isec10-test1-gtSegComp.mat';
    %resultsFileName = '../results-packing-isec10-train-gtSeg.mat';
    %resultsFileName = '../results-packing-isec10-test1-gtSeg.mat';
    %resultsFileName = '../results-packing-isec10-train-percep.mat';
    %resultsFileName = '../results-packing-isec10-test1-percep.mat';
    %resultsFileName = '../results-packing-isec10-test1-noComp.mat';
    %resultsFileName = '../results-packing-isec11-test2-gtSegComp.mat';
    %resultsFileName = '../results-packing-isec11-test2-gtSeg.mat';
    %resultsFileName = '../results-packing-isec11-test2-percep.mat';
    %resultsFileName = '../results-packing-isec11-test2-noComp.mat';
    
    if contains(resultsFileName, 'isec11'), timeFactor = 0.90; else, timeFactor = 1.0; end
    
    fmt = '%0.3f';
    %pm = [' ' char(177) ' '];
    pm = ' $\pm$ ';

    %% Load

    close('all');

    if ~exist(resultsFileName, 'file')
        disp([resultsFileName ' not found.']);
        return
    end
    load(resultsFileName);
    disp(' ');
    
    nEpisodes = length(nPlaced);
    disp(['Completed ' num2str(nEpisodes) ' episodes.']);
    
    %% Task Completion Related
    
    placeSuccess = [ones(1, sum(nPlaced)), zeros(1, nObjects * nEpisodes - sum(nPlaced))];
    avgPlaceSuccess = mean(placeSuccess);
    errPlaceSuccess = std(placeSuccess) / sqrt(length(placeSuccess));
    disp(['Place success rate: ' num2str(avgPlaceSuccess, fmt) pm num2str(errPlaceSuccess, fmt)])
    
    nRegraspPlans = length(planLength);
    execSuccess = [ones(1, sum(nPlaced)), zeros(1, nRegraspPlans - sum(nPlaced))];
    avgExecSuccess = mean(execSuccess);
    errExecSuccess = std(execSuccess) / sqrt(length(execSuccess));
    disp(['Place execution success rate: ' num2str(avgExecSuccess, fmt) pm num2str(errExecSuccess, fmt)]);
    
    nRegraspPlanAttempts = length(regraspPlanningTime);
    regraspPlanFound = [ones(1, nRegraspPlans), zeros(1, nRegraspPlanAttempts - nRegraspPlans)];
    avgRegraspPlanFound = mean(regraspPlanFound);
    errRegraspPlanFound = std(regraspPlanFound) / sqrt(length(regraspPlanFound));
    disp(['Regrasp plan found rate: ' num2str(avgRegraspPlanFound, fmt) pm num2str(errRegraspPlanFound, fmt)]);
    
    graspSuccessRate = num2str(mean(graspSuccess), fmt);
    errGraspSuccessRate = num2str(std(graspSuccess) / sqrt(length(graspSuccess)), fmt);
    disp(['Grasp success rate: ' graspSuccessRate pm errGraspSuccessRate]);
    
    graspAntipodalRate = num2str(mean(graspAntipodal), fmt);
    errGraspAntipodalRate = num2str(std(graspAntipodal) / sqrt(length(graspAntipodal)), fmt);
    disp(['Grasp antipodal rate: ' graspAntipodalRate pm errGraspAntipodalRate]);
    
    placeStableRate = num2str(mean(tempPlaceStable), fmt);
    errPlaceStableRate = num2str(std(tempPlaceStable) / sqrt(length(tempPlaceStable)), fmt);
    disp(['Temporary placement stability rate: ' placeStableRate pm errPlaceStableRate]);
    
    avgPlanLength = num2str(mean(planLength), fmt);
    errPlanLength = num2str(std(double(planLength)) / sqrt(length(planLength)), fmt);
    nPlanLengthsGreaterThan2 = num2str(sum(planLength > 2));
    nPlanLengths = num2str(length(planLength));
    disp(['Average plan length: ' avgPlanLength pm errPlanLength]);
    disp(['Number of plan lengths greater than 2: ' nPlanLengthsGreaterThan2 ' / ' nPlanLengths]);

    %% Task Performance Related
    
    % number of episodes with the given number of items packed
    
    nEpisodesHist = zeros(nObjects + 1, 1);
    for idx = 1:length(nEpisodesHist)
        nEpisodesHist(idx) = sum(nPlaced == (idx - 1));
    end
    
    figure;
    x = 0:nObjects;
    bar(x, nEpisodesHist);
    xlabel('Number of Items Packed');
    ylabel('Number of Episodes');
    
    % height of pile
    
    avgPackingHeight = zeros(nObjects, 1);
    stdPackingHeight = zeros(nObjects, 1);
    for idx = 1:length(avgPackingHeight)
        jdx = nPlaced == idx;
        avgPackingHeight(idx) = mean(packingHeight(jdx));
        stdPackingHeight(idx) = std(packingHeight(jdx));
    end
    
    figure; hold('on');
    x = 1:nObjects;
    bar(x, avgPackingHeight * 100, 'r');
    errPackingHeight = (stdPackingHeight ./ sqrt(nEpisodesHist(2:end)));
    errorBar = errorbar(x, avgPackingHeight * 100, errPackingHeight * 100, errPackingHeight * 100);
    errorBar.Color = [0 0 0];
    errorBar.LineStyle = 'none';  
    ylim([0, 16]);
    ylabel('Average Packing Height (cm)');
    xlabel('Number of Items Packed');
    
    avg6PackingHeight = num2str(avgPackingHeight(6) * 100, fmt);
    err6PackingHeight = num2str(errPackingHeight(6) * 100, fmt);
    disp(['Average packing of 6 height: ' avg6PackingHeight pm err6PackingHeight ' (cm)']);
    
    avg5PackingHeight = num2str(avgPackingHeight(5) * 100, fmt);
    err5PackingHeight = num2str(errPackingHeight(5) * 100, fmt);
    disp(['Average packing of 5 height: ' avg5PackingHeight pm err5PackingHeight ' (cm)']);
    
    avg4PackingHeight = num2str(avgPackingHeight(4) * 100, fmt);
    err4PackingHeight = num2str(errPackingHeight(4) * 100, fmt);
    disp(['Average packing of 4 height: ' avg4PackingHeight pm err4PackingHeight ' (cm)']);
    
    %% Planning Times
    
    taskPlanningTime = timeFactor * taskPlanningTime;
    avgTaskPlanningTime = num2str(mean(taskPlanningTime), fmt);
    errTaskPlanningTime = num2str(std(taskPlanningTime) / sqrt(length(taskPlanningTime)), fmt);
    disp(['Average task planning time: ' avgTaskPlanningTime pm errTaskPlanningTime ' (s)']);
    
    regraspPlanningTime = timeFactor * regraspPlanningTime;
    avgRegraspPlanningTime = num2str(mean(regraspPlanningTime), fmt);
    errRegraspPlanningTime = num2str(std(regraspPlanningTime) / sqrt(length(regraspPlanningTime)), fmt);
    disp(['Average regrasp planning time: ' avgRegraspPlanningTime pm errRegraspPlanningTime ' (s)']);
    
    totalTime = timeFactor * totalTime;
    totalTime = num2str(totalTime / 3600, fmt);
    disp(['Total time: ' totalTime ' (h)']);