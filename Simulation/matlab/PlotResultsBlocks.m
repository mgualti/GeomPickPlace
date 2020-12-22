function PlotResultsBlocks()

    %% Parameters
    
    %resultsFileName = '../results-blocks-isec10-test-noCost.mat';
    %resultsFileName = '../results-blocks-isec10-test-step.mat';
    %resultsFileName = '../results-blocks-isec10-test-antipodal.mat';
    %resultsFileName = '../results-blocks-isec10-test-contact.mat';
    %resultsFileName = '../results-blocks-isec10-test-mc.mat';
    resultsFileName = '../results-blocks-isec10-test-sp.mat';

    %resultsFileName = '../results-blocks-isec10-test-gtSegComp.mat';
    %resultsFileName = '../results-blocks-isec10-train-gtSeg.mat';
    %resultsFileName = '../results-blocks-isec10-test-gtSeg.mat';
    %resultsFileName = '../results-blocks-isec10-train-percep.mat';
    %resultsFileName = '../results-blocks-isec10-test-percep.mat';
    %resultsFileName = '../results-blocks-isec10-test-noComp.mat';
    
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
    
    nEpisodesCompleted = length(nPlaced);
    disp(['Completed ' num2str(nEpisodesCompleted) ' episodes.']);

    %% Task Completion Related
    
    avgPlaced = num2str(mean(nPlaced), fmt);
    errPlaced = num2str(std(double(nPlaced)) / sqrt(length(nPlaced)), fmt);
    disp(['Average placed: ' avgPlaced pm errPlaced]);
    
    nEpisodes = length(nPlaced);
    placeSuccess = [ones(1, sum(nPlaced)), zeros(1, nObjects * nEpisodes - sum(nPlaced))];
    avgPlaceSuccess = num2str(mean(placeSuccess), fmt);
    errPlaceSuccess = num2str(std(placeSuccess) / sqrt(length(placeSuccess)), fmt);
    disp(['Place success rate: ' avgPlaceSuccess pm errPlaceSuccess])
    
    nRegraspPlans = length(planLength);
    execSuccess = [ones(1, sum(nPlaced)), zeros(1, nRegraspPlans - sum(nPlaced))];
    avgExecSuccess = num2str(mean(execSuccess), fmt);
    errExecSuccess = num2str(std(execSuccess) / sqrt(length(execSuccess)), fmt);
    disp(['Place execution success rate: ' avgExecSuccess pm errExecSuccess]);
    
    planFound = [ones(1, nRegraspPlans), zeros(1, nObjects * nEpisodes - nRegraspPlans)];
    avgPlanFound = num2str(mean(planFound), fmt);
    errPlanFound = num2str(std(planFound) / sqrt(length(planFound)), fmt);
    disp(['Plan found rate: ' avgPlanFound pm errPlanFound]);
    
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
    disp(['Average plan length: ' avgPlanLength pm errPlanLength]);
    
    %% Task Performance Related
    
    avgOrderCorrect = num2str(mean(orderCorrect), fmt);
    errOrderCorrect = num2str(std(double(orderCorrect)) / sqrt(length(orderCorrect)), fmt);
    disp(['Average order correct: ' avgOrderCorrect pm errOrderCorrect]);
    
    avgLongestEndUp = num2str(mean(longestEndUp), fmt);
    errLongestEndUp = num2str(std(double(longestEndUp)) / sqrt(length(longestEndUp)), fmt);
    disp(['Average longest end up: ' avgLongestEndUp pm errLongestEndUp]);
    
    %% Times
   
    avgTaskPlanningTime = num2str(mean(timeFactor * taskPlanningTime), fmt);
    errTaskPlanningTime = num2str(std(timeFactor * taskPlanningTime) / sqrt(length(taskPlanningTime)), fmt);
    disp(['Average task planning time: ' avgTaskPlanningTime pm errTaskPlanningTime ' (s)']);
    
    avgRegraspPlanningTime = num2str(mean(timeFactor * regraspPlanningTime), fmt);
    errRegraspPlanningTime = num2str(std(timeFactor * regraspPlanningTime) / sqrt(length(regraspPlanningTime)), fmt);
    disp(['Average regrasp planning time: ' avgRegraspPlanningTime pm errRegraspPlanningTime ' (s)']);
    
    totalPlanningTime = num2str(timeFactor * (sum(taskPlanningTime) + sum(regraspPlanningTime)) / 3600, fmt);
    disp(['Total planning time: ' totalPlanningTime ' (h)']);
    
    totalTime = num2str(timeFactor * totalTime / 3600, fmt);
    disp(['Total time: ' totalTime ' (h)']);