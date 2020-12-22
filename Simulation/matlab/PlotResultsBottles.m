function PlotResultsBottles()

    %% Parameters
    
    %resultsFileName = '../results-bottles-isec10-test-noCost.mat';
    %resultsFileName = '../results-bottles-isec10-test-step.mat';
    %resultsFileName = '../results-bottles-isec10-test-antipodal.mat';
    %resultsFileName = '../results-bottles-isec10-test-contact.mat';
    %resultsFileName = '../results-bottles-isec10-test-mc.mat';
    resultsFileName = '../results-bottles-isec10-test-sp.mat';

    %resultsFileName = '../results-bottles-isec10-test-gtSegComp.mat';
    %resultsFileName = '../results-bottles-isec10-train-gtSeg.mat';
    %resultsFileName = '../results-bottles-isec10-test-gtSeg.mat';
    %resultsFileName = '../results-bottles-isec10-train-percep.mat';
    %resultsFileName = '../results-bottles-isec10-test-percep.mat';
    %resultsFileName = '../results-bottles-isec10-test-noComp.mat';
    
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
    
    taskSuccess = double(nPlaced == nObjects);
    avgTaskSuccess = num2str(mean(taskSuccess), fmt);
    errTaskSuccess = num2str(std(taskSuccess) / sqrt(length(taskSuccess)), fmt);
    disp(['Task success rate: ' avgTaskSuccess pm errTaskSuccess]);
    
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
    
    avgPlaced = num2str(mean(nPlaced), fmt);
    errPlaced = num2str(std(double(nPlaced)) / sqrt(length(nPlaced)), fmt);
    disp(['Average number of objects placed: ' avgPlaced pm errPlaced]);
    
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
    nPlanLengthsGreaterThan2 = num2str(sum(planLength > 2));
    nPlanLengths = num2str(length(planLength));
    disp(['Average plan length: ' avgPlanLength pm errPlanLength]);
    disp(['Number of plan lengths greater than 2: ' nPlanLengthsGreaterThan2 ' / ' nPlanLengths]);
    
    %% Task Performance Related
    
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