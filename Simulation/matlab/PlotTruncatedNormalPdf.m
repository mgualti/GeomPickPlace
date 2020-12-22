function PlotTruncatedNormalPdf()
    
    %% parameters
    
    thetaMax = 12 * pi / 180;
    theta0 = 0 * pi / 180;
    theta1 = 6 * pi / 180;
    theta2 = 12 * pi / 180;
    sigma = 0.07;
    
    %% create distributions
    
    tnd0 = makedist('normal', theta0, sigma);
    tnd0 = truncate(tnd0, 0, pi);
    
    tnd1 = makedist('normal', theta1, sigma);
    tnd1 = truncate(tnd1, 0, pi);
    
    tnd2 = makedist('normal', theta2, sigma);
    tnd2 = truncate(tnd2, 0, pi);
    
    %% plot
    
    x = 0:0.001:pi;
    y0 = pdf(tnd0, x);
    y1 = pdf(tnd1, x);
    y2 = pdf(tnd2, x);
    y3 = linspace(0, max(pdf(tnd0, x)), length(x));
    
    close('all'); figure; hold('on');
    plot(x * 180 / pi, y0, 'linewidth', 2);
    plot(x * 180 / pi, y1, 'linewidth', 2);
    plot(x * 180 / pi, y2, 'linewidth', 2);
    plot(thetaMax * 180 / pi * ones(size(x)), y3);
    
    xlim([0, 25]); grid('on'); xlabel('\theta'); ylabel('probability density');
    legend('\mu = 0^\circ', '\mu = 6^\circ', '\mu = 12^\circ', '\theta_{max} = 12^\circ');