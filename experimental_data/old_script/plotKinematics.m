function plotKinematics(files, trialIdx, RGB, styles, plot_sz, titlePrefix, f, ti)

    %% --- Load Data ---
    nTrials = numel(trialIdx);
    v_x1 = cell(1,nTrials);
    v_x2 = cell(1,nTrials);
    v_x = cell(1,nTrials);    
    v_y1 = cell(1,nTrials);
    v_y2 = cell(1,nTrials);
    v_y = cell(1,nTrials);
    v = cell(1,nTrials);

    y = cell(1,nTrials);
    body_ang = cell(1,nTrials);

    for i = 1:nTrials
        dat = readmatrix(files(trialIdx(i)));
        if i == 1
            t = dat(1:end,1); % time [s]
        end

        v_x1{i}      = -dat(1:end,3) * 1000;
        v_x2{i}      = -dat(1:end,6) * 1000;
        v_x{i}       = 0.5 * (v_x1{i} + v_x2{i});

        v_y1{i}      = -dat(1:end,4) * 1000;
        v_y2{i}      = -dat(1:end,7) * 1000;
        v_y{i}       = 0.5 * (v_y1{i} + v_y2{i});

        v{i} = sqrt(v_x{i}.^2 + v_y{i}.^2);

        y_raw1       = dat(1:end,2);
        y_raw2       = dat(1:end,5);
        y{i}         = 0.5.*(y_raw1 + y_raw2) * 1000;

        body_ang{i} = dat(1:end,8);
    end

    % Compute means across trials
    v_x_mean      = mean(cat(2, v_x{:}), 2);
    v_x_mat = cat(2, v_x{:});          % [time × trials]
    v_x_std = std(v_x_mat, 0, 2);      % std across trials

    v_y_mean      = mean(cat(2, v_y{:}), 2);
    v_y_mat = cat(2, v_y{:});          % [time × trials]
    v_y_std = std(v_y_mat, 0, 2);      % std across trials

    v_mean      = mean(cat(2, v{:}), 2);
    v_mat = cat(2, v{:});          % [time × trials]
    v_std = std(v_mat, 0, 2);      % std across trials

    y_mean        = mean(cat(2, y{:}), 2);
    y_mat = cat(2, y{:});          % [time × trials]
    y_min_global = min(y_mat, [], 'all');
    y_max_global = max(y_mat, [], 'all');

    body_ang_mean = mean(cat(2, body_ang{:}), 2);
    theta_mat = cat(2, body_ang{:});   % [time × trials]
    theta_max_global = max(theta_mat, [], 'all');
    theta_min_global = min(theta_mat, [], 'all');
    

    %% --- Plot Time Series ---
    series = { {v, v_mean, 'v [mm/s]', 'Forward Speed vs. Time'}, ...
               {y, y_mean, 'height (mm)', 'Body Height vs. Time'}, ...
               {body_ang, body_ang_mean, '\theta [rad]', 'Body Angle vs Time'} };


figure('Position', plot_sz.Position)

for s = 1:3
    yMean  = series{s}{2};
    ylab   = series{s}{3};
    ttl    = series{s}{4};
    trials = series{s}{1};

    subplot(3,1,s); hold on;

        % for k = 1:nTrials
        %     plot(t, trials{k}, '-', 'Color', RGB(k,:), 'LineWidth', 1.5);
        %     legend('1','2','3','4')
        % end

    if s == 1
        ylim([-50, 600])
    
        % Plot mean time-series (optional: keep this as your main curve)
        plot(t, v_mean, styles(99).LineSpec, ...
             'Color', styles(99).Color, ...
             'LineWidth', 1.5);
        hold on;
    
        % Constant band from trial-wise steady means
        xl = xlim;
        v_lo = v_steady_mean - v_steady_std;
        v_hi = v_steady_mean + v_steady_std;
    
        hPatch = patch([xl(1) xl(2) xl(2) xl(1)], ...
                       [v_lo  v_lo  v_hi  v_hi], ...
                       styles(2).Color, ...
                       'FaceAlpha', 0.35, ...
                       'EdgeColor', 'none');
        uistack(hPatch, 'bottom');
    

    elseif s == 2
        % Mean curve
        plot(t, y_mean, styles(99).LineSpec, ...
             'Color', styles(99).Color, ...
             'LineWidth', styles(99).LineWidth);
        hold on;
    
        xl = xlim;
    
        y_lo = y_type_mean - y_type_std_global;
        y_hi = y_type_mean + y_type_std_global;
    
        % Shade mean ± global std (constant band)
        hPatch = patch([xl(1) xl(2) xl(2) xl(1)], ...
                       [y_lo  y_lo  y_hi  y_hi], ...
                       styles(2).Color, ...
                       'FaceAlpha', 0.35, ...
                       'EdgeColor', 'none');
        uistack(hPatch, 'bottom');
    
        % Reference lines
        yline(y_type_mean, ':', sprintf('\\mu=%.2f', y_type_mean), 'LineWidth', 1);
        yline(y_hi, ':', sprintf('+\\sigma=%.2f', y_hi), 'LineWidth', 1);
        yline(y_lo, ':', sprintf('-\\sigma=%.2f', y_lo), 'LineWidth', 1);
    
        ylim([y_lo - 0.5, y_hi + 0.5])


        elseif s == 3
        % Mean curve
        plot(t, body_ang_mean, styles(99).LineSpec, ...
             'Color', styles(99).Color, ...
             'LineWidth', 1.5);
        hold on;
    
        xl = xlim;
    
        th_lo = theta_type_mean - theta_type_std_global;
        th_hi = theta_type_mean + theta_type_std_global;
    
        % Shade mean ± global std (constant band)
        hPatch = patch([xl(1) xl(2) xl(2) xl(1)], ...
                       [th_lo th_lo th_hi th_hi], ...
                       styles(2).Color, ...
                       'FaceAlpha', 0.35, ...
                       'EdgeColor', 'none');
        uistack(hPatch, 'bottom');
    
        % Reference lines
        yline(theta_type_mean, ':', sprintf('\\mu=%.2f', theta_type_mean), 'LineWidth', 1);
        yline(th_hi, ':', sprintf('+\\sigma=%.2f', th_hi), 'LineWidth', 1);
        yline(th_lo, ':', sprintf('-\\sigma=%.2f', th_lo), 'LineWidth', 1);
    
        ylim([th_lo - 5, th_hi + 5])
    end

    grid on;
    xlabel('Time [s]');
    ylabel(ylab);
    title([ttl ' - ' titlePrefix]);
end