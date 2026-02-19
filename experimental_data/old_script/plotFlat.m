function plotFlat(files, trialIdx, RGB, styles, plot_sz, titlePrefix, points, f, ti)

    %% --- Load Data ---
    nTrials = numel(trialIdx);
    v_1 = cell(1,nTrials);
    v_2 = cell(1,nTrials);
    v_x = cell(1,nTrials);
    y = cell(1,nTrials);
    body_ang = cell(1,nTrials);
    rot = cell(1,nTrials);

    for i = 1:nTrials
        dat = readmatrix(files(trialIdx(i)));
        if i == 1
            t = dat(1:points,1); % time [s]
        end

        v_1{i}      = -dat(1:points,4) * 1000;
        v_2{i}      = -dat(1:points,8) * 1000;
        v_x{i}      = 0.5 * (v_1{i} + v_2{i});

        y_raw1       = dat(1:points,3);
        y_raw2       = dat(1:points,7);
        y_raw        = 0.5.*(y_raw1+y_raw2);

        y{i}        = (y_raw - y_raw(3) + min(y_raw)) * 1000;

        body_ang{i} = dat(1:points,12);
        rot{i}      = dat(1:points,11);
    end

    % Compute means across trials
    v_x_mean      = mean(cat(2, v_x{:}), 2);
    v_x_mat = cat(2, v_x{:});          % [time × trials]
    v_x_std = std(v_x_mat, 0, 2);      % std across trials

    y_mean        = mean(cat(2, y{:}), 2);
    y_mat = cat(2, y{:});          % [time × trials]
    y_min_global = min(y_mat, [], 'all');
    y_max_global = max(y_mat, [], 'all');

    body_ang_mean = mean(cat(2, body_ang{:}), 2);
    theta_mat = cat(2, body_ang{:});   % [time × trials]
    theta_max_global = max(theta_mat, [], 'all');
    theta_min_global = min(theta_mat, [], 'all');

    rot_mean = mean(cat(2, rot{:}), 2);
    rot_mat = cat(2, rot{:});   % [time × trials]
    rot_max_global = max(rot_mat, [], 'all');
    rot_min_global = min(rot_mat, [], 'all');


    % ---- Steady-state mean AFTER t = 0.3 s (computed per call) ----
    idx_steady = t > ti;
    v_x_steady_mean = mean(v_x_mean(idx_steady), 'omitnan');

    % ---- Per-trial steady-state mean speed (scalar per trial) ----
    v_x_trial_mean = nan(1, nTrials);
    for k = 1:nTrials
        v_x_trial_mean(k) = mean(v_x{k}(idx_steady), 'omitnan');  % scalar
    end
    
    % ---- Mean/std across trials of those steady means ----
    v_x_steady_mean = mean(v_x_trial_mean, 'omitnan');            % scalar
    v_x_steady_std  = std(v_x_trial_mean, 0, 2, 'omitnan');       % scalar


    % ---- First time mean speed reaches steady-state average ----
    idx_cross = find(v_x_mean >= v_x_steady_mean, 1, 'first');
    
    if ~isempty(idx_cross)
        t_reach_avg = t(idx_cross);
    else
        t_reach_avg = NaN;  % safety fallback
    end

    
% ######################

% ---- Global std of ALL steady-state data points ----
y_all_steady     = y_mat(idx_steady, :);
theta_all_steady = theta_mat(idx_steady, :);
rot_all_steady   = rot_mat(idx_steady,:);

y_type_mean     = mean(y_all_steady(:), 'omitnan');
theta_type_mean = mean(theta_all_steady(:), 'omitnan');
rot_type_mean   = mean(rot_all_steady(:), 'omitnan');

y_type_std_global     = std(y_all_steady(:), 0, 'omitnan');
theta_type_std_global = std(theta_all_steady(:), 0, 'omitnan');
rot_type_std_global = std(rot_all_steady(:), 0, 'omitnan');

% ###########################


    %% --- Plot Time Series ---
    series = { {v_x, v_x_mean, 'v_x [mm/s]', 'Forward Speed vs. Time'}, ...
               {y, y_mean, 'height (mm)', 'Body Height vs. Time'}, ...
               {body_ang, body_ang_mean, '\theta [deg]', 'Body Angle vs Time'},...
               {rot, rot_mean, '\omega [deg/s]', 'angular velocity vs Time'}};


figure('Position', plot_sz.Position)

for s = 1:4
    yMean  = series{s}{2};
    ylab   = series{s}{3};
    ttl    = series{s}{4};
    trials = series{s}{1};

    subplot(4,1,s); hold on;

        for k = 1:nTrials
            plot(t, trials{k}, '-', 'Color', RGB(k,:), 'LineWidth', 1.5);
            legend('1','2','3','4')
        end

    if s == 1
        ylim([-50, 600])
    
        % Plot mean time-series (optional: keep this as your main curve)
        plot(t, v_x_mean, styles(99).LineSpec, ...
             'Color', styles(99).Color, ...
             'LineWidth', 1.5);
        hold on;
    
        % Constant band from trial-wise steady means
        xl = xlim;
        v_lo = v_x_steady_mean - v_x_steady_std;
        v_hi = v_x_steady_mean + v_x_steady_std;
    
        hPatch = patch([xl(1) xl(2) xl(2) xl(1)], ...
                       [v_lo  v_lo  v_hi  v_hi], ...
                       styles(2).Color, ...
                       'FaceAlpha', 0.35, ...
                       'EdgeColor', 'none');
        uistack(hPatch, 'bottom');
    
        % Reference line for steady mean
        yline(v_x_steady_mean, 'r--', ...
            sprintf('Steady mean=%.1f, std=%.1f', v_x_steady_mean, v_x_steady_std), ...
            'LineWidth', 1.2, ...
            'LabelHorizontalAlignment','right', ...
            'LabelVerticalAlignment','bottom');
    

            if ~isnan(t_reach_avg)
            xline(t_reach_avg, 'b--', ...
                sprintf('Reach mean @ %.2f ms', t_reach_avg.*1000), ...
                'LineWidth', 1.2, ...
                'LabelVerticalAlignment', 'top', ...
                'LabelHorizontalAlignment', 'left');
            end

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

        elseif s == 4
        % Mean curve
        plot(t, rot_mean, styles(99).LineSpec, ...
             'Color', styles(99).Color, ...
             'LineWidth', 1.5);
        hold on;
    
        xl = xlim;
    
        r_lo = rot_type_mean - rot_type_std_global;
        r_hi = rot_type_mean + rot_type_std_global;
    
        % Shade mean ± global std (constant band)
        hPatch = patch([xl(1) xl(2) xl(2) xl(1)], ...
                       [r_lo r_lo r_hi r_hi], ...
                       styles(2).Color, ...
                       'FaceAlpha', 0.35, ...
                       'EdgeColor', 'none');
        uistack(hPatch, 'bottom');
    
        % Reference lines
        yline(rot_type_mean, ':', sprintf('\\mu=%.2f', rot_type_mean), 'LineWidth', 1);
        yline(r_hi, ':', sprintf('+\\omega=%.2f', r_hi), 'LineWidth', 1);
        yline(r_lo, ':', sprintf('-\\omega=%.2f', r_lo), 'LineWidth', 1);
    
        ylim([r_lo - 20000, r_hi + 20000])


    end

    grid on;
    xlabel('Time [s]');
    ylabel(ylab);
    title([ttl ' - ' titlePrefix]);
end