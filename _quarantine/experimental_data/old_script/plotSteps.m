function plotSteps(files, trialIdx, RGB, styles, plot_sz, titlePrefix, f, ti)

    %% --- Load Data ---
    nTrials = numel(trialIdx);

    t_cell   = cell(1,nTrials);
    v_cell   = cell(1,nTrials);
    y_cell   = cell(1,nTrials);
    th_cell  = cell(1,nTrials);
    rot_cell = cell(1,nTrials);   % FIXED (was ntrials)

    for i = 1:nTrials
        dat = readmatrix(files(trialIdx(i)));

        t_raw  = dat(:,1);

        vx1 = -dat(:,4) * 1000;
        vx2 = -dat(:,8) * 1000;
        vx  = 0.5 * (vx1 + vx2);

        vy1 = -dat(:,5) * 1000;
        vy2 = -dat(:,9) * 1000;
        vy  = 0.5 * (vy1 + vy2);

        v_raw = sqrt(vx.^2 + vy.^2);

        y_raw = 0.5 * (dat(:,3) + dat(:,7)) * 1000;

        th_raw  = dat(:,12);
        rot_raw = dat(:,11);

        t_cell{i}   = t_raw;
        v_cell{i}   = v_raw;
        y_cell{i}   = y_raw;
        th_cell{i}  = th_raw;
        rot_cell{i} = rot_raw;   % STORE ROT
    end

    %% --- Build common time vector ---
    dt_trials = NaN(1,nTrials);
    t_start   = NaN(1,nTrials);
    t_end     = NaN(1,nTrials);

    for i = 1:nTrials
        tii = t_cell{i};
        tii = tii(isfinite(tii));

        if numel(tii) < 2
            continue
        end

        tii = sort(tii);
        tii = unique(tii,'stable');

        if numel(tii) < 2
            continue
        end

        dti = diff(tii);
        dti = dti(isfinite(dti) & dti > 0);

        if ~isempty(dti)
            dt_trials(i) = median(dti);
        end

        t_start(i) = tii(1);
        t_end(i)   = tii(end);
    end

    dt_common = median(dt_trials(isfinite(dt_trials)));

    t0 = min(t_start(isfinite(t_start)));
    t1 = max(t_end(isfinite(t_end)));

    t = (t0:dt_common:t1).';

    %% --- Interpolation ---
    v_mat   = NaN(numel(t), nTrials);
    y_mat   = NaN(numel(t), nTrials);
    th_mat  = NaN(numel(t), nTrials);
    rot_mat = NaN(numel(t), nTrials);   % NEW

    for i = 1:nTrials
        [tv, vv] = cleanForInterp(t_cell{i}, v_cell{i});
        
        if numel(tv) >= 2
            v_mat(:,i) = interp1(tv, vv, t, 'linear', NaN);
        end

        [ty, yy] = cleanForInterp(t_cell{i}, y_cell{i});
        if numel(ty) >= 2
            y_mat(:,i) = interp1(ty, yy, t, 'linear', NaN);
        end

        [tt, th] = cleanForInterp(t_cell{i}, th_cell{i});
        if numel(tt) >= 2
            th_mat(:,i) = interp1(tt, th, t, 'linear', NaN);
        end

        % ---------- ROT ----------
        [tr, rr] = cleanForInterp(t_cell{i}, rot_cell{i});
        if numel(tr) >= 2
            rot_mat(:,i) = interp1(tr, rr, t, 'linear', NaN);
        end
    end

    %% --- Means ---
    v_mean   = mean(v_mat,   2, 'omitnan');
    y_mean   = mean(y_mat,   2, 'omitnan');
    th_mean  = mean(th_mat,  2, 'omitnan');
    rot_mean = mean(rot_mat, 2, 'omitnan');   % NEW

%% --- Plot ---
meanStyle = styles(min(99, numel(styles)));

figure('Position', plot_sz.Position)

legendLabels = [ ...
    "Trial 1", ...
    "Trial 2", ...
    "Trial 3", ...
    "Mean" ];

%% ===============================
% 1) Speed
%% ===============================
subplot(4,1,1); hold on; grid on;

h = gobjects(nTrials+1,1);  % store handles

% Individual trials
for i = 1:nTrials
    h(i) = plot(t, v_mat(:,i), ...
        'Color', RGB(i,:), ...
        'LineWidth', 1);
end

% Mean curve
h(end) = plot(t, v_mean, meanStyle.LineSpec, ...
    'Color', meanStyle.Color, ...
    'LineWidth', 2);

ylabel('v [mm/s]');
title(['Forward Speed vs. Time - ' titlePrefix]);

legend(h, legendLabels, 'Location','best');

%% ===============================
% 2) Height
%% ===============================
subplot(4,1,2); hold on; grid on;

for i = 1:nTrials
    plot(t, y_mat(:,i), ...
        'Color', RGB(i,:), ...
        'LineWidth', 1);
end

plot(t, y_mean, meanStyle.LineSpec, ...
    'Color', meanStyle.Color, ...
    'LineWidth', 2);

ylabel('height [mm]');
title('Body Height vs. Time');

%% ===============================
% 3) Theta
%% ===============================
subplot(4,1,3); hold on; grid on;

for i = 1:nTrials
    plot(t, th_mat(:,i), ...
        'Color', RGB(i,:), ...
        'LineWidth', 1);
end

plot(t, th_mean, meanStyle.LineSpec, ...
    'Color', meanStyle.Color, ...
    'LineWidth', 2);

ylim([-30 15])
ylabel('\theta [rad]');
title('Body Angle vs. Time');

%% ===============================
% 4) Rotation
%% ===============================
subplot(4,1,4); hold on; grid on;

for i = 1:nTrials
    plot(t, rot_mat(:,i), ...
        'Color', RGB(i,:), ...
        'LineWidth', 1);
end

plot(t, rot_mean, meanStyle.LineSpec, ...
    'Color', meanStyle.Color, ...
    'LineWidth', 2);

ylabel('rot');
xlabel('Time [s]');
title('Rotation vs. Time');
end


function [t_clean, x_clean] = cleanForInterp(t, x)

    m = isfinite(t) & isfinite(x);
    t = t(m);
    x = x(m);

    if isempty(t)
        t_clean = [];
        x_clean = [];
        return
    end

    [t, order] = sort(t);
    x = x(order);

    [t_clean, ia] = unique(t,'stable');
    x_clean = x(ia);

end