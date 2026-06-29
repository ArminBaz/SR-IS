% plot_confusion_matrices.m
% Reads checkpoints/, computes BIC winners, and plots confusion matrices for humans and rats side by side.

% 1=MF  2=MB  3=SR  4=Hybrid  5=SR-IS
% fit_cols = [1 2 3 4 5];   % All models
% fit_cols = [4 5];         % Hybrid vs. SR-IS
% fit_cols = [3,5];         % SR vs. SR-IS
fit_cols = [3,4, 5];         % SR vs. Hybrid vs. SR-IS

% -------------------------------------------------------------------------
n_humans     = 18;
n_rats       = 9;
n_models     = 5;
n_params_vec = [2, 1, 2, 3, 3];
n_datapoints = 25 * 10;
model_names  = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};
gen_tags     = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};

% Build full 5x5 confusion matrices: C(gen_model, fitted_winner)
C_human = zeros(n_models, n_models);
C_rat   = zeros(n_models, n_models);

for gen_idx = 1:n_models
    gen_name = gen_tags{gen_idx};

    % Humans
    for r = 1:n_humans
        ckpt = sprintf('checkpoints/%s_human_%d.mat', gen_name, r);
        if exist(ckpt, 'file')
            tmp = load(ckpt, 'sim_ll_r');
            bic = -2*tmp.sim_ll_r + n_params_vec' .* log(n_datapoints);
            % Only consider fitted models in fit_cols when picking winner
            bic_sub = inf(n_models, 1);
            bic_sub(fit_cols) = bic(fit_cols);
            [~, winner] = min(bic_sub);
            C_human(gen_idx, winner) = C_human(gen_idx, winner) + 1;
        end
    end

    % Rats
    for r = 1:n_rats
        ckpt = sprintf('checkpoints/%s_rat_%d.mat', gen_name, r);
        if exist(ckpt, 'file')
            tmp = load(ckpt, 'sim_ll_r');
            bic = -2*tmp.sim_ll_r + n_params_vec' .* log(n_datapoints);
            bic_sub = inf(n_models, 1);
            bic_sub(fit_cols) = bic(fit_cols);
            [~, winner] = min(bic_sub);
            C_rat(gen_idx, winner) = C_rat(gen_idx, winner) + 1;
        end
    end
end

% Slice to only the selected fitted columns
C_human = C_human(:, fit_cols);
C_rat   = C_rat(:,   fit_cols);

% Row-normalise to proportions
C_human_pct = 100 * C_human ./ max(sum(C_human, 2), 1);
C_rat_pct   = 100 * C_rat   ./ max(sum(C_rat,   2), 1);

n_fit = numel(fit_cols);
col_labels = model_names(fit_cols);

% -------------------------------------------------------------------------
%  Plot
% -------------------------------------------------------------------------
font_name = 'Helvetica Neue';
font_sz   = 9;
cell_font = 8;

fig = figure('Units','inches','Position',[1 1 10 4.4], ...
             'Color','w','Renderer','painters');

titles   = {'Humans', 'Rats'};
C_list   = {C_human_pct, C_rat_pct};
raw_list = {C_human, C_rat};

ax_w  = 0.34;   ax_h = 0.62;
ax_b  = 0.18;
ax_pos = {[0.07  ax_b  ax_w ax_h], ...
          [0.52  ax_b  ax_w ax_h]};

axs = gobjects(1,2);
for sp = 1:2
    ax = axes('Position', ax_pos{sp});   axs(sp) = ax;
    C    = C_list{sp};
    Craw = raw_list{sp};

    imagesc(C);
    clim([0 100]);
    colormap(ax, flipud(gray));

    set(ax, 'XTick', 1:n_fit,     'YTick', 1:n_models, ...
            'XTickLabel', col_labels, 'YTickLabel', model_names, ...
            'TickLength', [0 0], ...
            'FontName', font_name, 'FontSize', font_sz, ...
            'XAxisLocation', 'top', ...
            'TickDir', 'out', ...
            'XColor', [0 0 0], 'YColor', [0 0 0]);
    xtickangle(ax, 0);

    xlabel(ax, 'Fitted model',    'FontName', font_name, 'FontSize', font_sz, 'FontWeight', 'normal');
    ylabel(ax, 'Simulated model', 'FontName', font_name, 'FontSize', font_sz, 'FontWeight', 'normal');
    title(ax,  titles{sp},        'FontName', font_name, 'FontSize', font_sz+1, 'FontWeight', 'normal');

    % Cell annotations
    for row = 1:n_models
        for col = 1:n_fit
            val = C(row, col);
            raw = Craw(row, col);
            txt_color = [0 0 0];
            if val >= 50; txt_color = [1 1 1]; end
            text(col, row, sprintf('%d%%\n(%d)', round(val), raw), ...
                 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment',   'middle', ...
                 'FontName', font_name, ...
                 'FontSize', cell_font, ...
                 'Color',    txt_color);
        end
    end

    % Grid lines
    hold(ax, 'on');
    for k = 0.5:1:(n_fit+0.5)
        plot(ax, [k k],             [0.5 n_models+0.5], 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
    end
    for k = 0.5:1:(n_models+0.5)
        plot(ax, [0.5 n_fit+0.5],   [k k],              'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
    end
    hold(ax, 'off');
end

% Colorbar
cb = colorbar(axs(2), 'Location', 'eastoutside');
cb.Ticks          = [0 100];
cb.TickLength     = 0;
cb.Label.String   = 'Proportion recovered (%)';
cb.Label.FontName = font_name;
cb.Label.FontSize = font_sz;
cb.Label.Color    = [0 0 0];
cb.FontName       = font_name;
cb.FontSize       = font_sz - 1;
cb.Color          = [0 0 0];

% Restore both axes to same size
axs(1).Position = ax_pos{1};
axs(2).Position = ax_pos{2};

set(fig, 'PaperPositionMode', 'auto');
