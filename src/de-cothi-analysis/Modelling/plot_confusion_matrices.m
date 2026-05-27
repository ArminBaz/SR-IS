% plot_confusion_matrices.m
% Reads checkpoints/, computes BIC winners, and plots confusion matrices for humans and rats side by side.

n_humans     = 18;
n_rats       = 9;
n_models     = 5;
n_params_vec = [2, 1, 2, 3, 3];
n_datapoints = 25 * 10;
model_names  = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};
gen_tags     = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};

% Build confusion matrices: C(gen_model, fitted_winner)
C_human = zeros(n_models, n_models);
C_rat   = zeros(n_models, n_models);

for gen_idx = 1:n_models
    gen_name = gen_tags{gen_idx};

    % Humans
    n_ppt = n_humans;
    for r = 1:n_ppt
        ckpt = sprintf('checkpoints/%s_human_%d.mat', gen_name, r);
        if exist(ckpt, 'file')
            tmp = load(ckpt, 'sim_ll_r');
            bic = -2*tmp.sim_ll_r + n_params_vec' .* log(n_datapoints);
            [~, winner] = min(bic);
            C_human(gen_idx, winner) = C_human(gen_idx, winner) + 1;
        end
    end

    % Rats
    n_ppt = n_rats;
    for r = 1:n_ppt
        ckpt = sprintf('checkpoints/%s_rat_%d.mat', gen_name, r);
        if exist(ckpt, 'file')
            tmp = load(ckpt, 'sim_ll_r');
            bic = -2*tmp.sim_ll_r + n_params_vec' .* log(n_datapoints);
            [~, winner] = min(bic);
            C_rat(gen_idx, winner) = C_rat(gen_idx, winner) + 1;
        end
    end
end

% Row-normalise to proportions
row_sum_human = sum(C_human, 2);
row_sum_rat   = sum(C_rat,   2);
C_human_pct = 100 * C_human ./ max(row_sum_human, 1);
C_rat_pct   = 100 * C_rat   ./ max(row_sum_rat,   1);

% -------------------------------------------------------------------------
%  Plot
% -------------------------------------------------------------------------
font_name = 'Helvetica Neue';
font_sz   = 9;
cell_font = 8;

% Figure dimensions (inches): two equal square panels + colorbar on right
fig_w = 10;   fig_h = 4.4;
fig = figure('Units','inches','Position',[1 1 fig_w fig_h], ...
             'Color','w','Renderer','painters');

titles   = {'Humans', 'Rats'};
C_list   = {C_human_pct, C_rat_pct};
raw_list = {C_human, C_rat};

% Manual axes positions [left bottom width height] in normalised units.
% Both panels identical size; colorbar sits to the right of panel 2.
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
    axis square;

    set(ax, 'XTick', 1:n_models, 'YTick', 1:n_models, ...
            'XTickLabel', model_names, 'YTickLabel', model_names, ...
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
        for col = 1:n_models
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
    for k = 0.5:1:(n_models+0.5)
        plot(ax, [k k],              [0.5 n_models+0.5], 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
        plot(ax, [0.5 n_models+0.5], [k k],              'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
    end
    hold(ax, 'off');
end

% Colorbar anchored to right panel; manually position so it doesn't resize axs(2)
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

% Restore both axes to same size (colorbar call may have shrunk axs(2))
axs(1).Position = ax_pos{1};
axs(2).Position = ax_pos{2};

set(fig, 'PaperPositionMode', 'auto');

% Save
out_pdf = 'confusion_matrices.pdf';
out_png = 'confusion_matrices.png';
exportgraphics(fig, out_pdf, 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, out_png, 'Resolution', 300,       'BackgroundColor', 'white');
fprintf('Saved %s and %s\n', out_pdf, out_png);
