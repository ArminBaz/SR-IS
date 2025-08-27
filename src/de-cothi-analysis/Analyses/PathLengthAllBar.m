%% Save dir
save_dir = '../figures/';
cmap = brewermap(5,'Set1');

%% Load paths
% Humans
load('Humans/humans.mat')
load('Humans/SR_IS.mat')
load('Humans/SR.mat')

human_SR = SR;
human_SR_IS = SR_imp;

% Rats
load('Rats/rat.mat')
load('Rats/SR_IS.mat')
load('Rats/SR.mat')

rat_SR = SR;
rat_SR_IS = SR_imp;

%% Humans
% dat = humans;
% SR = human_SR;
% SR_IS = human_SR_IS;
% species = 'Humans';

%% Rats
dat = rat;
SR = rat_SR;
SR_IS = rat_SR_IS;
species = 'Rats';

%% Dimensions
[num_subj, num_mazes, num_start_locations] = size(dat);
paths_array = cell(num_subj, num_mazes, num_start_locations);

%% Clean data
for subject = 1:num_subj
    for maze = 1:num_mazes
        for start_point = 1:num_start_locations
            current_path = dat{subject, maze, start_point};
            
            if ~isempty(current_path)
                first_34_idx = find(current_path == 34, 1, 'first');
                
                if ~isempty(first_34_idx)
                    paths_array{subject, maze, start_point} = current_path(1:first_34_idx);
                else
                    paths_array{subject, maze, start_point} = current_path;
                end
            end
        end
    end
end

%% Perform path length analysis
lengths = cellfun(@length, paths_array);
sr_lengths = cellfun(@length, SR);
sr_is_lengths = cellfun(@length, SR_IS);

%% 1. MAZE ANALYSIS - Bar Plot
avg_path_length_by_maze = squeeze(mean(lengths, [1, 3]));
sr_avg_path_length_by_maze = squeeze(mean(sr_lengths, [1,3]));
sr_is_avg_path_length_by_maze = squeeze(mean(sr_is_lengths, [1,3]));

% Calculate squared differences for each maze
sr_maze_diffs = (avg_path_length_by_maze - sr_avg_path_length_by_maze).^2;
sr_is_maze_diffs = (avg_path_length_by_maze - sr_is_avg_path_length_by_maze).^2;

% Calculate means and standard errors
mean_sr_maze = mean(sr_maze_diffs);
mean_sr_is_maze = mean(sr_is_maze_diffs);
se_sr_maze = std(sr_maze_diffs) / sqrt(length(sr_maze_diffs));
se_sr_is_maze = std(sr_is_maze_diffs) / sqrt(length(sr_is_maze_diffs));

% Statistical test (paired t-test since we're comparing same mazes)
[h_maze, p_maze, ~, stats_maze] = ttest(sr_maze_diffs, sr_is_maze_diffs);

fprintf('=== MAZE ANALYSIS ===\n');
fprintf('Mean squared difference SR: %.3f ± %.3f\n', mean_sr_maze, se_sr_maze);
fprintf('Mean squared difference SR-IS: %.3f ± %.3f\n', mean_sr_is_maze, se_sr_is_maze);
fprintf('Paired t-test: t(%d) = %.3f, p = %.4f\n', stats_maze.df, stats_maze.tstat, p_maze);
if h_maze
    fprintf('*** SIGNIFICANT DIFFERENCE (p < 0.05) ***\n\n');
else
    fprintf('No significant difference (p >= 0.05)\n\n');
end

% Create bar plot for maze analysis
figure('Position', [100, 100, 400, 300]);
bar_data = [mean_sr_maze, mean_sr_is_maze];
error_data = [se_sr_maze, se_sr_is_maze];

b = bar(bar_data, 'LineWidth', 2, 'FaceColor', 'w');
b.BarWidth = 0.6;

hold on;
set(gca,'LineWidth',2)
% Add error bars
errorbar(1:2, bar_data, error_data, 'k', 'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 10);

% Extend y-axis by one tick interval to make room for potential significance indicators
current_ticks = get(gca, 'YTick');
tick_interval = current_ticks(2) - current_ticks(1);
current_ylim = ylim;
ylim([current_ylim(1), current_ylim(2) + tick_interval]);

% Add significance indicator if significant
if h_maze
    y_max = max(bar_data + error_data) * 1.1;
    plot([1, 2], [y_max, y_max], 'k-', 'LineWidth', 2);
    plot([1, 1], [y_max - y_max*0.02, y_max], 'k-', 'LineWidth', 2);
    plot([2, 2], [y_max - y_max*0.02, y_max], 'k-', 'LineWidth', 2);
    
    % Add asterisk above the significance bar
    text(1.5, y_max * 1.05, '*', 'FontSize', 16, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'Times New Roman');
    
    % Extend y-axis by one more tick
    current_ticks = get(gca, 'YTick');
    tick_interval = current_ticks(2) - current_ticks(1);
    new_max_tick = current_ticks(end) + tick_interval;
    set(gca, 'YTick', [current_ticks, new_max_tick]);
    ylim([0, new_max_tick]);
    ylim([0, new_max_tick]);
end

title(sprintf('(%s)', species), ...
      'FontSize', 20, 'FontWeight', 'normal', 'FontName', 'Times New Roman');
ylabel('Average Difference', 'FontSize', 18, 'FontName', 'Times New Roman');
set(gca, 'XTickLabel', {'SR', 'SR-IS'}, 'FontSize', 18, 'FontName', 'Times New Roman');
box off;

% Save figure
% exportgraphics(gcf, [save_dir,'BarPlot_MazeDiffs_PathLength_', species, '.pdf'], ...
%               'ContentType', 'vector', 'BackgroundColor', 'none');

%% 2. SUBJECT ANALYSIS - Bar Plot
avg_path_length_by_subject = squeeze(mean(lengths, [2, 3]));
sr_avg_path_length_by_subject = squeeze(mean(sr_lengths, [2,3]));
sr_is_avg_path_length_by_subject = squeeze(mean(sr_is_lengths, [2,3]));

% Calculate differences for each of the individual runs (following original method)
avg_path_length_by_subject_expanded = repelem(avg_path_length_by_subject, 100);
sr_subj_diffs_all = (avg_path_length_by_subject_expanded - sr_avg_path_length_by_subject).^2;
sr_is_subj_diffs_all = (avg_path_length_by_subject_expanded - sr_is_avg_path_length_by_subject).^2;

% Average back to subject-level differences (one per subject)
sr_subj_diffs = arrayfun(@(i) mean(sr_subj_diffs_all((i-1)*100+1:i*100)), 1:num_subj);
sr_is_subj_diffs = arrayfun(@(i) mean(sr_is_subj_diffs_all((i-1)*100+1:i*100)), 1:num_subj);

% Calculate means and standard errors
mean_sr_subj = mean(sr_subj_diffs);
mean_sr_is_subj = mean(sr_is_subj_diffs);
se_sr_subj = std(sr_subj_diffs) / sqrt(length(sr_subj_diffs));
se_sr_is_subj = std(sr_is_subj_diffs) / sqrt(length(sr_is_subj_diffs));

% Statistical test (paired t-test since we're comparing same subjects)
[h_subj, p_subj, ~, stats_subj] = ttest(sr_subj_diffs, sr_is_subj_diffs);

fprintf('=== SUBJECT ANALYSIS ===\n');
fprintf('Mean squared difference SR: %.3f ± %.3f\n', mean_sr_subj, se_sr_subj);
fprintf('Mean squared difference SR-IS: %.3f ± %.3f\n', mean_sr_is_subj, se_sr_is_subj);
fprintf('Paired t-test: t(%d) = %.3f, p = %.4f\n', stats_subj.df, stats_subj.tstat, p_subj);
if h_subj
    fprintf('*** SIGNIFICANT DIFFERENCE (p < 0.05) ***\n\n');
else
    fprintf('No significant difference (p >= 0.05)\n\n');
end

% Create bar plot for subject analysis
figure('Position', [100, 100, 400, 300]);
bar_data_subj = [mean_sr_subj, mean_sr_is_subj];
error_data_subj = [se_sr_subj, se_sr_is_subj];

b2 = bar(bar_data_subj, 'LineWidth', 2, 'FaceColor', 'w');
b2.BarWidth = 0.6;

hold on;
set(gca,'LineWidth',2)
% Add error bars
errorbar(1:2, bar_data_subj, error_data_subj, 'k', 'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 10);

% Extend y-axis by one tick interval to make room for potential significance indicators
current_ticks = get(gca, 'YTick');
tick_interval = current_ticks(2) - current_ticks(1);
current_ylim = ylim;
ylim([current_ylim(1), current_ylim(2) + tick_interval]);

% Add significance indicator if significant
if h_subj
    y_max = max(bar_data_subj + error_data_subj) * 1.1;
    plot([1, 2], [y_max, y_max], 'k-', 'LineWidth', 2);
    plot([1, 1], [y_max - y_max*0.02, y_max], 'k-', 'LineWidth', 2);
    plot([2, 2], [y_max - y_max*0.02, y_max], 'k-', 'LineWidth', 2);
    
    % Add asterisk above the significance bar
    text(1.5, y_max * 1.05, '*', 'FontSize', 16, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'FontName', 'Times New Roman');
    
    % Extend y-axis by one more tick
    current_ticks = get(gca, 'YTick');
    tick_interval = current_ticks(2) - current_ticks(1);
    new_max_tick = current_ticks(end) + tick_interval;
    set(gca, 'YTick', [current_ticks, new_max_tick]);
end

title(sprintf('%s', species), ...
      'FontSize', 20, 'FontWeight', 'normal', 'FontName', 'Times New Roman');
ylabel('Average Difference', 'FontSize', 18, 'FontName', 'Times New Roman');
set(gca, 'XTickLabel', {'SR', 'SR-IS'}, 'FontSize', 18, 'FontName', 'Times New Roman');
box off;

% Save figure
exportgraphics(gcf, [save_dir,'BarPlot_SubjectDiffs_PathLength_', species, '.pdf'], ...
    'ContentType', 'vector', 'BackgroundColor', 'none');