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

% Get the dimensions of the input array
[num_subj, num_mazes, num_start_locations] = size(dat);

% Initialize output arrays
paths_array = cell(num_subj, num_mazes, num_start_locations);
median_path = zeros(num_subj, num_mazes, num_start_locations);

%% Clean data
% Loop through every element in the 3D array
for subject = 1:num_subj
    for maze = 1:num_mazes
        for start_point = 1:num_start_locations
            
            % Get the current path
            current_path = dat{subject, maze, start_point};
            
            % Check if the path is not empty
            if ~isempty(current_path)
                % Find the first occurrence of index 34
                first_34_idx = find(current_path == 34, 1, 'first');
                
                % If index 34 is found, trim the path
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

%% 1. Average path length for each maze (across all subjects and starting points)
% Result: array of size (25,1) - one average per maze
avg_path_length_by_maze = squeeze(mean(lengths, [1, 3]));
sr_avg_path_length_by_maze = squeeze(mean(sr_lengths, [1,3]));
sr_is_avg_path_length_by_maze = squeeze(mean(sr_is_lengths, [1,3]));

% Differences
sr_maze_diffs = (avg_path_length_by_maze - sr_avg_path_length_by_maze).^2;
sr_is_maze_diffs = (avg_path_length_by_maze - sr_is_avg_path_length_by_maze).^2;

fprintf('Mean, Std difference across mazes SR: %.3f\n', mean(sr_maze_diffs));
fprintf('Mean, Std difference across mazes SR-IS: %.3f\n', mean(sr_is_maze_diffs));

% Plot differences across all mazes %
figure('Position', [100, 100, 700, 400]);

x = 1:length(sr_maze_diffs);
plot(x, sr_maze_diffs, 'o-', 'LineWidth', 2, 'MarkerSize', 6, 'Color', cmap(2,:), 'DisplayName', 'SR');
hold on;
plot(x, sr_is_maze_diffs, 's-', 'LineWidth', 2, 'MarkerSize', 6, 'Color', cmap(3,:), 'DisplayName', 'SR-IS');

% Add average lines with alpha versions of the colors
avg_SR = mean(sr_maze_diffs);
avg_SR_IS = mean(sr_is_maze_diffs);
plot([0.5, length(sr_maze_diffs)+0.5], [avg_SR, avg_SR], ...
     '--', 'LineWidth', 3, 'Color', [cmap(2,:) 0.6], 'DisplayName', sprintf('Avg = %.2f', avg_SR));
plot([0.5, length(sr_is_maze_diffs)+0.5], [avg_SR_IS, avg_SR_IS], ...
     '--', 'LineWidth', 3, 'Color', [cmap(3,:) 0.6], 'DisplayName', sprintf('Avg = %.2f', avg_SR_IS));

title(sprintf('Squared Maze Differences (%s)', species), 'FontSize', 22, 'FontWeight', 'normal', 'FontName', 'Times New Roman');
xlabel('Maze Number', 'FontSize', 20, 'FontName', 'Times New Roman');
ylabel('Squared Difference', 'FontSize', 20, 'FontName', 'Times New Roman');
legend('show', 'Location', 'best', 'FontName', 'Times New Roman');
grid on;
xlim([0.5, length(sr_maze_diffs)+0.5]);
% exportgraphics(gcf, [save_dir,'MazeDiffs_PathLength_Humans.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none');
exportgraphics(gcf, [save_dir,'MazeDiffs_PathLength_Rats.pdf'], 'ContentType', 'vector', 'BackgroundColor', 'none');

%% 2. Average path length for each subject (across all mazes and starting points)
% Result: array of size (18,1) - one average per subject
avg_path_length_by_subject = squeeze(mean(lengths, [2, 3]));
sr_avg_path_length_by_subject = squeeze(mean(sr_lengths, [2,3]));
sr_is_avg_path_length_by_subject = squeeze(mean(sr_is_lengths, [2,3]));

% Calculate differences for each of the 1800 individual runs
avg_path_length_by_subject_expanded = repelem(avg_path_length_by_subject, 100);
sr_subj_diffs_all = (avg_path_length_by_subject_expanded - sr_avg_path_length_by_subject).^2;
sr_is_subj_diffs_all = (avg_path_length_by_subject_expanded - sr_is_avg_path_length_by_subject).^2;

% Average back to 18 points (one per subject) with standard error
sr_subj_diffs = arrayfun(@(i) mean(sr_subj_diffs_all((i-1)*100+1:i*100)), 1:num_subj);
sr_is_subj_diffs = arrayfun(@(i) mean(sr_is_subj_diffs_all((i-1)*100+1:i*100)), 1:num_subj);
sr_subj_se = arrayfun(@(i) std(sr_subj_diffs_all((i-1)*100+1:i*100))/sqrt(100), 1:num_subj);
sr_is_subj_se = arrayfun(@(i) std(sr_is_subj_diffs_all((i-1)*100+1:i*100))/sqrt(100), 1:num_subj);

fprintf('Mean, Std difference across subjects SR: %.3f\n', mean(sr_subj_diffs));
fprintf('Mean, Std difference across subjects SR-IS: %.3f\n', mean(sr_is_subj_diffs));

% Plot differences across all subjects with error bars
figure('Position', [100, 100, 700, 400]);
x = 1:num_subj;

% Add shaded error bands
x_fill = [x, fliplr(x)];
y_upper_sr = sr_subj_diffs + sr_subj_se;
y_lower_sr = max(0, sr_subj_diffs - sr_subj_se);
y_fill_sr = [y_upper_sr, fliplr(y_lower_sr)];
fill(x_fill, y_fill_sr, cmap(2,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

hold on;

y_upper_sr_is = sr_is_subj_diffs + sr_is_subj_se;
y_lower_sr_is = max(0, sr_is_subj_diffs - sr_is_subj_se);
y_fill_sr_is = [y_upper_sr_is, fliplr(y_lower_sr_is)];
fill(x_fill, y_fill_sr_is, cmap(3,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

% Plot main lines
plot(x, sr_subj_diffs, 'o-', 'LineWidth', 2, 'MarkerSize', 6, 'Color', cmap(2,:), 'DisplayName', 'SR');
plot(x, sr_is_subj_diffs, 's-', 'LineWidth', 2, 'MarkerSize', 6, 'Color', cmap(3,:), 'DisplayName', 'SR-IS');

% Add average lines with alpha versions of the colors
avg_SR = mean(sr_subj_diffs);
avg_SR_IS = mean(sr_is_subj_diffs);
last_point = num_subj+0.1;
plot([0.5, last_point], [avg_SR, avg_SR], ...
'--', 'LineWidth', 3, 'Color', [cmap(2,:) 0.6], 'DisplayName', sprintf('Avg = %.2f', avg_SR));
plot([0.5, last_point], [avg_SR_IS, avg_SR_IS], ...
'--', 'LineWidth', 3, 'Color', [cmap(3,:) 0.6], 'DisplayName', sprintf('Avg = %.2f', avg_SR_IS));

title(sprintf('Squared Subject Differences (%s)', species), 'FontSize', 22, 'FontWeight', 'normal', 'FontName', 'Times New Roman');
xlabel('Subject Number', 'FontSize', 20, 'FontName', 'Times New Roman');
% ylabel('Squared Difference', 'FontSize', 16);
legend('show', 'Location', 'best', 'FontName', 'Times New Roman');
grid on;
xlim([0.5, last_point]);
% exportgraphics(gcf, [save_dir,'SubjectDiffs_PathLength_Humans.pdf'], 'Resolution', 300, 'ContentType', 'vector');
exportgraphics(gcf, [save_dir,'SubjectDiffs_PathLength_Rats.pdf'], 'Resolution', 300, 'ContentType', 'vector');



