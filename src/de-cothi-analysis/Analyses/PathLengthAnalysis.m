%% Save dir
save_dir = '../Figures/';
%% Load paths
% Humans
load('Humans/humans.mat')
load('Humans/SR_IS.mat')
load('Humans/SR.mat')
human_SR = SR;
human_SR_imp = SR_imp;
% Rats
load('Rats/rat.mat')
load('Rats/SR_IS.mat')
load('Rats/SR.mat')
rat_SR = SR;
rat_SR_imp = SR_imp;
% Get the dimensions of the input array
[num_agents, num_mazes, num_start_locations] = size(humans);
% Initialize output arrays
median_path = zeros(num_mazes, num_start_locations);

%% Get Median path lengths and standard errors for humans
human_lengths = cellfun(@length, humans);
median_human = squeeze(median(human_lengths, 1));
stderr_human = squeeze(std(human_lengths, 0, 1)) ./ sqrt(num_agents);

human_sr_lengths = cellfun(@length, human_SR);
median_human_sr = squeeze(median(human_sr_lengths, 1));
stderr_human_sr = squeeze(std(human_sr_lengths, 0, 1)) ./ sqrt(num_agents);

human_sr_imp_lengths = cellfun(@length, human_SR_imp);
median_human_sr_imp = squeeze(median(human_sr_imp_lengths, 1));
stderr_human_sr_imp = squeeze(std(human_sr_imp_lengths, 0, 1)) ./ sqrt(num_agents);

%% Get Median path lengths and standard errors for rats
rat_lengths = cellfun(@length, rat);
median_rat = squeeze(median(rat_lengths, 1));
stderr_rat = squeeze(std(rat_lengths, 0, 1)) ./ sqrt(size(rat, 1));

rat_sr_lengths = cellfun(@length, rat_SR);
median_rat_sr = squeeze(median(rat_sr_lengths, 1));
stderr_rat_sr = squeeze(std(rat_sr_lengths, 0, 1)) ./ sqrt(size(rat_SR, 1));

rat_sr_imp_lengths = cellfun(@length, rat_SR_imp);
median_rat_sr_imp = squeeze(median(rat_sr_imp_lengths, 1));
stderr_rat_sr_imp = squeeze(std(rat_sr_imp_lengths, 0, 1)) ./ sqrt(size(rat_SR_imp, 1));

%% Plot median trajectories across starting locations for humans
maze_index = 15;
cmap = brewermap(6,'Set1');

% Extract data for the specified maze
start_locations = 1:size(median_human, 2);
median_data = {
    median_human(maze_index, :),
    median_human_sr(maze_index, :),
    median_human_sr_imp(maze_index, :)
};
stderr_data = {
    stderr_human(maze_index, :),
    stderr_human_sr(maze_index, :),
    stderr_human_sr_imp(maze_index, :)
};

% Plot median path lengths with standard error shading
figure;
hold on;

% Plot shaded error regions first (so they appear behind lines)
for i = 1:3
    y_median = median_data{i};
    y_stderr = stderr_data{i};
    
    % Create shaded region using fill
    x_fill = [start_locations, fliplr(start_locations)];
    y_fill = [y_median + y_stderr, fliplr(y_median - y_stderr)];
    
    % Fill with lighter version of the line color (fixed opacity)
    fill(x_fill, y_fill, cmap(i,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end

% Plot the median lines on top
for i = 1:3
    plot(start_locations, median_data{i}, 'Color', cmap(i,:), 'LineWidth', 2);
end

hold off;
title(sprintf('Maze %d (Humans)', maze_index), 'FontSize', 18, 'FontWeight','normal');
xlabel('Starting Location', 'FontSize', 16);
ylabel('Path Length', 'FontSize', 16);
legend('Human', 'SR', 'SR-IS', 'Location', 'best', 'FontSize', 12);
grid on;
set(gcf, 'color', 'w');
ax = gca;
set(gca, 'FontName', 'Times New Roman')
ax.YAxis.FontSize = 16;
ax.XAxis.FontSize = 16;
ax.Title.FontSize = 18;
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 8 3]);
exportgraphics(gcf, [save_dir,'human_maze15.pdf'], 'Resolution', 300);

%% Plot median trajectories across starting locations for rats
maze_index = 15;
cmap = brewermap(6,'Set1');

% Extract data for the specified maze
start_locations = 1:size(median_rat, 2);
median_data = {
    median_rat(maze_index, :),
    median_rat_sr(maze_index, :),
    median_rat_sr_imp(maze_index, :)
};
stderr_data = {
    stderr_rat(maze_index, :),
    stderr_rat_sr(maze_index, :),
    stderr_rat_sr_imp(maze_index, :)
};

% Plot median path lengths with standard error shading
figure;
hold on;

% Store handles for proper legend
h_fill = zeros(1, 3);
h_line = zeros(1, 3);

% Plot shaded error regions first (so they appear behind lines)
for i = 1:3
    y_median = median_data{i};
    y_stderr = stderr_data{i};
    
    % Create shaded region using fill
    x_fill = [start_locations, fliplr(start_locations)];
    y_fill = [y_median + y_stderr, fliplr(y_median - y_stderr)];
    
    % Fill with lighter version of the line color (fixed opacity)
    h_fill(i) = fill(x_fill, y_fill, cmap(i,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end

% Plot the median lines on top with different line widths
line_widths = [2, 2, 2]; % Thicker for Rat, thinner for SR and SR-IS
for i = 1:3
    h_line(i) = plot(start_locations, median_data{i}, 'Color', cmap(i,:), 'LineWidth', line_widths(i));
end

hold off;
title(sprintf('Maze %d (Rats)', maze_index), 'FontSize', 18, 'FontWeight', 'normal');
xlabel('Starting Location', 'FontSize', 16);
ylabel('Path Length', 'FontSize', 16);
legend('Rat', 'SR', 'SR-IS', 'Location', 'best', 'FontSize', 12);
grid on;
set(gcf, 'color', 'w');
ax = gca;
set(gca, 'FontName', 'Times New Roman')
ax.YAxis.FontSize = 16;
ax.XAxis.FontSize = 16;
ax.Title.FontSize = 18;
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 8 3]);
exportgraphics(gcf, [save_dir,'rat_maze15.pdf'], 'Resolution', 300);