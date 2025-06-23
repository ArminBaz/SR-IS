%% Save dir
save_dir = '../Figures/';

%% Load paths
% Humans
load('Humans/humans.mat')
load('Humans/SR_imp_alpha0.6_gamma0.5.mat')
load('Humans/SR.mat')

human_SR = SR;
human_SR_imp = SR_imp;

% Rats
load('Rats/rat.mat')
load('Rats/SR_imp_alpha0.6_gamma0.5.mat')
load('Rats/SR.mat')

rat_SR = SR;
rat_SR_imp = SR_imp;

% Get the dimensions of the input array
[num_agents, num_mazes, num_start_locations] = size(humans);

% Initialize output arrays
median_path = zeros(num_mazes, num_start_locations);

%% Get Median path lengths for humans
human_lengths = cellfun(@length, humans);
median_human = squeeze(median(human_lengths, 1));

human_sr_lengths = cellfun(@length, human_SR);
median_human_sr = squeeze(median(human_sr_lengths, 1));

human_sr_imp_lengths = cellfun(@length, human_SR_imp);
median_human_sr_imp = squeeze(median(human_sr_imp_lengths, 1));

%% Correlation between path lengths for human



%% Get Median path lengths for rats
rat_lengths = cellfun(@length, rat);
median_rat = squeeze(median(rat_lengths, 1));

rat_sr_lengths = cellfun(@length, rat_SR);
median_rat_sr = squeeze(median(rat_sr_lengths, 1));

rat_sr_imp_lengths = cellfun(@length, rat_SR_imp);
median_rat_sr_imp = squeeze(median(rat_sr_imp_lengths, 1));

%% Plot median trajectories across starting locations for humans
maze_index = 22;
cmap = brewermap(6,'Set1');

% Extract data for the specified maze
start_locations = 1:size(median_human, 2);

median_data = {
    median_human(maze_index, :),
    median_human_sr(maze_index, :),
    median_human_sr_imp(maze_index, :)
};

% Plot median path lengths
figure;
hold on;
for i = 1:3
    plot(start_locations, median_data{i}, 'Color', cmap(i,:), 'LineWidth', 2);
end
hold off;
   
title(sprintf('Maze %d (Humans)', maze_index), 'FontSize', 18, 'FontWeight','normal');
xlabel('Starting Location', 'FontSize', 16);
ylabel('Path Length', 'FontSize', 16);
legend('Human', 'SR', 'SR-IS', 'Location', 'best', 'FontSize', 12);
grid on;

% set(gca, 'LineWidth', 2, 'FontSize', 16);
set(gcf, 'color', 'w');

ax = gca;
set(gca, 'FontName', 'Times New Roman')
ax.YAxis.FontSize = 16;
ax.XAxis.FontSize = 16;
ax.Title.FontSize = 18;
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 8 3]);
exportgraphics(gcf, [save_dir,'human_maze22.pdf'], 'Resolution', 300);

%% Plot median trajectories across starting locations for rats
maze_index = 22;
cmap = brewermap(6,'Set1');

% Extract data for the specified maze
start_locations = 1:size(median_rat, 2);

median_data = {
    median_rat(maze_index, :),
    median_rat_sr(maze_index, :),
    median_rat_sr_imp(maze_index, :)
};

% Plot median path lengths
figure;
hold on;
for i = 1:3
    plot(start_locations, median_data{i}, 'Color', cmap(i,:), 'LineWidth', 2);
end
hold off;

title(sprintf('Maze %d (Rats)', maze_index), 'FontSize', 18, 'FontWeight', 'normal');
xlabel('Starting Location', 'FontSize', 16);
legend('Rat', 'SR', 'SR-IS', 'Location', 'best', 'FontSize', 12);
grid on;

% set(gca, 'LineWidth', 2, 'FontSize', 16);
set(gcf, 'color', 'w');

ax = gca;
set(gca, 'FontName', 'Times New Roman')
ax.YAxis.FontSize = 16;
ax.XAxis.FontSize = 16;
ax.Title.FontSize = 18;
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 8 3]);
exportgraphics(gcf, [save_dir,'rat_maze22.pdf'], 'Resolution', 300);