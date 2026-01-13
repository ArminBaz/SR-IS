%% Save dir & maze
save_dir = '/Users/abizzle/Research/LinearRL-TD/figures/';
mazes = 1:25;  % Iterate through all 25 mazes

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

%% Set up figure
figure('Position', [50 50 1500 1000])  % Adjust size as needed
n_rows = 5;  % 5 rows
n_cols = 10;  % 2 plots (human/rat) × 5 columns

%% Process all mazes
for maze_idx = 1:length(mazes)
    maze = mazes(maze_idx);
    
    %% Get path lengths for humans
    human_lengths = cellfun(@length, humans);
    human_sr_lengths = cellfun(@length, human_SR);
    human_sr_imp_lengths = cellfun(@length, human_SR_imp);
    
    %% Get path lengths for rats
    rat_lengths = cellfun(@length, rat);
    rat_sr_lengths = cellfun(@length, rat_SR);
    rat_sr_imp_lengths = cellfun(@length, rat_SR_imp);
    
    %% Human correlation
    n_ppt = 18;
    ppt_ids = zeros(n_ppt,n_ppt*100);
    for i = 1:n_ppt
        tmp_ppt_id = zeros(1,n_ppt);
        tmp_ppt_id(i) = 1;
        ppt_ids(i,:) = reshape(repmat(tmp_ppt_id,100,1),1,[]);
    end
    
    human_ppt_rhos = zeros(n_ppt,2);
    for i = 1:n_ppt
        human_ppt_rhos(i,1) = corr(squeeze(median(human_sr_lengths(ppt_ids(i,:)==1,maze,:),1)), squeeze(median(human_lengths(i,maze,:),1)));
        human_ppt_rhos(i,2) = corr(squeeze(median(human_sr_imp_lengths(ppt_ids(i,:)==1,maze,:),1)), squeeze(median(human_lengths(i,maze,:),1)));
    end
    human_rhos = nanmean(human_ppt_rhos,1);
    human_rhos_err = nanstd(human_ppt_rhos,[],1)/sqrt(n_ppt);
    
    %% Rat correlation
    n_ppt = 9;
    ppt_ids = zeros(n_ppt,n_ppt*100);
    for i = 1:n_ppt
        tmp_ppt_id = zeros(1,n_ppt);
        tmp_ppt_id(i) = 1;
        ppt_ids(i,:) = reshape(repmat(tmp_ppt_id,100,1),1,[]);
    end
    
    rat_ppt_rhos = zeros(n_ppt,2);
    for i = 1:n_ppt
        rat_ppt_rhos(i,1) = corr(squeeze(median(rat_sr_lengths(ppt_ids(i,:)==1,maze,:),1)), squeeze(median(rat_lengths(i,maze,:),1)));
        rat_ppt_rhos(i,2) = corr(squeeze(median(rat_sr_imp_lengths(ppt_ids(i,:)==1,maze,:),1)), squeeze(median(rat_lengths(i,maze,:),1)));
    end
    rat_rhos = nanmean(rat_ppt_rhos,1);
    rat_rhos_err = nanstd(rat_ppt_rhos,[],1)/sqrt(n_ppt);
    
    %% Plotting
    % Human subplot
    subplot(n_rows, n_cols, 2*maze_idx-1)
    hold on
    bar(human_rhos,'LineWidth',1,'EdgeColor','k','FaceColor','w')
    errorbar(human_rhos, human_rhos_err, '.k', 'LineWidth', 1)
    % Set axes properties
    set(gca,'LineWidth',1)
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 8)
    xticks([1,2])
    xticklabels({'SR', 'SR-IS'})
    title(['Humans Maze ' num2str(maze)], 'FontSize', 10, 'FontWeight','normal', 'FontName', 'Times New Roman');
    if mod(maze_idx, 5) == 1  % Only add ylabel for leftmost plots
        ylabel('Correlation coefficient', 'FontSize', 8, 'FontName', 'Times New Roman');
    end
    
    % Rat subplot
    subplot(n_rows, n_cols, 2*maze_idx)
    hold on
    bar(rat_rhos,'LineWidth',1,'EdgeColor','k','FaceColor','w')
    errorbar(rat_rhos, rat_rhos_err, '.k', 'LineWidth', 1)
    % Set axes properties
    set(gca,'LineWidth',1)
    set(gca, 'FontName', 'Times New Roman')
    set(gca, 'FontSize', 8)
    xticks([1,2])
    xticklabels({'SR', 'SR-IS'})
    title(['Rats Maze ' num2str(maze)], 'FontSize', 10, 'FontWeight','normal', 'FontName', 'Times New Roman');
end

%% Final touches
set(gcf,'color','w');
% Save all correlations
% exportgraphics(gcf, [save_dir,'length_corr_all_mazes.png'], 'Resolution', 300);