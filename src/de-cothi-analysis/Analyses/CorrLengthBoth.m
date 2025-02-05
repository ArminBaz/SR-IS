%% Save dir & maze
save_dir = '/Users/abizzle/Research/LinearRL-TD/figures/';
maze = 22;
 
%% Load paths
% Humans
load('Humans/humans.mat')
load('Humans/SR_imp_alpha0.6_gamma0.5.mat')
% load('Humans/SR_imp_test.mat')
load('Humans/SR.mat')

human_SR = SR;
human_SR_imp = SR_imp;

% Rats
load('Rats/rat.mat')
load('Rats/SR_imp_alpha0.6_gamma0.5.mat')
load('Rats/SR.mat')

rat_SR = SR;
rat_SR_imp = SR_imp;

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

ppt_rhos = zeros(n_ppt,2);
for i = 1:n_ppt
    ppt_rhos(i,1) = corr(squeeze(median(human_sr_lengths(ppt_ids(i,:)==1,maze,:),1)), squeeze(median(human_lengths(i,maze,:),1)));
    ppt_rhos(i,2) = corr(squeeze(median(human_sr_imp_lengths(ppt_ids(i,:)==1,maze,:),1)), squeeze(median(human_lengths(i,maze,:),1)));
end

rhos = nanmean(ppt_rhos,1);
rhos_err = nanstd(ppt_rhos,[],1)/sqrt(n_ppt);

%% Human plotting
figure
hold on
bar(rhos,'LineWidth',2,'EdgeColor','k','FaceColor','w')
errorbar(rhos, rhos_err, '.k', 'LineWidth', 2)

% Set axes properties
set(gca,'LineWidth',2)
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 18)  % Set base font size

% Set x-axis ticks and labels
xticks([1,2])
xticklabels({'SR', 'SR-IS'})
set(gca, 'XTickLabel', get(gca, 'XTickLabel'), 'FontSize', 18)

% Set title and labels
title('Humans Maze 22', 'FontSize', 20, 'FontWeight','normal', 'FontName', 'Times New Roman');
ylabel('Correlation coefficient', 'FontSize', 18, 'FontName', 'Times New Roman');

% Final touches
set(gcf,'color','w');
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 5 4]);

% Save human correlation
exportgraphics(gcf, [save_dir,'length_corr_human_maze22.png'], 'Resolution', 300);

%% Rat correlation
n_ppt = 9;
ppt_ids = zeros(n_ppt,n_ppt*100);

for i = 1:n_ppt
    tmp_ppt_id = zeros(1,n_ppt);
    tmp_ppt_id(i) = 1;
    ppt_ids(i,:) = reshape(repmat(tmp_ppt_id,100,1),1,[]);
end

ppt_rhos = zeros(n_ppt,2);
for i = 1:n_ppt
    ppt_rhos(i,1) = corr(squeeze(median(rat_sr_lengths(ppt_ids(i,:)==1,maze,:),1)), squeeze(median(rat_lengths(i,maze,:),1)));
    ppt_rhos(i,2) = corr(squeeze(median(rat_sr_imp_lengths(ppt_ids(i,:)==1,maze,:),1)), squeeze(median(rat_lengths(i,maze,:),1)));
end

rhos = nanmean(ppt_rhos,1);
rhos_err = nanstd(ppt_rhos,[],1)/sqrt(n_ppt);

%% Rat plotting
figure
hold on
bar(rhos,'LineWidth',2,'EdgeColor','k','FaceColor','w')
errorbar(rhos, rhos_err, '.k', 'LineWidth', 2)

% Set axes properties
set(gca,'LineWidth',2)
set(gca, 'FontName', 'Times New Roman')
set(gca, 'FontSize', 18)  % Set base font size

% Set x-axis ticks and labels
xticks([1,2])
xticklabels({'SR', 'SR-IS'})
set(gca, 'XTickLabel', get(gca, 'XTickLabel'), 'FontSize', 18)

% Set title and labels
title('Rats Maze 22', 'FontSize', 20, 'FontWeight','normal', 'FontName', 'Times New Roman');
ylabel('Correlation coefficient', 'FontSize', 18, 'FontName', 'Times New Roman');

% Final touches
set(gcf,'color','w');
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 5 4]);

% Save rat correlation
exportgraphics(gcf, [save_dir,'length_corr_rat_maze22.png'], 'Resolution', 300);