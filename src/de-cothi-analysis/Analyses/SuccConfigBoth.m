%% Create the figure
figure('Position', [100, 100, 1000, 600])
% Save dir
save_dir = '/Users/abizzle/Research/LinearRL-TD/figures/';

%% Create subplots for humans and rats
%% Humans
load('Humans/humans.mat')
load('Humans/MB.mat')
load('Humans/Q.mat')
load('Humans/SR_imp_alpha0.6_gamma0.5.mat')
% load('Humans/SR_imp.mat')
load('Humans/SR.mat')

subplot(2,1,1)
imagesc([squeeze(nanmean(human_succ,[1,3])); squeeze(nanmean(Q_succ,[1,3])); squeeze(nanmean(MB_succ,[1,3])); squeeze(nanmean(SR_succ,[1,3])); squeeze(nanmean(SR_imp_succ,[1,3]))])
text(0.5, 1.2, 'Humans', 'FontSize', 18, 'FontWeight', 'normal', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontName', 'Times New Roman')
yticks(1:5)
yticklabels({'Human','MF','MB','SR', 'SR-IS'})
xticks([1, 5, 10, 15, 20, 25]);
xticklabels({'1','5','10','15','20','25'})
set(gca, 'TickLength', [0 0])  % Remove tick marks
set(gca,'LineWidth',2)
set(gca,'FontSize',14)
set(gca, 'FontName', 'Times New Roman')
ax = gca;
ax.YAxis.FontSize = 16;
ax.XAxis.FontSize = 16;

%% Rats
load('Rats/rat.mat')
load('Rats/MB.mat')
load('Rats/Q.mat')
load('Rats/SR_imp_alpha0.6_gamma0.5.mat')
% load('Rats/SR_imp.mat')
load('Rats/SR.mat')

subplot(2,1,2)
imagesc([squeeze(nanmean(rat_succ,[1,3])); squeeze(nanmean(Q_succ,[1,3])); squeeze(nanmean(MB_succ,[1,3])); squeeze(nanmean(SR_succ,[1,3])); squeeze(nanmean(SR_imp_succ,[1,3]))])
text(0.5, 1.2, 'Rats', 'FontSize', 18, 'FontWeight', 'normal', 'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontName', 'Times New Roman')
yticks(1:5)
yticklabels({'Rat','MF','MB','SR', 'SR-IS'})
xticks([1, 5, 10, 15, 20, 25]);
xticklabels({'1','5','10','15','20','25'})
xlabel('Configuration')
set(gca, 'TickLength', [0 0])  % Remove tick marks
set(gca,'LineWidth',2)
set(gca,'FontSize',14)
set(gca, 'FontName', 'Times New Roman')
ax = gca;
ax.YAxis.FontSize = 16;
ax.XAxis.FontSize = 16;

% Adjust colormap and aspect ratio for both subplots
colormap(jet)
caxis([0, 1])  % Set color axis limits
for i = 1:2
    subplot(2,1,i)
    pbaspect([25,4,1])
end

% Dim the colors
brighten(-0.1)  % Adjust this value to make colors more or less dim

% Add colorbar  
cb = colorbar('Position', [0.93 0.1 0.02 0.8]);
cb.Ticks = [0 1];
cb.TickLabels = {'0', '1'};
cb.FontSize = 14;
cb.FontName = 'Times New Roman';
cb.Label.String = 'Proportional Goal Reached';
cb.Label.Rotation = 270;
cb.Label.Position = [3, 0.5, 0];
cb.Label.FontSize = 14;

% Adjust overall figure appearance
set(gcf,'color','w');
set(gcf, 'PaperPositionMode', 'auto');

% Save
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [0 0 8 4]);
exportgraphics(gcf, [save_dir,'succ_both.png'], 'Resolution', 300);