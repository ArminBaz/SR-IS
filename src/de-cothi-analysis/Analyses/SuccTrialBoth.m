%% Save dir
save_dir = '/Users/abizzle/Research/LinearRL-TD/figures/';

%% Humans
load('Humans/humans.mat')
load('Humans/MB.mat')
load('Humans/Q.mat')
load('Humans/SR_imp_alpha0.6_gamma0.5.mat')
load('Humans/SR.mat')

cmap = brewermap(6,'Set1');
n_ppt = 18;
%% succ by trial
figure
hold on
x = (1:10)';

% Humans
y = squeeze(nanmean(human_succ,[1,2]));
CI = [y'-squeeze(nanstd(human_succ,[],[1,2]))'/sqrt(n_ppt); y'+squeeze(nanstd(human_succ,[],[1,2]))'/sqrt(n_ppt)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(3,:),'linewidth',2)

% Model Free
y = squeeze(nanmean(Q_succ,[1,2]));
CI = [y'-squeeze(nanstd(Q_succ,[],[1,2]))'/sqrt(n_ppt*100); y'+squeeze(nanstd(Q_succ,[],[1,2]))'/sqrt(n_ppt*100)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(5,:),'linewidth',2)

% Model Based
y = squeeze(nanmean(MB_succ,[1,2]));
CI = [y'-squeeze(nanstd(MB_succ,[],[1,2]))'/sqrt(n_ppt*100); y'+squeeze(nanstd(MB_succ,[],[1,2]))'/sqrt(n_ppt*100)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(2,:),'linewidth',2)

% SR
y = squeeze(nanmean(SR_succ,[1,2]));
CI = [y'-squeeze(nanstd(SR_succ,[],[1,2]))'/sqrt(n_ppt*100); y'+squeeze(nanstd(SR_succ,[],[1,2]))'/sqrt(n_ppt*100)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(4,:),'linewidth',2)

% SR Importance Sampling
y = squeeze(nanmean(SR_imp_succ,[1,2]));
CI = [y'-squeeze(nanstd(SR_imp_succ,[],[1,2]))'/sqrt(n_ppt*100); y'+squeeze(nanstd(SR_imp_succ,[],[1,2]))'/sqrt(n_ppt*100)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(1,:),'linewidth',2)

set(gca, 'FontName', 'Times New Roman')
set(gca,'FontSize',22)
title('Humans', 'FontSize', 24, 'FontWeight','normal', 'FontName', 'Times New Roman');
ylabel('Proportion goal reached')
xlabel('Trial #')
leg = legend({'','Human','','MF','','MB','','SR','','SR-IS'});
set(leg, 'FontSize', 16)
ylim([0,1])
xlim([0.98,10])
yticks([0,0.25,0.5,0.75,1])
set(gca,'LineWidth',2)
set(gcf,'color','w');

exportgraphics(gcf, [save_dir,'human_succ_trial.png'], 'Resolution', 300);

%% Rats
load('Rats/rat.mat')
load('Rats/MB.mat')
load('Rats/Q.mat')
load('Rats/SR_imp_alpha0.6_gamma0.5.mat')
load('Rats/SR.mat')

cmap = brewermap(6,'Set1');
n_ppt = 9;
%% succ by trial
figure
hold on
x = (1:10)';

% Rats
y = squeeze(nanmean(rat_succ,[1,2]));
CI = [y'-squeeze(nanstd(rat_succ,[],[1,2]))'/sqrt(n_ppt); y'+squeeze(nanstd(rat_succ,[],[1,2]))'/sqrt(n_ppt)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(3,:),'linewidth',2)

% Model Free
y = squeeze(nanmean(Q_succ,[1,2]));
CI = [y'-squeeze(nanstd(Q_succ,[],[1,2]))'/sqrt(n_ppt*100); y'+squeeze(nanstd(Q_succ,[],[1,2]))'/sqrt(n_ppt*100)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(5,:),'linewidth',2)

% Model Based
y = squeeze(nanmean(MB_succ,[1,2]));
CI = [y'-squeeze(nanstd(MB_succ,[],[1,2]))'/sqrt(n_ppt*100); y'+squeeze(nanstd(MB_succ,[],[1,2]))'/sqrt(n_ppt*100)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(2,:),'linewidth',2)

% SR
y = squeeze(nanmean(SR_succ,[1,2]));
CI = [y'-squeeze(nanstd(SR_succ,[],[1,2]))'/sqrt(n_ppt*100); y'+squeeze(nanstd(SR_succ,[],[1,2]))'/sqrt(n_ppt*100)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(4,:),'linewidth',2)

% SR Importance Sampling
y = squeeze(nanmean(SR_imp_succ,[1,2]));
CI = [y'-squeeze(nanstd(SR_imp_succ,[],[1,2]))'/sqrt(n_ppt*100); y'+squeeze(nanstd(SR_imp_succ,[],[1,2]))'/sqrt(n_ppt*100)];
fill([x;flipud(x)],[CI(1,:),fliplr(CI(2,:))],[.9 .9 .9],'linestyle','none');
plot(x,y,'color',cmap(1,:),'linewidth',2)

set(gca, 'FontName', 'Times New Roman')
set(gca,'FontSize',22)
title('Rats', 'FontSize', 24, 'FontWeight','normal', 'FontName', 'Times New Roman');
ylabel('Proportion goal reached')
xlabel('Trial #')
leg = legend({'','Rat','','MF','','MB','','SR','','SR-IS'});
set(leg, 'FontSize', 16)
ylim([0,1])
xlim([0.98,10])
yticks([0,0.25,0.5,0.75,1])
set(gca,'LineWidth',2)
set(gcf,'color','w');

exportgraphics(gcf, [save_dir,'rat_succ_trial.png'], 'Resolution', 300);