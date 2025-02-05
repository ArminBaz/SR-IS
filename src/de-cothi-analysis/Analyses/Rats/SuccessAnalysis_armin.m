load('rat.mat')
load('MB.mat')
load('Q.mat')
load('SR_imp_alpha0.6_gamma0.5.mat')
load('SR.mat')

cmap = brewermap(6,'Set1');
n_ppt = 9;
%% succ by trial
figure
hold on
x = (1:10)';

% Humans
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

ylabel('Proportion goal reached')
xlabel('Trial #')
legend({'','Rat','','MF','','MB','','SR','','Imp Samp'})
ylim([0,1])
xlim([0.98,10])
yticks([0,0.25,0.5,0.75,1])
set(gca,'LineWidth',2)
set(gcf,'color','w');
set(gca,'FontSize',16)

%% succ by config 
figure
imagesc([squeeze(nanmean(rat_succ,[1,3])); squeeze(nanmean(Q_succ,[1,3])); squeeze(nanmean(MB_succ,[1,3])); squeeze(nanmean(SR_succ,[1,3])); squeeze(nanmean(SR_imp_succ,[1,3]))])
colormap jet
pbaspect([25,4,1])
yticks(1:5)
yticklabels({'Rat','MF','MB','SR', 'Imp Samp'})
xticks([])
set(gca,'LineWidth',2)
set(gcf,'color','w');
set(gca,'FontSize',18)

%% config correlations
ppt_ids = zeros(n_ppt,n_ppt*100);

for i = 1:n_ppt
    tmp_ppt_id = zeros(1,n_ppt);
    tmp_ppt_id(i) = 1;
    ppt_ids(i,:) = reshape(repmat(tmp_ppt_id,100,1),1,[]);
end

ppt_rhos = zeros(n_ppt,4);
for i = 1:n_ppt
    ppt_rhos(i,1) = corr(squeeze(nanmean(Q_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(rat_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,2) = corr(squeeze(nanmean(MB_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(rat_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,3) = corr(squeeze(nanmean(SR_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(rat_succ(i,:,:),[1,3]))','Type','Spearman');
    ppt_rhos(i,4) = corr(squeeze(nanmean(SR_imp_succ(ppt_ids(i,:)==1,:,:),[1,3]))', squeeze(nanmean(rat_succ(i,:,:),[1,3]))','Type','Spearman');
end

rhos = nanmean(ppt_rhos,1);
rhos_err = nanstd(ppt_rhos,[],1)/sqrt(n_ppt);

figure
hold on
bar(rhos,'LineWidth',2,'EdgeColor','k','FaceColor','w')
xticks([1,2,3, 4])
xticklabels({'MF','MB','SR', 'Imp Samp'})
errorbar(rhos, rhos_err, '.k', 'LineWidth', 2)
% errorbar([1,2,3,4], rhos, rhos-rhos_err(1,:), rhos-rhos_err(2,:), '.k', 'LineWidth', 2)
% for i = 1:n_ppt
%     plot((1:4)+ (-0.05+rand(1,3)/10), ppt_rhos(i,:),'k.-','MarkerSize',12,'linewidth',0.5)
% end
set(gca,'LineWidth',2)
set(gcf,'color','w');
set(gca,'FontSize',18)