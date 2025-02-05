function runjob(input)
    input = 1;
    % Load data
    load('humans.mat')
    load('SR_imp.mat')
    load('mazes.mat')
    load('ppt_ids.mat')

    % Process a single participant
    ppt = input;
    human_pd_sim_SR_imp = cell(1, 25, 10);

    agent_ids = find(ppt_ids(ppt,:));

    for config = 1:25
        map = 1*(mazes{config} == -1);
        for trial = 1:10
            reference_traj = humans{ppt,config,trial};
            SR_imp_pd_sim = zeros(100,length(reference_traj));
            parfor i = 1:length(agent_ids)
                ag_id = agent_ids(i);
                % Use SR_imp directly without nested indexing
                % SR_imp_pd_sim(i,:) = pd_similarity(reference_traj, SR_imp{ag_id,config,trial}, map);
            end
            human_pd_sim_SR_imp{1,config,trial} = SR_imp_pd_sim;
        end
    end

    % Save results for this participant
    save(sprintf('/out/results_ppt_%02d.mat', ppt), 'human_pd_sim_SR_imp');
end