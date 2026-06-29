% collect_sampled_results.m
% Run locally from Modelling/ after all CARC sampled jobs finish.
% Reads checkpoints_sampled/ and prints a confusion matrix of BIC winners.

n_sims       = 25;
n_models     = 5;
n_params_vec = [2, 1, 2, 3, 3];
n_datapoints = 25 * 10;
model_names  = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};
species_names = {'human', 'rat'};
gen_tags     = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};

for sp = 1:2
    species = species_names{sp};
    fprintf('\n========== %s ==========\n', upper(species));

    C = zeros(n_models, n_models);   % C(gen, winner)
    missing = 0;

    for gen_idx = 1:n_models
        for sim_idx = 1:n_sims
            ckpt = sprintf('checkpoints_sampled/%s_%s_sim%d.mat', ...
                gen_tags{gen_idx}, species, sim_idx);
            if exist(ckpt, 'file')
                tmp = load(ckpt, 'sim_ll_r');
                bic = -2*tmp.sim_ll_r + n_params_vec' .* log(n_datapoints);
                [~, winner] = min(bic);
                C(gen_idx, winner) = C(gen_idx, winner) + 1;
            else
                fprintf('  MISSING: %s %s sim%d\n', gen_tags{gen_idx}, species, sim_idx);
                missing = missing + 1;
            end
        end
    end

    fprintf('\nConfusion matrix (rows=simulated, cols=fitted winner):\n');
    fprintf('%-8s', '');
    for m = 1:n_models; fprintf('%-10s', model_names{m}); end
    fprintf('\n');
    for gen_idx = 1:n_models
        fprintf('%-8s', model_names{gen_idx});
        for m = 1:n_models
            pct = 100 * C(gen_idx, m) / max(sum(C(gen_idx,:)), 1);
            fprintf('%-10s', sprintf('%d (%.0f%%)', C(gen_idx,m), pct));
        end
        fprintf('\n');
    end
    fprintf('Missing checkpoints: %d / %d\n', missing, n_models * n_sims);
end
