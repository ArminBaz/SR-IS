% collect_recovery_results.m
%
% Run this locally from the Modelling/ directory after all CARC jobs finish.
% Reads checkpoints/ and computes BIC winners for each generating model.

n_humans     = 18;
n_rats       = 9;
n_models     = 5;
n_params_vec = [2, 1, 2, 3, 3];
n_datapoints = 25 * 10;
model_names  = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};

gen_model_tags = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};

for gen_idx = 1:5
    gen_name = gen_model_tags{gen_idx};
    fprintf('\n========== %s MODEL RECOVERY ==========\n', gen_name);

    for sp = 1:2
        if sp == 1; species = 'human'; n_ppt = n_humans;
        else;        species = 'rat';   n_ppt = n_rats;  end

        sim_ll = zeros(n_models, n_ppt);
        missing = 0;

        for r = 1:n_ppt
            ckpt = sprintf('checkpoints/%s_%s_%d.mat', gen_name, species, r);
            if exist(ckpt, 'file')
                tmp = load(ckpt, 'sim_ll_r');
                sim_ll(:, r) = tmp.sim_ll_r;
            else
                fprintf('  MISSING: %s %s %d\n', gen_name, species, r);
                missing = missing + 1;
                sim_ll(:, r) = NaN;
            end
        end

        BIC = -2*sim_ll + n_params_vec' .* log(n_datapoints);
        [~, best] = min(BIC, [], 1);

        fprintf('\n--- %s recovery: %ss (%d missing) ---\n', gen_name, species, missing);
        counts = histcounts(best(~isnan(best(1,:))), 0.5:(n_models+0.5));
        for m = 1:n_models
            fprintf('  %-8s: %2d / %2d  (%.0f%%)\n', ...
                model_names{m}, counts(m), n_ppt - missing, ...
                100 * counts(m) / max(n_ppt - missing, 1));
        end

        % Save results
        out_file = sprintf('%s_model_recovery_%ss.mat', gen_name, species);
        save(out_file, 'sim_ll', 'BIC', 'best');
    end
end

fprintf('\nDone. Results saved to *_model_recovery_*.mat\n');
