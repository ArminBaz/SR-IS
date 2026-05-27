% setup_carc_jobs.m
%
% Generates CARC job files. Creates a jobs/ subdirectory containing one .m file per participant.

% Add path to genbatch (copy genbatch.m from how2CARC into Modelling/ first)
n_humans = 18;
n_rats   = 9;

inputs = [];
for model_idx = 1:5
    for species_idx = 1:2
        n_ppt = n_humans * (species_idx == 1) + n_rats * (species_idx == 2);
        for r = 1:n_ppt
            inputs = [inputs; model_idx, species_idx, r];
        end
    end
end

fprintf('Generating %d job files...\n', size(inputs, 1));
genbatch(inputs, 'run_recovery_job');
fprintf('Done. Job files written to jobs/\n');
fprintf('Now copy jobs/ and submit_jobs to CARC and run:\n');
fprintf('  chmod 755 submit_jobs && ./submit_jobs job*\n');
