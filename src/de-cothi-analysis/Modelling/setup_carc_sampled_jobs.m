% setup_carc_sampled_jobs.m
% Run from Modelling/ directory on CARC (or locally — generates files only).
%
% Generates 250 job .m files (5 models x 2 species x 25 sims) in jobs_sampled/
% and a submit_sampled_jobs shell script.
%
% On CARC, after running this script:
%   chmod 755 submit_sampled_jobs
%   ./submit_sampled_jobs jobs_sampled/job*

n_sims       = 25;
model_names  = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};
species_names = {'human', 'rat'};

out_dir = 'jobs_sampled';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

n_jobs = 0;
for model_idx = 1:5
    for species_idx = 1:2
        for sim_idx = 1:n_sims
            fname = sprintf('%s/job_%s_%s_sim%02d.m', ...
                out_dir, model_names{model_idx}, species_names{species_idx}, sim_idx);
            fid = fopen(fname, 'w');
            fprintf(fid, 'addpath(genpath(pwd));\n');
            fprintf(fid, 'run_recovery_sampled_job([%d, %d, %d]);\n', ...
                model_idx, species_idx, sim_idx);
            fclose(fid);
            n_jobs = n_jobs + 1;
        end
    end
end

fprintf('Generated %d job files in %s/\n', n_jobs, out_dir);

% Write submit script
fid = fopen('submit_sampled_jobs', 'w');
fprintf(fid, '#!/bin/sh\n');
fprintf(fid, 'scriptnames="$@"\n');
fprintf(fid, 'MATLABCMD="2025b"\n');
fprintf(fid, 'WALLTIME=01:30:00\n');
fprintf(fid, 'MEM=8gb\n');
fprintf(fid, '\n');
fprintf(fid, 'module purge\n');
fprintf(fid, 'module load matlab/${MATLABCMD}\n');
fprintf(fid, '\n');
fprintf(fid, 'for mfile in ${scriptnames}\n');
fprintf(fid, 'do\n');
fprintf(fid, 'sbatch -t $WALLTIME --mem=$MEM --wrap="matlab -nojvm -nodisplay -nodesktop < ${mfile}"\n');
fprintf(fid, 'sleep .05\n');
fprintf(fid, 'done\n');
fclose(fid);

fprintf('Submit script written: submit_sampled_jobs\n');
fprintf('\nTo submit all jobs on CARC:\n');
fprintf('  chmod 755 submit_sampled_jobs\n');
fprintf('  ./submit_sampled_jobs jobs_sampled/job*\n');
