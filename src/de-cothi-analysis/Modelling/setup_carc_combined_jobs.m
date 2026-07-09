% setup_carc_combined_jobs.m
% Generates 100 job .m files (5 models x 20 sims) in jobs_combined/
% and a submit_combined_jobs shell script.
%
% On CARC, after running this script:
%   chmod 755 submit_combined_jobs
%   ./submit_combined_jobs jobs_combined/job*

n_sims      = 20;
model_names = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};

out_dir = 'jobs_combined';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

n_jobs = 0;
for model_idx = 1:5
    for sim_idx = 1:n_sims
        fname = sprintf('%s/job_%s_sim%02d.m', ...
            out_dir, model_names{model_idx}, sim_idx);
        fid = fopen(fname, 'w');
        fprintf(fid, 'addpath(genpath(pwd));\n');
        fprintf(fid, 'run_recovery_combined_job([%d, %d]);\n', ...
            model_idx, sim_idx);
        fclose(fid);
        n_jobs = n_jobs + 1;
    end
end

fprintf('Generated %d job files in %s/\n', n_jobs, out_dir);

% Write submit script
fid = fopen('submit_combined_jobs', 'w');
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

fprintf('Submit script written: submit_combined_jobs\n');
fprintf('\nTo submit all jobs on CARC:\n');
fprintf('  chmod 755 submit_combined_jobs\n');
fprintf('  ./submit_combined_jobs jobs_combined/job*\n');
