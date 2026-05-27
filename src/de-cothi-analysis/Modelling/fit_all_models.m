function sim_ll_r = fit_all_models(sim_dat)
% FIT_ALL_MODELS  Fit all 5 candidate models to a single participant's

A = []; b = []; Aeq = []; beq = []; nonlcon = [];

fit_opts_MF  = optimset('Display','off','MaxFunEvals',100,'MaxIter',100);
fit_opts_MB  = optimset('Display','off','MaxFunEvals',10, 'MaxIter',10);
fit_opts_SR  = optimset('Display','off','MaxFunEvals',100,'MaxIter',100);
fit_opts_Hyb = optimset('Display','off','MaxFunEvals',20, 'MaxIter',20);
fit_opts_IS  = optimset('Display','off','MaxFunEvals',100,'MaxIter',100);

x0_MF  = [0.5, 0.5];        lb_MF  = [0,0];     ub_MF  = [1,1];
x0_MB  = 0.5;               lb_MB  = 0;          ub_MB  = 1;
x0_SR  = [0.5, 1.0];        lb_SR  = [0,0];      ub_SR  = [1,10];
x0_Hyb = [0.5, 0.5, 0.5];   lb_Hyb = [0,0,0];   ub_Hyb = [1,1,1];
x0_IS  = [0.5, 0.5, 0.5];   lb_IS  = [0,0,0];   ub_IS  = [1,1,100];

sim_ll_r = zeros(5, 1);

fprintf('    Fitting MF...');     t0 = tic;
[~, ll] = fmincon(@(x) -Q_lik(x, sim_dat), ...
    x0_MF, A,b,Aeq,beq, lb_MF, ub_MF, nonlcon, fit_opts_MF);
sim_ll_r(1) = -ll;   fprintf(' %.1fs\n', toc(t0));

fprintf('    Fitting MB...');     t0 = tic;
[~, ll] = fmincon(@(x) -MB_lik(x, sim_dat), ...
    x0_MB, A,b,Aeq,beq, lb_MB, ub_MB, nonlcon, fit_opts_MB);
sim_ll_r(2) = -ll;   fprintf(' %.1fs\n', toc(t0));

fprintf('    Fitting SR...');     t0 = tic;
[~, ll] = fmincon(@(x) -SR_lik(x, sim_dat), ...
    x0_SR, A,b,Aeq,beq, lb_SR, ub_SR, nonlcon, fit_opts_SR);
sim_ll_r(3) = -ll;   fprintf(' %.1fs\n', toc(t0));

fprintf('    Fitting Hybrid...');  t0 = tic;
[~, ll] = fmincon(@(x) -Hybrid_lik_V(x, sim_dat), ...
    x0_Hyb, A,b,Aeq,beq, lb_Hyb, ub_Hyb, nonlcon, fit_opts_Hyb);
sim_ll_r(4) = -ll;   fprintf(' %.1fs\n', toc(t0));

fprintf('    Fitting SR-IS...');   t0 = tic;
try
    [~, ll] = fmincon(@(x) -SR_IS_lik_stable(x, sim_dat), ...
        x0_IS, A,b,Aeq,beq, lb_IS, ub_IS, nonlcon, fit_opts_IS);
    sim_ll_r(5) = -ll;
catch
    sim_ll_r(5) = -1e8;
end
fprintf(' %.1fs\n', toc(t0));

end
