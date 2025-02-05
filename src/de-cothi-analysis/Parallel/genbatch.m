function genbatch(inputs, mfile_name, jobdir, bias_num)
%     genbatch(FNAME,INPUTS,BIAS,JOBDIR)
% INPUTS is a matrix. Each row of this matrix corresponds to an input jobs.
% mfile_name is the name of an m-file. The default is "runjob".
% example
% genbatch((1:20)'); % this creates 20 jobs that call runjob with inputs 1,
% 2, ....
% -------------------------------------------------------------------------

%%% check inputs
if nargin<1, error('Number of inputs in %s is less than 1!'); end
if nargin<2
    mfile_name = 'runjob';
end
if nargin<3
    jobdir = fullfile(pwd, 'jobs');
end
if nargin<4, bias_num = 0; end

ftxt = get_job_text(mfile_name);
ffname = 'job';
fext = '.m';
currdir = pwd;

%%% output directory
if(~exist(jobdir,'dir')), mkdir(jobdir); end
cd(jobdir);

%%% for each row of inputs, write a separate file
ind = 1:size(inputs,1);
for i=1:size(inputs,1)
    jobid = sprintf('%s_temp%02d%04d',ffname,bias_num,ind(i));
    
    fnametarget = sprintf('%s%s',jobid,fext);
    fid = fopen(fnametarget,'w');
    
%     ftxtvar = sprintf('function %s\n',jobid);
%     fprintf(fid,'%s', ftxtvar);

    ftxtvar = sprintf('jobID=''%s'';\n',jobid);
    fprintf(fid,'%s', ftxtvar);
    
    ftxtvar = sprintf('%s=%s;\n','input',mat2str(inputs(i,:)));
    fprintf(fid,'%s', ftxtvar);
    
    fprintf(fid,'%s', ftxt);
    fclose(fid);
end

% get back
cd(currdir);
end


function str = get_job_text(mfile_name)

i=0;
txt{i+1} = "fprintf('input is %s\n',mat2str(input));"; i=i+1;
txt{i+1} = "cd(fullfile(pwd,'..'));"; i=i+1;
txt{i+1} = "try"; i=i+1;
txt{i+1} = "    "; i=i+1;
txt{i+1} = "    %------%------%------"; i=i+1;
txt{i+1} = sprintf("    %s(input);", mfile_name); i=i+1;
txt{i+1} = "    "; i=i+1;
txt{i+1} = "    %------%------%------"; i=i+1;
txt{i+1} = "    "; i=i+1;
txt{i+1} = "catch message"; i=i+1;
txt{i+1} = "    disp(message.message);"; i=i+1;
txt{i+1} = "end";


str = '';
for i=1:length(txt)
    str=  sprintf('%s\n%s', str, txt{i});
end

end