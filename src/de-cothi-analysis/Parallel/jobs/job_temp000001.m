jobID='job_temp000001';
input=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18];

fprintf('input is %s\n',mat2str(input));
cd(fullfile(pwd,'..'));
try
    
    %------%------%------
    runjob(input);
    
    %------%------%------
    
catch message
    disp(message.message);
end