% use PR_SQP_DS and MS-SQO to solve the following problem which is a modification of hs118-.test:
%
%   min  f(x)+g(y) 
%   s.t. Ax+By=b£¬
%        u1¡ÜCx¡Üv1, p1¡ÜDy¡Üq1,
%        u2¡Ü x¡Üv2, p2¡Ü y¡Üq2
%
% Implementation and numerical experience are 
% described in the following papers: 
%
% [1] Jian J.B., Zhang C., Yin J.H., Yang L.F., Ma G.D., Monotone
% splitting sequential quadratic optimization algorithm with applications
% in electric power systems. J. Optim. Theory Appl., 2020, 186: 226-247
%
% [2] Jian J.B., Zhang C., Yin J.H., A Peaceman-Rachford splitting sequential
% quadratic programming double step-length method for two-block nonconvex
% optimization (in Chinese). Sci Sin Math
%
% For any problems, please contact Jianghua Yin (jianghuayin1017@126.com).
clc
clear all
close all
format long

global A B b C D u v p q 

problem_set = [5];% 30 50 70 90 100 200 300 400 500 600 700 800 900 1000];

N = length(problem_set);
fid = fopen('mytext.txt','w');
for k=1:N
    n1 = problem_set(k);   
    n2 = n1;
    % problem data
    A = [speye(n1) speye(n1)];
    B = [speye(n2) -speye(n2)];
    if n1==5
        b = [60;50;70;85;100];
    else
        b = zeros(n1,1);
        b(1:5) = [60;50;70;85;100];
        for i=6:n1
            b(i) = 100+5*(i-4);
        end
    end
    C0 = zeros(n1-1,n1);
    for i=1:(n1-1)
        C0(i,i) = -1;
        C0(i,i+1) = 1;
    end
    C = [C0 zeros(n1-1,n1);zeros(n1-1,n1) C0]; 
    C = sparse(C);
    D = [C0 zeros(n1-1,n1)];   
    D = sparse(D);
    u1 = -7*ones(2*(n1-1),1);  
    u2 = zeros(2*n1,1);        
    u2(1) = 8;
    u2(n1+1) = 43;
    u = [u1;u2];
    v1 = [6*ones(n1-1,1);7*ones(n1-1,1)]; 
    if n1==5
        v2 = [21;90*ones(n1-1,1);57;120*ones(n1-1,1)];  
    else
        v2 = zeros(2*n1,1);
        v2(1:5) = [21;90*ones(4,1)];
        v2(n1+1:n1+5) = [57;120*ones(4,1)];
        for i=6:n1
            v2(i) = 90+3*i;
        end
        for i=n1+6:2*n1
            v2(i) = 120+6*(i-n1);
        end
    end
    v = [v1;v2];
    p1 = -7*ones(n1-1,1);      
    p2 = [3;zeros(2*n1-1,1)];        
    p = [p1;p2];
    q1 = 6*ones(n1-1,1);       
    if n1==5
        q2 = [16;60*ones(4,1);inf*ones(5,1)];
    else
        q2 = zeros(2*n1,1);
        q2(1:5) = [16;60*ones(4,1)];
        q2(n1+1:2*n1) = inf*ones(n1,1);
        for i=6:n1
            q2(i) = 60+i;
        end
    end
    q = [q1;q2];
    
    x_optim = [8;1;1;3;5;49;56;63;70;77];
    y_optim = [3;0;6;12;18;0;7;0;0;0];

    if n1==5
        model = 1; % a convex programming
    else
        model = 2; % a nonconvex programming
    end

    fprintf(' ********** Comparison starts **********\n');
    
    fprintf(' ********** run ADMM **********\n');
    % ********** start ADMM **********

    para1.xi = 0.001;
    para1.beta = 72;
    para1.tol = 10^-3;  
    para1.maxiter = 3000;
    
    out1 = ADMM(n1,n2,model,para1);
    
    Itr1 = length(out1.objfval);
    Tcpu1 = out1.Tcpu;
    F_ADMM = out1.objfval(end);
    phieq1 = out1.phieq;     
    phiiq1 = out1.phiiq;    
    if n1==5
        x_stop1 = out1.x;
        y_stop1 = out1.y;
    end
    
     fprintf(' ********** run B_ADMM **********\n');
    % ********** start bADMM **********

    para2.xi = 0.001;
    para2.beta = 72;
    para2.tol = 10^-3;  
    para2.maxiter = 3000;
    
    out2 = B_ADMM(n1,n2,model,para2);
    
    Itr2 = length(out2.objfval);
    Tcpu2 = out2.Tcpu;
    F_B_ADMM = out2.objfval(end);
    phieq2 = out2.phieq;     
    phiiq2 = out2.phiiq;    
    if n1==5
        x_stop2 = out2.x;
        y_stop2 = out2.y;
    end

    fprintf(' %.2f & %.4f',  Tcpu1,F_ADMM);
    fprintf(' & %.2f & %.4f',  Tcpu2,F_B_ADMM);
    
end
fclose(fid);

    
    