% use PR_SQP_DS and MS-SQO to solve the following problem which is a modification of hs118-.test:
%
%   min  f(x)+g(y) 
%   s.t. Ax+By=b，
%        u1≤Cx≤v1, p1≤Dy≤q1,
%        u2≤ x≤v2, p2≤ y≤q2
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
%clear all
close all
format long

global A B b C D u v p q 

problem_set =[50];% 50 100 200 300 400 500 600 700 800 900 1000];

N = length(problem_set);
fid = fopen('mytext.txt','w');
for k=1:N
    n1 = problem_set(k);
    n2 = n1;
    % problem data
    A = [speye(n1) speye(n1)];
    B =  speye(n2);
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
    C = [C0 zeros(n1-1,n1);zeros(n1-1,n1) C0]; %2*(n1-1)行2*n1列
    C = sparse(C);
    C1 = [C0 zeros(n1-1,n1) zeros(n1-1,n1);zeros(n1-1,n1) C0 zeros(n1-1,n1);zeros(n1-1,n1) zeros(n1-1,n1) C0]; %2*(n1-1)行2*n1列
    C1 = sparse(C1);
    D = [C0 zeros(n1-1,n1)];   %(n1-1)行2*n1列
    D = sparse(D);
    u1 = -7*ones(2*(n1-1),1);  %2*(n1-1)行1列
    u2 = zeros(2*n1,1);   %2*n1行1列 
    u2(1) = 8;
    u2(n1+1) = 43;
    u = [u1;u2];  %4*n1-2行1列
    v1 = [6*ones(n1-1,1);7*ones(n1-1,1)]; %2*(n1-1)行1列
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
    v = [v1;v2];  %4*n1-2行1列
    p1 = -7*ones(n1-1,1);      
    p2 = [3;zeros(n1-1,1)];        
    p = [p1;p2];
    q1 = 6*ones(n1-1,1);       
    if n1==5
        q2 = [16;60*ones(4,1);inf*ones(5,1)];
    else
        q2 = zeros(n1,1);
        q2(1:5) = [16;60*ones(4,1)];
        for i=6:n1
            q2(i) = 60+i;
        end
    end
    q = [q1;q2];
    M=[-A,-B;C1;-C1];
    b0=[-b;v1;-p1;q1;-u1];
    lb=[u2;p2];
    ub=[v2;q2];
    f=ones(3*n1,1);
    x = linprog(f,M,b0,[],[],lb,ub);
    para1.x0 = x(1:2*n1);
    para1.y0 = x(2*n1+1:3*n1);
    
    x_optim = [8;1;1;3;5;49;56;63;70;77];
    y_optim = [3;0;6;12;18];

    if n1==5
        model = 1; % a convex programming
    else
        model = 2; % a nonconvex programming
    end

    fprintf(' ********** Comparison starts **********\n');
    
    % ********** start PFD-SQP0 **********
    
    % line search parameters
    para1.rho = 0.45;  
    para1.sigma = 0.9; 
    
    if n1==5
        para1.beta = 16;   
    else
        para1.beta = 61;  
    end
    % dual direction coefficients satisfy r+s>0 & r>=0
    para1.r = 0.002;   
    para1.s = 0.001;   
    % estimate the positive definiteness of Hxk+beta*A'*A or
    % Hyk+beta*B'*B
    para1.etax = 0.001;
    para1.etay = 0.001;
%     termination
    para1.tol = 10^-8;  
    para1.maxiter = 10000;
    para1.num_iterations=140;

     fprintf(' ********** run PF-SQP **********\n');
     out0 = PF_SQP(n1,n2,model,para1);
     Itr0 = length(out0.objfval);
     Tcpu0 = out0.Tcpu; 
     F_PF_SQP = out0.objfval(end); 
     
    fprintf(' ********** run PFD-SQP0 **********\n');
    out1 = PFD_SQP0(n1,n2,model,para1);
    Itr1 = length(out1.objfval);
    tern1=out1.tern;
    Tcpu1 = out1.Tcpu;
    bi1=round(tern1*100/Itr1) ;
    F_PFD_SQP0 = out1.objfval(end); 
    RE_F_0=(F_PFD_SQP0-F_PF_SQP)*100/F_PF_SQP;
    RE_C_0=(Tcpu1-Tcpu0)*100/Tcpu0;
    iteration_points_0=out1.iteration_points;
    norms_X_0=out1.norms_X;
%    iter_re_0=out1.iter_re;
   % iteration_points_0=out1.iteration_points;
%     phiiq1 = out1.phiiq; 
%    beta1 = out1.beta;
%     fprintf('Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', Itr1,Tcpu1,F_PFD_SQP,phieq1,phiiq1);
%      fprintf('Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', Itr1,Tcpu1,F_PFD_SQP0,phieq1,phiiq1);
%     fprintf('beta=%d, r=%.3f, s=%.3f,tol=%.1e\n', beta1,para1.r,para1.s,para1.tol);
%     % the following data only use to plot
   
%     if n1==5
%         x_stop2 = out5.x;
%         y_stop2 = out5.y(1:n1+1,:);
    
 fprintf(' ********** run PFD-SQP05 **********\n');
     
    out2 = PFD_SQP05(n1,n2,model,para1);
    Itr2 = length(out2.objfval);
    tern2=out2.tern;
    Tcpu2 = out2.Tcpu;
    bi2=round(tern2*100/Itr2);
    F_PFD_SQP05 = out2.objfval(end);
     RE_F_05=(F_PFD_SQP05-F_PF_SQP)*100/F_PF_SQP;
    RE_C_05=(Tcpu2-Tcpu0)*100/Tcpu0;
    norms_X_05=out2.norms_X;
     iteration_points_05=out2.iteration_points;
 %   iter_re_05=out2.iter_re;

 %   beta2= out2.beta;
%     fprintf('Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', Itr1,Tcpu1,F_PFD_SQP,phieq1,phiiq1);
%      fprintf('Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', Itr2,Tcpu2,F_PFD_SQP5,phieq2,phiiq2);
%     fprintf('beta=%d, r=%.3f, s=%.3f,tol=%.1e\n', beta1,para1.r,para1.s,para1.tol);
%     % the following data only use to plot
%     if n1==5
%         x_stop1 = out1.x;
%         y_stop1 = out1.y;
%     end
% fprintf(' ********** run PR-SQP75 **********\n');
%     out3 = PFD_SQP75(n1,n2,model,para1);
%     Itr3 = length(out3.objfval);
%     tern3=out3.tern;
%     Tcpu3 = out3.Tcpu;
%     F_PFD_SQP75 = out3.objfval(end);
%     beta3= out3.beta;
% %     fprintf('Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', Itr1,Tcpu1,F_PFD_SQP,phieq1,phiiq1);
%       fprintf('Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', Itr3,Tcpu3,F_PFD_SQP75,phieq3,phiiq3);
% %     fprintf('beta=%d, r=%.3f, s=%.3f,tol=%.1e\n', beta1,para1.r,para1.s,para1.tol);
% %     % the following data only use to plot
%     if n1==5
%         x_stop1 = out1.x;
%         y_stop1 = out1.y;
%     end
    fprintf(' ********** run PFD-SQP1 **********\n');
    out3 = PFD_SQP1(n1,n2,model,para1);
    Itr3 = length(out3.objfval);
    tern3=out3.tern;
    Tcpu3 = out3.Tcpu;
    bi3=round(tern3*100/Itr3);
    F_PFD_SQP1 = out3.objfval(end);
     RE_F_1=(F_PFD_SQP1-F_PF_SQP)*100/F_PF_SQP;
    RE_C_1=(Tcpu3-Tcpu0)*100/Tcpu0;
    norms_X_1=out3.norms_X;
    iteration_points_1=out3.iteration_points;
 %   iter_re_1=out3.iter_re;
 %  iteration_points_1=out3.iteration_points;
%      if n1==5
%         x_stop1 = out3.x;
%         y_stop1 = out3.y;
%     end

%    beta4= out4.beta;
%     fprintf('Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', Itr1,Tcpu1,F_PFD_SQP,phieq1,phiiq1);
%      fprintf('tren=%d, Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n',tern4, Itr4,Tcpu4,F_PFD_SQP,phieq4,phiiq4);
%     fprintf('beta=%d, r=%.3f, s=%.3f,tol=%.1e\n', beta1,para1.r,para1.s,para1.tol);
%     % the following data only use to plot
%     if n1==5
%         x_stop1 = out4.x;
%         y_stop1 = out4.y;
%     end

%     % ********** start MS-SQO **********
%     
%     para2.rho = 0.6;  
%     para2.sigma = 0.7;
%     para2.xi = 0.001;
%     para2.beta = 72;
%     para2.etax = 0.0001;
%     para2.etay = 0.0001;
% %     termination
%     para2.tol = 10^-5;  
%     para2.maxiter = 10000;
%     
%     out5 = MS_SQO(n1,n2,model,para2);
%     
%     Itr5 = length(out5.objfval);
%     Tcpu5 = out5.Tcpu;
%     F_MS_SQO = out5.objfval(end);
%     phieq5 = out5.phieq;     
%     phiiq5 = out5.phiiq;    
%     if n1==5
%         x_stop2 = out5.x;
%         y_stop2 = out5.y(1:n1,:);
%     end

%  
    fprintf('%.2f &  %.4f&', Tcpu0, F_PF_SQP);
    
    fprintf('%.2f&%d/%d=%d &  %.2f& %.2f&  %.2f&', Tcpu1, tern1, Itr1,bi1,F_PFD_SQP0,RE_F_0,RE_C_0);
    
    fprintf(' %.2f& %d/%d=%d & %.2f& %.2f& ',Tcpu2,  tern2, Itr2,bi2, F_PFD_SQP05,RE_F_05);
%     
      fprintf(' %.2f& %d/%d=%d & %.2f& %.2f& ',Tcpu3,  tern3, Itr3,bi3, F_PFD_SQP1,RE_F_1);
      
 %   fprintf( '%d',iter_re_1);
     
     
%     fprintf('tren=%d, Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n',tern1, Itr1,Tcpu1,F_PFD_SQP0,phieq1,phiiq1);
%     fprintf('tren=%d, Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n',tern2, Itr2,Tcpu2,F_PFD_SQP5,phieq2,phiiq2);
%    fprintf('tren=%d, Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', tern3, Itr3,Tcpu3,F_PFD_SQP75,phieq3,phiiq3);
 %    fprintf('tren=%d, Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n',tern4, Itr4,Tcpu4,F_PFD_SQP,phieq4,phiiq4);
 %    fprintf('Itr=%d, Tcpu=%.3f, objfval=%.5f, nequc=%.6f, ninequc=%.6f\n', Itr5,Tcpu5,F_MS_SQO,phieq5,phiiq5);
%     fprintf(fid,'%d & %d & %d & %.2f & %.2f & %.2f & %.2f & %d & %.2f & %.2f & %.2f & %.2f\\\\\n', ...
%         n1, beta1, Itr1, Tcpu1, phieq1, phiiq1, F_PFD_SQP, ...
%         Itr2, Tcpu2, phieq2, phiiq2, F_MS_SQO);
%     
     if n1==50
      figure(1), plot(1:Itr3,iteration_points_1,'-o',1:Itr1,iteration_points_0,'-x',1:Itr2,iteration_points_05,'-*','LineWidth', 1, 'MarkerSize', 3);
      legend('PFD-SQOM_{1}', 'PFD-SQOM_{0}','PFD-SQOM_{0.5}');
      xlabel('iter(k)'); 
        ylabel('norm(u_{k+1}-u_{k})');

         figure(2), plot(1:Itr3-2,norms_X_1,'-o',1:Itr1-2,norms_X_0,'-x',1:Itr2-2,norms_X_05,'-*','LineWidth', 1, 'MarkerSize', 3);
        legend('PFD-SQOM_{1}', 'PFD-SQOM_{0}','PFD-SQOM_{0.5}');
          xlabel('iter(k)'); 
        ylabel('norm(u_{k+1}-u_{*})/norm(u_{k}-u_{*})');
     end
 %       h = title('c_0 (-x), c_{1/2} (-*),c_1 (-o)');   
%           figure(1), plot(1:para1.num_iterations,iter_re_0,'-x','LineWidth', 1, 'MarkerSize', 3);
%           figure(2), plot(1:para1.num_iterations,iter_re_05,'-*','LineWidth', 1, 'MarkerSize', 3);
%           figure(3), plot(1:69,iter_re_1,'-o','LineWidth', 1, 'MarkerSize', 3);
         
%         h = title('x\_stop1 (+), x\_optim (o)');                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
%         set(h,'Interpreter','latex','fontsize',13)
% 
%         figure(2), plot(1:n1,y_stop1,'k+',1:n1,y_optim,'ro');
%         g = title('y\_stop1 (+), y\_optim (o)');
%         set(g,'Interpreter','latex','fontsize',13)
%     end
%     if model == 1
%         fprintf('%10s\t%10s\n', 'x_stop1', ...
%         'x_optim');
%         for i=1:length(x_stop1)
%             fprintf('%10.5f\t%10d\n', x_stop1(i), ...
%             x_optim(i) );
%         end
%         fprintf('%10s\t%10s\n', 'y_stop1', ...
%          'y_optim');
%         for i=1:length(y_stop1)
%             fprintf('%10.5f\t%10d\n', y_stop1(i), ...
%             y_optim(i)  );
%         end
%     end
% if n1==5
%         figure(1), plot(1:2*n1,x_stop1,'k+',1:2*n1,x_optim,'ro',1:2*n1,x_stop2(1:2*n1),'b*');
%         h = title('x\_stop1 (+), x\_optim (o), x\_stop2 (*)');                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
%         set(h,'Interpreter','latex','fontsize',13)
% 
%         figure(2), plot(1:n2,y_stop1,'k+',1:n2,y_optim,'ro',1:n2,y_stop2(1:n2),'b*');
%         g = title('y\_stop1 (+), y\_optim (o), y\_stop2 (*)');
%         set(g,'Interpreter','latex','fontsize',13)
% end
% 
%         if model == 1
%         fprintf('%10s\t%10s\t%10s\n', 'x_stop1', ...
%         'x_stop2','x_optim');
%         for i=1:length(x_stop1)
%             fprintf('%10.5f\t%10.5f\t%10d\n', x_stop1(i), ...
%            x_stop2(i),  x_optim(i) );
%         end
%         fprintf('%10s\t%10s\t%10s\n', 'y_stop1', ...
%         'y_stop2', 'y_optim');
%         for i=1:length(y_stop1)
%             fprintf('%10.5f\t%10.5f\t%10d\n', y_stop1(i), ...
%             y_stop2(i),y_optim(i)  );
%         end
%         end
% 
%     result = [x_stop1(1:2*n1) x_stop2(1:2*n1) x_optim ];
%     fid1=fopen('myresult1.txt','w');
%     for i=1:length(x_stop1)
%         fprintf(fid1,'%.2f & %.2f & %d\n', ...
%         result(i,1), result(i,2), result(i,3));
%     end
%    fclose(fid1);
%    result = [y_stop1  y_stop2  y_optim ];
%     fid2=fopen('myresult2.txt','w');
%     for i=1:length(y_stop1)
%         fprintf(fid2,'%.2f & %.2f & %d\n', ...
%         result(i,1), result(i,2), result(i,3));
%      end
%    fclose(fid1);

    
end
fclose(fid);





%     NI = [NI;Itr];
% end
% figure(1)
% plot(2:2:100,NI,'r*-','LineWidth', 2) 
% set(gca,'FontName','Times','FontSize',16)
% xlabel('\beta');ylabel('Itr');   
    
    
    