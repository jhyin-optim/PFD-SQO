function out = PFD_SQP05(n1,n2,model,para)
% 
% Consider the following problem which is a modification of hs118-.test:
%
%   min  f(x)+g(y) 
%   s.t. Ax+By=b,
%        Ex+Fy≤d,
%        u1≤Cx≤v1, p1≤Dy≤q1,
%        u2≤ x≤v2, p2≤ y≤q2,
%
% This kind of problem can be solved by PR-SQP-DS; see the paper:
%
% Jian J.B., Zhang C., Yin J.H., A Peaceman-Rachford splitting sequential
% quadratic programming double step-length method for two-block nonconvex
% optimization (in Chinese). Sci Sin Math
%
% Copyright (c), Jun, 2020, Jianghua Yin, Dept. Math., 
% Guangxi University for Nationalities

global A B b C D u v p q 

tic

us = length(u);
u1 = u(1:2*(n1-1));
u1n=size(u1);
u0=zeros(u1n);
u2 = u(2*(n1-1)+1:us);
vs = length(v);
if us~=vs
    disp('Inconsistent input data, please check them!');
end
v1 = v(1:2*(n1-1));
v1n=size(v1);
v0=zeros(v1n);
v2 = v(2*(n1-1)+1:vs);
ps = length(p);
p1 = p(1:n1-1);
p1n=size(p1);
p0=zeros(p1n);
p2 = p(n1:ps); 
qs = length(q);
if ps~=qs
    disp('Inconsistent input data, please check them!');
end
q1 = q(1:n1-1);
q2 = q(n1:2*n1-1); %普通箱子与带矩阵系数箱子分开
q1n=size(q1);
q0=zeros(q1n);
% As=A(5:n1,:);
% As = sparse(As);
% 
% B0=speye(n2);
% Bs=[B0(5:n2,:) -speye(n2-4)];
% Bs = sparse(Bs);
% bs=b(5:n1,:);
 

D1=D(:,1:n1);
E=-A;
F=-speye(n1); 
b1=-b;
% save matrices and matrix-vector multiplication
% AtA = As'*As;
% BtB = Bs'*Bs;
% AtB = As'*Bs;
% BtA = Bs'*As;
% Atb = As'*bs;
% Btb = Bs'*bs;
%C0=zeros(size(D,1),2*n1);
%D0=zeros(size(C,1),n1);
C_big = [C;-C;E];   
D_big = [D1;-D1;F];
%  C_big = [C;-C;E];   
%  D_big = [;D1;-D1;F];
%  C_big1 = [C;-C;C0;C0;E];   
%  D_big1 = [D1;-D1;D0;D0;F];
%c1=[v1;-u1;q0;p0;b1];
%d1=[v0;u0;q1;-p1;b1];
d1=[v1;-u1;q1;-p1;b1];
% c1 = [v1;-u1];
% d1 = [q1;-p1];

% set parameters
% line search parameters
rho = para.rho; 
sigma = para.sigma;  %控制参数

% penalty parameter
%beta = para.beta; 
% dual direction coefficient
%s = 0.001;   

% % control the positive definiteness of Hxk+beta*A'*A or Hyk+beta*B'*B
% delta = para.delta;
etax = para.etax;
etay = para.etay;

% termination
tol = para.tol;  %精度
maxiter = para.maxiter;   %最大迭代

x0 =  para.x0;
% x0 = [10*ones(n1,1);50*ones(n1,1);zeros(n1-1,1)];
y0 =  para.y0;
% rela_accur8_old = norm([x0;y0]);
 num_iterations=para.num_iterations;
% iter_re=zeros(num_iterations, 1);
iteration_points = zeros(num_iterations, 1);
x_iter= [];%zeros(num_iterations, 1);
ter=0;
Itr=0;
M=100;
for k=1:maxiter
    % update matrices only for the nonconvex case,i.e., model=2.
    if model == 1    %n1=5
    % initialization for the approximation of Hessian
    Hx0 = Hessf(x0,n1,model);  % the Hessian of f
    Hy0 = Hessg(y0,n2,model);  % the Hessian of g
    barHx0 = Hx0;
    barHy0 = Hy0+etay*eye(n2);
    else
        Hx0 = Hessf(x0,n1,model);  
        Hy0 = Hessg(y0,n2,model);  
        mineigHx = min(min(Hx0));
        if mineigHx > etax
            barHx0 = Hx0;
        elseif abs(mineigHx) <= etax
            barHx0 = Hx0+(etax-mineigHx)*speye(2*n1);
        else
            barHx0 = Hx0+2*abs(mineigHx)*speye(2*n1);
        end
        mineigHy = min(min(Hy0));
        if mineigHy > etay
            barHy0 = Hy0;
        elseif abs(mineigHy) <= etay
            barHy0 = Hy0+(etay-mineigHy)*speye(n2);
        else
            barHy0 = Hy0+2*abs(mineigHy)*speye(n2);
        end
    end
    % ==================
    %     x-subprolem , y-subprolem 
    % ==================
    nablaLx0 = Gradf(x0,n1,model);
    nablaLy0 = Gradg(y0,n2,model);
    true_f = nablaLx0-barHx0*x0;
    true_g = nablaLy0-barHy0*y0;
    c=0.5;
    hk=E*x0+F*y0-b1;
    c_big=[v1;-u1;0.5*c*hk-F*y0+b1];
    d_big=[q1;-p1;0.5*c*hk-E*x0+b1];
    if n1==5
        [barxk0 ,~, ~, ~, lambdax] = quadprog(barHx0,true_f,C_big,c_big,[],[],u2,v2,x0);
         mukx=lambdax.ineqlin(4*n1-3:5*n1-4,:);
        %mukx=lambdax.ineqlin(6*n1-5:7*n1-6,:);
        [baryk0 ,~, ~, ~, lambday] = quadprog(barHy0,true_g,D_big,d_big,[],[],p2,q2,y0);
        muky=lambday.ineqlin(2*n1-1:3*n1-2);
        %muky=lambday.ineqlin(6*n1-5:7*n1-6,:);
    else
         [barxk0 ,~, ~, ~, lambdax] = cplexqp(barHx0,true_f,C_big,c_big,[],[],u2,v2,x0);
        mukx=lambdax.ineqlin(4*n1-3:5*n1-4,:);
        %mukx=lambdax.ineqlin(6*n1-5:7*n1-6,:);
         [baryk0 ,~, ~, ~, lambday] = cplexqp(barHy0,true_g,D_big,d_big,[],[],p2,q2,y0);
          muky=lambday.ineqlin(2*n1-1:3*n1-2);
        %muky=lambday.ineqlin(6*n1-5:7*n1-6,:);
    end
    %判断分裂是否有效
    if k<2
        m1=100;
    else
    m1= min(7000,5*M);
    M=m1;
    end
   dK=[barxk0-x0;baryk0-y0];
    %m2=1;
    mu=norm(mukx-muky);
      Itr=Itr+1;
%    youxiao=mu-m1*(norm(dkx)^(1/3)+norm(dky)^(1/3));
    if  mu<m1*(norm(dK)^1.01)
        ter=ter+1;
        barxk= barxk0;
        baryk= baryk0;
        tmax=2/3;
    else 
        xy0=[x0;y0];
        barHxy0=blkdiag(barHx0,barHy0);
        true_fg = [nablaLx0-barHx0*x0; nablaLy0-barHy0*y0]; 
        E1=[E F];            
        C1=[C zeros(2*(n1-1),n1)];
        C2=[-C zeros(2*(n1-1),n1)];
        D2=[zeros(n1-1,2*n1) D1];
        D3=[zeros(n1-1,2*n1) -D1];
        E_big=[E1;C1;C2;D2;D3];
        up=[u2; p2];
        vq=[v2; q2];
        d2=[b1;v1;-u1;q1;-p1];
        if n1==5
        barxyk =quadprog(barHxy0,true_fg',E_big,d2,[],[],up,vq,xy0);
        else
        barxyk =cplexqp(barHxy0,true_fg',E_big,d2,[],[],up,vq,xy0);
        end
        barxk=barxyk(1:2*n1,:);
        baryk=barxyk(2*n1+1:3*n1,:);  
        tmax=1;
    end
        dkx = barxk-x0;
        dky = baryk-y0;
    % start Armijo line search
    Lkxy = fval(x0,n1,y0,n2,model); %L_beta(x0,y0,lam0)   % fval(x0,n1,y0,n2,model)=f(x0)+theta(y0)
    t = tmax;
    x_trail = x0+t*dkx;
    y_trail = y0+t*dky;
    Lkxy_trail = fval(x_trail,n1,y_trail,n2,model);  %%L_beta(x_trail,y0,lam0)
    dkxtbarHx0dkx = dkx'*barHx0*dkx;
    dkytbarHy0dky = dky'*barHy0*dky;
    while (Lkxy_trail > Lkxy-rho*t*(dkxtbarHx0dkx+dkytbarHy0dky)) && (t>10^-10)
        t = t*sigma;
        x_trail = x0+t*dkx;
        y_trail = y0+t*dky;
        Lkxy_trail = fval(x_trail,n1,y_trail,n2,model);
    end
    x1 = x_trail;
    y1 = y_trail;    
    % ==================
    %     lambda-update 
    % ==================
  %  lam1 = Lam(x1,y1,lam0,s);
    
    out.objfval(k) = fval(x1,n1,y1,n2,model);
    % termination test
    
   dk = [x1-x0;y1-y0];
     x_old = [x0;y0];
     x_new = [x1;y1];
%   rela_accur8 = (norm(dk))/(norm(x_old)+1);
    rela_accur8 = norm(dk);
     iteration_points(k,:) = rela_accur8;%存储
 
 %   iter_re(k,:)=rela_accur8/rela_accur8_old;
   % rela_accur8_old=rela_accur8; 
%     % case 1
%     if (norm(x1-x0)+norm(y1-y0)+norm(lam1-lam0))/(norm(x0)+norm(y0)+norm(lam0)+1) < tol
%         break;
%     end
%     % case 2
%     if (abs_accur< tol || rela_accur1<tol) 
%     if rela_accur < tol
% %         out.Tcpu = toc; 
%         break;        
%     end
%     % case 3
%     if abs_accur < tol
% %         out.Tcpu = toc; 
%         break;        
%     end
    % case 4
    if rela_accur8 < tol
         x_f = [x1;y1];
%         out.Tcpu = toc; 
        break;        
    end
    
    
    % Updating iteration
     x_iter(:,1) = x_old;
     x_iter(:,k+1) = x_new;%存储
    x0 = x1;
    y0 = y1;
 %   lam0 = lam1;
    
end  
[~,n] = size(x_iter);
X=repmat(x_f, 1,n);
 X_iter= x_iter-X;
 norms_X = zeros(1, n-2);
 for i = 1:n-2
    norms_X(i) = norm(X_iter(:, i+2))/norm(X_iter(:, i+1));
end
out.norms_X=norms_X;
% figure;
% plot(1:num_iterations, iteration_points(:, 1), '-o', 'LineWidth', 1, 'MarkerSize', 3);
% hold on;
% title('Iteration Points vs. Iteration Count');
% xlabel('Iteration Count');
% ylabel('Iteration Points');
% legend({'Variable 1', 'Variable 2'}, 'Location', 'best');
% grid on;
%out.iteration_points= iteration_points;
%out.iter_re=iter_re;
out.iteration_points= iteration_points(1:Itr,:);
out.tern=ter;
out.Tcpu = toc;
out.x = x1;
out.y = y1;
out.phieq = 0;
%norm(min(As*x1+Bs*y1-bs,0),inf);
%out.phiiq = max(norm(max(C_big*x1+D_big*y1-d1,0),inf));
% out.nequ = norm(A*x1+B*y1-b,inf);
% ninequC = norm(max(C_big*x1-c_big,0),inf)
% ninequD = norm(max(D_big*y1-d_big,0),inf)
% X = [x1;-x1];
% V = [v2;-u2];
% boundn = norm(max(X-V,0),inf)
%out.beta = beta;
% out.objfval = fval(x1,n1,y1,n2,model);

%% ------------------SUBFUNCTION-----------------------------
% function out=Hessf(x,n1,model)
% % out = zeros(2*n1);
% % for i=1:n1
% %     out(i,i) = 0.0002;
% % end
% % for i=(n1+1):(2*n1)
% %     out(i,i) = 0.0002;
% % end
% if model == 1
%     out = 0.0002*eye(2*n1);
% else
%     out = zeros(2*n1);
%     for i=1:n1
%         out(i,i) = 0.0002-0.003*x(i);
%     end
%     for i=(n1+1):(2*n1)
%         out(i,i) = 0.0002+0.0048*x(i);
%     end
% end
% out = sparse(out);
% 
% function out=Hessg(y,n2,model)
% out = zeros(2*n2);
% if model == 1
%     for i=1:n2
%         out(i,i) = 0.0003; 
%     end
% else
%     for i=1:n2
%         out(i,i) = 0.0003+0.006*y(i); 
%     end
% end
% out = sparse(out);
% 
% function out = Gradf(x,n1,model)
% out = zeros(2*n1,1);
% if model == 1
%     for i=1:n1
%         out(i) = 2.3+0.0002*x(i);
%     end
%     for i=(n1+1):(2*n1)
%         out(i) = 1.7+0.0002*x(i);
%     end
% else
%     for i=1:n1
%         out(i) = 2.3+0.0002*x(i)-0.0015*x(i)^2;
%     end
%     for i=(n1+1):(2*n1)
%         out(i) = 1.7+0.0002*x(i)+0.0024*x(i)^2;
%     end 
% end
% 
% function out = Gradg(y,n2,model)
% out = zeros(2*n2,1);
% if model == 1
%     for i=1:n2
%         out(i) = 2.2+0.0003*y(i);
%     end
% else
%     for i=1:n2
%         out(i) = 2.2+0.0003*y(i)+0.003*y(i)^2;
%     end
% end
% 
% function out = fval(x,n1,y,n2,model)
% out = 0;
% if model ==1
%     for i=1:n1
%         out = out+2.3*x(i)+0.0001*x(i)^2;
%     end
%     for j=(n1+1):(2*n1)
%         out = out+1.7*x(j)+0.0001*x(j)^2;
%     end
%     for k=1:n2
%         out = out+2.2*y(k)+0.00015*y(k)^2;
%     end
% else
%     for i=1:n1
%         out = out+2.3*x(i)+0.0001*x(i)^2-0.0005*x(i)^3;
%     end
%     for j=(n1+1):(2*n1)
%         out = out+1.7*x(j)+0.0001*x(j)^2+0.0008*x(j)^3;
%     end
%     for k=1:n2
%         out = out+2.2*y(k)+0.00015*y(k)^2+0.001*y(k)^3;
%     end
% end



function out=Hessf(x,n1,model)
% out = zeros(2*n1);
% for i=1:n1
%     out(i,i) = 0.0002;
% end
% for i=(n1+1):(2*n1)
%     out(i,i) = 0.0002;
% end
if model == 1
    out = 0.0002*eye(2*n1);
else
    out = zeros(2*n1);
    for i=1:n1
        out(i,i) = 0.0002-0.003*x(i)+exp(sin(x(i)))*((cos(x(i)))^2-sin(x(i)));
    end
    for i=(n1+1):(2*n1)
        out(i,i) = 0.0002+0.0048*x(i)+exp(cos(x(i)))*((sin(x(i)))^2-cos(x(i)));
    end
end
out = sparse(out);

function out=Hessg(y,n2,model)
out = zeros(n2);
if model == 1
    for i=1:n2
        out(i,i) = 0.0003; 
    end
else
    for i=1:n2
        out(i,i) = 0.0003+0.006*y(i)+exp(cos(y(i)))*((sin(y(i)))^2-cos(y(i))); 
    end
end
out = sparse(out);

function out = Gradf(x,n1,model)
out = zeros(2*n1,1);
if model == 1
    for i=1:n1
        out(i) = 2.3+0.0002*x(i);
    end
    for i=(n1+1):(2*n1)
        out(i) = 1.7+0.0002*x(i);
    end
else
    for i=1:n1
        out(i) = 2.3+0.0002*x(i)-0.0015*x(i)^2+exp(sin(x(i)))*cos(x(i));
    end
    for i=(n1+1):(2*n1)
        out(i) = 1.7+0.0002*x(i)+0.0024*x(i)^2-exp(cos(x(i)))*sin(x(i));
    end 
end

function out = Gradg(y,n2,model)
out = zeros(n2,1);
if model == 1
    for i=1:n2
        out(i) = 2.2+0.0003*y(i);
    end
else
    for i=1:n2
        out(i) = 2.2+0.0003*y(i)+0.003*y(i)^2-exp(cos(y(i)))*sin(y(i));
    end
end

function out = fval(x,n1,y,n2,model)
out = 0;
if model ==1
    for i=1:n1
        out = out+2.3*x(i)+0.0001*x(i)^2;
    end
    for j=(n1+1):(2*n1)
        out = out+1.7*x(j)+0.0001*x(j)^2;
    end
    for k=1:n2
        out = out+2.2*y(k)+0.00015*y(k)^2;
    end
else
    for i=1:n1
        out = out+2.3*x(i)+0.0001*x(i)^2-0.0005*x(i)^3+exp(sin(x(i)));
    end
    for j=(n1+1):(2*n1)
        out = out+1.7*x(j)+0.0001*x(j)^2+0.0008*x(j)^3+exp(cos(x(j)));
    end
    for k=1:n2
        out = out+2.2*y(k)+0.00015*y(k)^2+0.001*y(k)^3+exp(cos(y(k)));
    end
end
