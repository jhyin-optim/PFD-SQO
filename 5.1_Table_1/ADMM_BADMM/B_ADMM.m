function out = B_ADMM(n1,n2,model,para)
% 
% Consider the following problem which is a modification of hs118-.test:
%
%   min  f(x)+g(y) 
%   s.t. Ax+By=b£¬
%        u1¡ÜCx¡Üv1, p1¡ÜDy¡Üq1,
%        u2¡Ü x¡Üv2, p2¡Ü y¡Üq2.
% This kind of problem can be solved by MS-SQO; see the paper:
% 
% Jian J.B., Zhang C., Yin J.H., Yang L.F., Ma G.D., Monotone
% splitting sequential quadratic optimization algorithm with applications
% in electric power systems. J. Optim. Theory Appl., 2020, 186: 226-247
% 
% Copyright (c), Jun, 2020, Jianghua Yin, Dept. Math., 
% Guangxi University for Nationalities

global A B b C D u v p q 

tic

us = length(u);
u1 = u(1:2*(n1-1));
u2 = u(2*(n1-1)+1:us);
vs = length(v);
if us~=vs
    disp('Inconsistent input data, please check them!');
end
v1 = v(1:2*(n1-1));
v2 = v(2*(n1-1)+1:vs);
ps = length(p);
p1 = p(1:n1-1);
p2 = p(n1:ps);
qs = length(q);
if ps~=qs
    disp('Inconsistent input data, please check them!');
end
q1 = q(1:n1-1);
q2 = q(n1:qs);
lb_x = [u2;p1];
ub_x = [v2;q1];
lb_y = [p2;u1];
ub_y = [q2;v1];
[m1,m2] = size(A);
A_big = [A zeros(m1,n1-1);C zeros(2*n1-2,n1-1);zeros(n1-1,m2) -eye(n1-1)];
A_big = sparse(A_big);
[m3,m4] = size(B);
B_big = [B zeros(m3,2*n1-2);zeros(2*n1-2,m4) -eye(2*n1-2);D zeros(n1-1,2*n1-2)];
B_big = sparse(B_big);
b_big = [b;zeros(3*n1-3,1)];
b_big = sparse(b_big);

beta = para.beta; 

xi = para.xi;

tol = para.tol; 
maxiter = para.maxiter;

objvL_beta = @(x,y,lambda,mu) -lambda'*(A_big*x+B_big*y-b_big)+ ...
    mu/2*sum((A_big*x+B_big*y-b_big).^2);
dlam = @(x,y) xi*(A_big*x+B_big*y-b_big);
x0 = [20*ones(n1,1);55;60*ones(n1-1,1);zeros(n1-1,1)];
y0 = [15;20*ones(n1-1,1);zeros(3*n1-2,1)];
lam0 = 1.6*ones(size(b_big));

for k=1:maxiter
    
 % ==================
 %     x-subprolem 
 % ==================
 % define x function handles
 
funx = @(x) fval(x,n1,y0,n2,model)+objvL_beta(x,y0,lam0,beta)+0.5*(norm(x-x0)).^2;

[ x1,fvalx]= fmincon(funx,x0,[],[],[],[],lb_x,ub_x,[],[]);
 % ==================
 %     y-subprolem 
 % ==================
% define y function handles
funy = @(y) fval(x1,n1,y,n2,model)+objvL_beta( x1,y,lam0,beta)+0.5*(norm(y-y0)).^2;

[y1,fvaly]= fmincon(funy,y0,[],[],[],[],lb_y,ub_y,[],[]);
% ==================
%     lambda-update 
% ==================
    dklam = dlam( x1,y1);
    lam1 = lam0-beta*dklam;
    out.objfval(k) = fval(x1,n1,y1,n2,model);
    dk = [x1-x0;y1-y0;lam1-lam0];
    x_old = [x0;y0;lam0];
    rela_accur8 = (norm(dk))/(norm(x_old)+1);
    if rela_accur8 < tol
%         out.Tcpu = toc; 
        break;        
    end
    % Updating iteration
    x0 = x1;
    y0 = y1;
    lam0 = lam1;
    
end
out.rela =  rela_accur8;
out.Tcpu = toc;
out.x =  x1;
out.y =  y1;
out.phieq = norm(A*x1(1:2*n1)+B*y1(1:2*n2)-b,inf);    % norm(A_big*x1+B_big*y1-b_big,inf);
out.phiiq = max([norm(min(C*x1(1:2*n1)-u1,0),inf);norm(max(C*x1(1:2*n1)-v1,0),inf); ...
    norm(min(D*y1(1:2*n2)-p1,0),inf);norm(max(D*y1(1:2*n2)-q1,0),inf)]);
out.beta = beta;
%out.objfval = fval(x1,n1,y1,n2,model);

%% ------------------SUBFUNCTION-----------------------------
% function out=Hessf(x,n1,model)
% % out = zeros(2*n1);
% % for i=1:n1
% %     out(i,i) = 0.0002;
% % end
% % for i=(n1+1):(2*n1)
% %     out(i,i) = 0.0002;
% % end
% out = zeros(3*n1-1);
% if model == 1
%     for i=1:(2*n1)
%         out(i,i) = 0.0002;
%     end
% else
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
% out = zeros(4*n2-2);
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
% out = zeros(3*n1-1,1);
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
% out = zeros(4*n2-2,1);
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
out = zeros(3*n1-1);
if model == 1
    for i=1:(2*n1)
        out(i,i) = 0.0002;
    end
else
    for i=1:n1
        out(i,i) = 0.0002-0.003*x(i)+exp(sin(x(i)))*((cos(x(i)))^2-sin(x(i)));
    end
    for i=(n1+1):(2*n1)
        out(i,i) = 0.0002+0.0048*x(i)+exp(cos(x(i)))*((sin(x(i)))^2-cos(x(i)));
    end
end
out = sparse(out);

function out=Hessg(y,n2,model)
out = zeros(4*n2-2);
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
out = zeros(3*n1-1,1);
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
out = zeros(4*n2-2,1);
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