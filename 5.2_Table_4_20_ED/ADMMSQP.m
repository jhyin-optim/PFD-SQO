% This code is  linear ADMM - SQP method for two-block separate minimization
% Refer to:  linear ADMM - SQP method for two-block separate minimization
% problem:f(x)+h(y)    s.t:Ax+By=b   
% Author:lao yi xian
% This code was written by lao yi xian
% Email: 741679896@qq.com
%========================================================================

function ADMMSQP
clear 
tic
%%%%%%%%%%%%��ȡx�ļ�����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 pathAndFilename='UC_AF/NS1_10_based_5_std.mod';
%pathAndFilename='UC_AF/NS2_15_based_5_std.mod';
 %pathAndFilename='UC_AF/NS3_20_based_5_std.mod';
% pathAndFilename='UC_AF/NS4_30_based_5_std.mod';
% pathAndFilename='UC_AF/NS5_40_based_5_std.mod';
%  pathAndFilename='UC_AF/NS6_50_based_5_std.mod';
% pathAndFilename='UC_AF/NS7_70_based_5_std.mod';
% pathAndFilename='UC_AF/NS8_80_based_5_std.mod';
% pathAndFilename='UC_AF/NS9_90_based_5_std.mod';
% pathAndFilename='UC_AF/NS10_100_based_5_std.mod';
% pathAndFilename='UC_AF/NS11_110_based_5_std.mod';
% pathAndFilename='UC_AF/NS12_120_based_5_std.mod';
% pathAndFilename='UC_AF/NS13_130_based_5_std.mod';
%  pathAndFilename='UC_AF/NS14_150_based_5_std.mod'; 
%  pathAndFilename='UC_AF/NS15_170_based_5_std.mod';
%  pathAndFilename='UC_AF/NS16_180_based_5_std.mod';
% pathAndFilename='UC_AF/NS17_200_based_5_std.mod';
% pathAndFilename='UC_AF/NS18_220_based_5_std.mod';
% pathAndFilename='UC_AF/NS19_240_based_5_std.mod';
 %pathAndFilename='UC_AF/NS20_250_based_5_std.mod';
 dataUC=readdataUC(pathAndFilename);
[~,qp ] = qpED( dataUC );
T=dataUC.T;
N=dataUC.N;
N1=ceil(dataUC.N/2);
N2=N-N1;
% z_wave=sparse(N*T,1);
% null1 = zeros(N2*T,N1*T);
% null2 = zeros(N1*T,N2*T);
% null3 = zeros(T,N*T);
% IG1 = [eye(T*N1),zeros(N1*T,N2*T)];
% IG2 = [zeros(N2*T,N1*T),eye(T*N2)];
c=qp.c_wan;
%c = [qp.c_wan;-qp.b1;-qp.b2];%%��ʽԼ���Ҷ���
% q01=qp.b1;
% q02=qp.b2;
E = qp.B1_wan;
% E = [qp.B1_wan;qp.A1;null1];%%xϵ������
F = qp.B2_wan;
%F = [qp.B2_wan;null2;qp.A2];%%yϵ������
M1=[qp.A1;-qp.A1];
U1=qp.b_up(1:N1*T);
U2=qp.b_up(N1*T+1:N*T);
D1=qp.b_down(1:N1*T);
D2=qp.b_down(N1*T+1:N*T);
UD1=[U1+qp.b1;D1+qp.b2];
UD2=[U2+qp.b2;D2+qp.b2];
M2=[qp.A2;-qp.A2];
%G=[null3;IG1;IG2];%%zϵ������
%C=[qp.A1;null1];%%z��������xϵ������
%D=[null2;qp.A2];%%z��������yϵ������
%IG=[IG1;IG2];%%z��������zϵ������
%d= [-qp.b1;-qp.b2];%%z���������Ҷ���
x_k = 1*qp.x_L+0*qp.x_U;%%%%%x�ĳ�ʼֵ
y_k = 1*qp.y_L+0*qp.y_U;%%%%%y�ĳ�ʼֵ
%z_k = 1*(-qp.b_down)+0*qp.b_up;%%%%%z�ĳ�ʼֵ
qp.x_juzhen=sparse(1:N1*T,1:N1*T,x_k);
qp.y_juzhen=sparse(1:N2*T,1:N2*T,y_k);
x_hat_k = x_k'.*x_k';
y_hat_k = y_k'.*y_k';
g_x = (3*qp.q1_UC*qp.x_juzhen+2*qp.C1_UC)*x_k+qp.K1_UC;%f(x)���ݶ�
g_y = (3*qp.q2_UC*qp.y_juzhen+2*qp.C2_UC)*y_k+qp.K2_UC;%theta(y)���ݶ�
H_x_k = qp.Q1_UC*qp.x_juzhen+2*qp.C1_UC;%f(x)��hessian��
H_y_k = qp.Q2_UC*qp.y_juzhen+2*qp.C2_UC;%theta(y)��hessian��

%%%%%��ʼ���㷨����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
betar =200;%������
theta = 0.49;%%�޸�
sigma = 0.8;  %%�޸�
epsilon =0.001;%% z����Ĳ���
max = 1000;%%�����ܴ���
TOL=0.005;%%��ֹ����
lamda = 1*ones(T,1);%�������ճ���
%%%%%%%%%%����Ŀ�꺯��ֵ%%%%%%%%%%%%%%%%%%%%%%%%%%%
niter = 1;
obj_f = x_hat_k*qp.q1_UC*x_k +x_k'*qp.C1_UC*x_k+qp.K1_UC'*x_k+sum(qp.d1_UC);%f(x)��ֵ
obj_theta = y_hat_k*qp.q2_UC*y_k +y_k'*qp.C2_UC*y_k+qp.K2_UC'*y_k+sum(qp.d2_UC);%theta(y)��ֵ
obj = obj_f+obj_theta;%f(x)+theta(y)��ֵ
%y_wave=y_k;


%W = betar*(B1'*B1);%����quadprog��Ҫ���ı�׼hesse��
       fprintf('Problem #:')
       fprintf('        niter        objective            runtime \n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%�㷨ѭ��%%%%%%%%%%%%%%%%%%%%%%%%%%
 while(niter <= max)
    %z_hat=d-(1/betar)*lamda_ie-C*x_k-D*y_k;
    H_x = H_x_k+betar*(E'*E);%x-QP ������hesse��    
    H_y = H_y_k+betar*(F'*F);%y-QP ������hesse�� 
   f_x=g_x-H_x_k*x_k+betar*E'*(F*y_k-c-(1/betar)*lamda);%x-QP������һ����ϵ��
   f_y=g_y-H_y_k*y_k+betar*F'*(E*x_k-c-(1/betar)*lamda);%y-QP������һ����ϵ��
%    %%%%% solve QP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     [x_wave] = quadprog(H_x,f_x',M1,UD1,[],[],qp.x_L,qp.x_U);%hesse�����ݶȣ��н�Լ��
     [y_wave] = quadprog(H_y,f_y',M2,UD2,[],[],qp.y_L,qp.y_U);
%       [x_wave, fval, exitflag, output] = cplexqp (H_x,f_x', [], [], [], [],qp.x_L,qp.x_U,[]);
%       [y_wave, fval, exitflag, output] = cplexqp (H_y,f_y',  [], [],[], [],qp.y_L,qp.y_U,[]);
   dx_k = x_wave - x_k;
   dy_k = y_wave - y_k;
%  for j=1:N*T
%      if z_hat(j)<=qp.b_up(j)&&z_hat(j)>=-qp.b_down(j)
%         z_wave(j)= z_hat(j);
%      else if z_hat(j)>qp.b_up(j)
%              z_wave(j)=qp.b_up(j);
%          else
%               z_wave(j)=-qp.b_down(j);
%          end
%      end 
%  end
  
%   dz_k = z_wave - z_k;
   dlamda_k =epsilon*(E*x_k+F*y_k-c);%%%����ֵʵ���Ͽ��������ԽСԽ�ã�˵��û�г��Ӹ��¿��ܻ���ã����ǳ��Ӳ�����������
   d_k = [(dx_k)',(dy_k)',(dlamda_k)']';
   xyl_k=[x_k',y_k',lamda']';
%   er1=norm(d_k)/(norm([x_k',y_x',c']')+1);
   er2=norm(d_k)/(norm(xyl_k)+1);
   phieq = norm(min(E*x_wave+F*y_wave-c,0),inf);
   
   if er2 <= TOL % | niter>300 %%%%%%%%%%%������ֹ׼������������,�ƺ�����Զ�ﲻ�����׼�������ж��ٴξͻ�ִ�ж��ٴΡ���%�޸ģ�
       runtime = toc;
     %  fprintf(  ' %4d,%10.4f, %10.4f\n',niter,  obj, runtime)
       fprintf(  '%d& %10.2f& %10.4f& %10.2f\n', ...,
               niter,  runtime , obj, phieq)
     %      norm(d_k,inf)
           
      break;       
   else  %%%%%%%%%%%%%%go step3 ,generate search direction.
      t_k=1;
      L1=1;
      t_k=t_k/theta;
      M = (E*x_k)+(F*y_k)-c;
      P_k = obj-lamda'*M+0.5*betar*(norm(M))^2;
    %%%%%%%%%%%%%%%%%%%%%%%%%
    while (L1==1) 
       % L1=0;
        t_k=t_k*theta; 
        x_new=x_k+t_k*dx_k;
        y_new=y_k+t_k*dy_k;
        x_hat_new=x_new'.*x_new';%����
        y_hat_new=y_new'.*y_new';%����
        %lamda_new = lamda+t_k*dlamda_k;
        f_new = x_hat_new*qp.q1_UC*x_new +x_new'* qp.C1_UC *x_new+qp.K1_UC' *x_new+sum(qp.d1_UC);%�޸�
        theta_new =y_hat_new*qp.q2_UC*y_new +y_new'* qp.C2_UC *y_new+qp.K2_UC' *y_new+sum(qp.d2_UC);%�޸�
        h_new = 0;
        M_new = E*x_new+F*y_new-c;
        P_new = f_new+theta_new+h_new-lamda'*M_new+0.5*betar*norm(M_new)^2;     
        
      if (P_new>P_k-sigma*t_k*(dx_k'*H_x*dx_k+dy_k'*H_y*dy_k))
               %disp(sprintf('Loop3'));
               L1=1;
               %break
            else
                L1=0;
      end  
     end 
   x_k = x_k+t_k*dx_k;
   y_k = y_k+t_k*dy_k;
   x_hat_k=x_k'.*x_k';%����
   y_hat_k=y_k'.*y_k';%����
   qp.x_juzhen=sparse(1:N1*T,1:N1*T,x_k);
   qp.y_juzhen=sparse(1:N2*T,1:N2*T,y_k);
   lamda = lamda+dlamda_k;
%   lamda_ie=lamda_ie+t_k*epsilon*((C*x_k)+(D*y_k)-d);
   niter=niter+1;
 %%%%%%%%%���µõ���ֵ����������һ��Ŀ�꺯��ֵ���ݶ�ֵ%%%%%%%%%%%%%%%%%%%%%  
obj_f = x_hat_k*qp.q1_UC*x_k +x_k'*qp.C1_UC*x_k+qp.K1_UC'*x_k+sum(qp.d1_UC);%f(x)��ֵ
obj_theta = y_hat_k*qp.q2_UC*y_k +y_k'*qp.C2_UC*y_k+qp.K2_UC'*y_k+sum(qp.d2_UC);%theta(y)��ֵ
obj = obj_f+obj_theta;%f(x)+theta(y)��ֵ
g_x = (3*qp.q1_UC*qp.x_juzhen+2*qp.C1_UC)*x_k+qp.K1_UC;%f(x)���ݶ�
g_y = (3*qp.q2_UC*qp.y_juzhen+2*qp.C2_UC)*y_k+qp.K2_UC;%theta(y)���ݶ�
H_x_k = qp.Q1_UC*qp.x_juzhen+2*qp.C1_UC;%f(x)��hessian��
H_y_k = qp.Q2_UC*qp.y_juzhen+2*qp.C2_UC;%theta(y)��hessian��

 
   end
   
 end
 fprintf('over') 
end
 % The end of the whole loop

 
 
 










    
 