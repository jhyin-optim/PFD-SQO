
function [ qp_i,qp ] = qpED( dataUC )
ramp = 1;
N = dataUC.N;   T = dataUC.T;
N1=ceil(dataUC.N/2);
N2=N-N1;
x_L = cell(N1,1);    x_U = cell(N1,1);
A1 = cell(N1,1);      b1 = cell(N1,1);
%B = cell(N,1);      
B1_wan = cell(N1,1);  q1_UC = cell(N1,1);
Q1_UC = cell(N1,1);   c1_UC = cell(N1,1);
 K1_UC = cell(N1,1);  x_l = cell(N1,1); x_u = cell(N1,1);
%ctype = cell(N,1);
y_L = cell(N2,1);    y_U = cell(N2,1);
A2 = cell(N2,1);      b2 = cell(N2,1);
%B = cell(N,1);      
B2_wan = cell(N2,1);  q2_UC = cell(N2,1);
Q2_UC = cell(N2,1);   c2_UC = cell(N2,1);
 K2_UC = cell(N2,1); y_l = cell(N2,1); y_u = cell(N2,1);
b_up = cell(N,1);
b_down = cell(N,1);

for i = 1:N
     b_up{i}= sparse(1:T,1,dataUC.p_rampup(i));% �����Ͻ�
    b_down{i}= sparse(1:T,1,dataUC.p_rampdown(i));% �����Ͻ�
    c_wan = dataUC.PD;
end
for i = 1:N1   % ���ջ���ֿ��γ�Լ�����Ի���i���������������Լ����Ϊһ��
%      x_L{i} = sparse();% �����½�
    x_L{i} = sparse(1:T,1,dataUC.p_low(i));% �����½�dataUC.p_low(i)
    %x_l{i} = sparse(1:T,1:T,dataUC.p_low(i));%����
    x_U{i} = sparse(1:T,1,dataUC.p_up(i));% �����Ͻ�
   % x_u{i} = sparse(1:T,1:T,dataUC.p_up(i));%����
   
    
    % Ramp rate limits constrains
    A1_ramp_up = sparse(T,T); 

    b1_ramp_up = sparse(T,1);

    for t = 1:T 
        if t == 1      
            A1_ramp_up(t, t) = -1;
            b1_ramp_up(t) =  dataUC.p_initial(i);
           
        else      
            A1_ramp_up(t, t-1) = 1;
            A1_ramp_up(t, t) = -1;
            b1_ramp_up(t) =  0;
         end
        
    end
    % form A_i  b_i          
    if ramp == 1
        A1{i} = A1_ramp_up;%%%%����
                    %A_ramp_down;%%%%����
            
        b1{i} = b1_ramp_up;
                   % b_ramp_down;
    end
    %power balance constraint
    B1_wan{i} = sparse(1:T, 1:T,1);
    % �γɹ���ƽ��ͱ���Լ�����Ҷ���

    % Ŀ�꺯��
    c1_UC{i} = dataUC.b(i) * ones(T,1); %������ϵ��    %�޸�
    C1_UC{i} = sparse(1:T,1:T, c1_UC{i});%����
    Q1_UC{i} = sparse(1:T,1:T, 6*dataUC.a(i));%�����󵼺�������ϵ��   %�޸�
    q1_UC{i} = sparse(1:T,1:T, dataUC.a(i));%Ŀ�꺯����������ϵ������û�󵼵�ʱ��    %�޸�
    K1_UC{i} = dataUC.c(i) * ones(T,1); %һ����ϵ��   %�޸�
    d1_UC{i}=dataUC.d(i) ;   %������ϵ��   %����  
end

% ���طֿ��ģ������
qp_i.A1 = A1;          qp_i.b_up = b_up;
qp_i.b_down = b_down;   qp_i.b1 = b1;
qp_i.B1_wan = B1_wan;   
qp_i.c1_UC = c1_UC;     qp_i.Q2_UC = Q1_UC; 
 qp_i.q1_UC = q1_UC;
qp_i.x_L = x_L;       qp_i.x_U = x_U;
qp_i.N1 = N1;           qp_i.T = T;
qp_i.K1_UC=K1_UC;       qp_i.c_wan = c_wan;
for i = N1+1:N   % ���ջ���ֿ��γ�Լ�����Ի���i���������������Լ����Ϊһ��
%      x_L{i} = sparse();% �����½�
    y_L{i} = sparse(1:T,1,dataUC.p_low(i));% �����½�dataUC.p_low(i)
    %y_l{i} = sparse(1:T,1:T,dataUC.p_low(i));%����
    y_U{i} = sparse(1:T,1,dataUC.p_up(i));% �����Ͻ�
    %y_u{i} = sparse(1:T,1:T,dataUC.p_up(i));%����
   
    
    % Ramp rate limits constrains
    A2_ramp_up = sparse(T,T); 

    b2_ramp_up = sparse(T,1);

    for t = 1:T 
        if t == 1      
            A2_ramp_up(t, t) = -1;
            b2_ramp_up(t) =  dataUC.p_initial(i);
           
        else      
            A2_ramp_up(t, t-1) = 1;
            A2_ramp_up(t, t) = -1;
            b2_ramp_up(t) =  0;
         end
        
    end
    % form A_i  b_i          
    if ramp == 1
        A2{i} = A2_ramp_up;%%%%����
                    %A_ramp_down;%%%%����
            
        b2{i} = b2_ramp_up;
                   % b_ramp_down;
    end
    %power balance constraint
    B2_wan{i} = sparse(1:T, 1:T,1);
    % Ŀ�꺯��
    c2_UC{i} = dataUC.b(i) * ones(T,1); %������ϵ��    %�޸�
    C2_UC{i} = sparse(1:T,1:T, c2_UC{i});%����
    Q2_UC{i} = sparse(1:T,1:T, 6*dataUC.a(i));%�����󵼺�������ϵ��   %�޸�
    q2_UC{i} = sparse(1:T,1:T, dataUC.a(i));%Ŀ�꺯����������ϵ������û�󵼵�ʱ��    %�޸�
    K2_UC{i} = dataUC.c(i) * ones(T,1); %һ����ϵ��   %�޸�
    d2_UC{i}=dataUC.d(i) ;   %������ϵ��   %����  
end

% ���طֿ��ģ������
qp_i.A2 = A2;          
qp_i.b2 = b2;
qp_i.B2_wan = B2_wan;   
qp_i.c2_UC = c2_UC;     qp_i.Q2_UC = Q2_UC; 
 qp_i.q2_UC = q2_UC;
qp_i.y_L = y_L;       qp_i.y_U = y_U;
qp_i.N2 = N2;          
qp_i.K2_UC=K2_UC; 
% �γɲ����� ����MIQPģ�͵Ĳ���
qp.A1 = [];           qp.b_up = [];
qp.b_down = [];       qp.b1 = [];

qp.c1_UC = [];     qp.Q1_UC = [];
qp.q1_UC = [];
qp.B1_wan = [];   qp.c_wan = c_wan;
qp.x_L = [];       qp.x_U = [];
qp.K1_UC =[];
qp.C1_UC=[];%����
qp.d1_UC=[];%����
%qp.x_juzhen=[];%����
qp.A2 = [];           qp.b_up = [];
qp.b_down = [];       qp.b2 = [];

qp.c2_UC = [];     qp.Q2_UC = [];
qp.q2_UC = [];
qp.B2_wan = [];   
qp.y_L = [];       qp.y_U = [];
qp.K2_UC =[];
qp.C2_UC=[];%����
qp.d2_UC=[];%����
for i = 1:N1
   qp.A1 = blkdiag(qp.A1, A1{i});
   qp.b1 = [qp.b1; b1{i}];
   
  
   qp.B1_wan = [qp.B1_wan, B1_wan{i}];
   qp.c1_UC = [qp.c1_UC; c1_UC{i}];
   qp.K1_UC = [qp.K1_UC; K1_UC{i}];
   qp.q1_UC = blkdiag(qp.q1_UC, q1_UC{i});
   qp.Q1_UC = blkdiag(qp.Q1_UC, Q1_UC{i});
   qp.x_L = [qp.x_L; x_L{i}];
   qp.x_U = [qp.x_U; x_U{i}];
   qp.d1_UC =[qp.d1_UC; d1_UC{i}]; 
  % qp.x_juzhen = blkdiag(qp.x_juzhen, (x_l{i}+x_u{i})/2);%����
   qp.C1_UC = blkdiag(qp.C1_UC, C1_UC{i});%����
end
for i = N1+1:N
   
   qp.A2 = blkdiag(qp.A2, A2{i});
   qp.b2 = [qp.b2; b2{i}];
   
   qp.B2_wan = [qp.B2_wan, B2_wan{i}];
   qp.c2_UC = [qp.c2_UC; c2_UC{i}];
   qp.K2_UC = [qp.K2_UC; K2_UC{i}];
   qp.q2_UC = blkdiag(qp.q2_UC, q2_UC{i});
   qp.Q2_UC = blkdiag(qp.Q2_UC, Q2_UC{i});
   qp.y_L = [qp.y_L; y_L{i}];
   qp.y_U = [qp.y_U; y_U{i}];
   qp.d2_UC =[qp.d2_UC; d2_UC{i}]; 
   qp.C2_UC = blkdiag(qp.C2_UC, C2_UC{i});%����
end
for i=1:N
 qp.b_up = [qp.b_up; b_up{i}];
   qp.b_down = [qp.b_down; b_down{i}];
end
end


