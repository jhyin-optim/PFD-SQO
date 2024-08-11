function dataUC=readdataUC(pathAndFilename)
fprintf('\n\n\n');
disp('------------------------��ȡ����������������ļ�----------------------------');

%% ��ȡ����ϵͳ����
% pathAndFilename='UC_AF/10_std.mod';
dataUC.pathAndFilename = pathAndFilename;
fid=fopen(pathAndFilename);
%���Ե�1������
fgetl(fid);                    %ProblemNum
%��ȡ��2������
tmp=fscanf(fid,'%s',1);                     %HorizonLen
dataUC.T=fscanf(fid,'%d',1);                       %HorizonLen   ����ʱ��T
%��ȡ��3������
tmp=fscanf(fid,'%s',1);                                %NumThermal
dataUC.N=fscanf(fid,'%d',1);                       %NumThermal   ����������N
%���Ե�4������
fgetl(fid);  fgetl(fid);   %  du Loads	1	24

% ��ȡ��5������ ����
dataUC.PD=fscanf(fid,'%f',[1,dataUC.T]);   
dataUC.PD = dataUC.PD';                   % ����
% ���Ե�6������
fgetl(fid);fgetl(fid);

%��ȡN̨�������
for i = 1:dataUC.N  
    unit_parameters(i,:) = fscanf(fid,'%f',[1,7]);
    fscanf(fid,'%s',1);
    dataUC.p_rampup(i) = fscanf(fid,'%f',1);
    dataUC.p_rampdown(i) = fscanf(fid,'%f',1);
end

fgetl(fid);fgetl(fid);

dataUC.p_rampup = dataUC.p_rampup';   
dataUC.p_rampdown= dataUC.p_rampdown';

dataUC.a = unit_parameters(:,5);
dataUC.b = unit_parameters(:,4);
dataUC.c = unit_parameters(:,3);
dataUC.d= unit_parameters(:,2);

dataUC.p_low = unit_parameters(:,6);
dataUC.p_up = unit_parameters(:,7);
dataUC.p_initial = 0.5*unit_parameters(:,7);%% ��ʼ״̬����Ϊ��������50%


dataUC.expended=0;

fclose(fid);



