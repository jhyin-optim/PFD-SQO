function dataUC=readdataUC(pathAndFilename)
fprintf('\n\n\n');
disp('------------------------读取机组组合问题数据文件----------------------------');

%% 读取电力系统参数
% pathAndFilename='UC_AF/10_std.mod';
dataUC.pathAndFilename = pathAndFilename;
fid=fopen(pathAndFilename);
%忽略第1行数据
fgetl(fid);                    %ProblemNum
%读取第2行数据
tmp=fscanf(fid,'%s',1);                     %HorizonLen
dataUC.T=fscanf(fid,'%d',1);                       %HorizonLen   即：时段T
%读取第3行数据
tmp=fscanf(fid,'%s',1);                                %NumThermal
dataUC.N=fscanf(fid,'%d',1);                       %NumThermal   即：机组数N
%忽略第4行数据
fgetl(fid);  fgetl(fid);   %  du Loads	1	24

% 读取第5行数据 负荷
dataUC.PD=fscanf(fid,'%f',[1,dataUC.T]);   
dataUC.PD = dataUC.PD';                   % 负荷
% 忽略第6行数据
fgetl(fid);fgetl(fid);

%读取N台机组参数
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
dataUC.p_initial = 0.5*unit_parameters(:,7);%% 初始状态设置为最大出力的50%


dataUC.expended=0;

fclose(fid);



