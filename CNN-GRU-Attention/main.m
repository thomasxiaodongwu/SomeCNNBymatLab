warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������
tic
% restoredefaultpath

%% ��������
res = xlsread('data.xlsx');

%%  ���ݷ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
outdim = 1;                                  % ���һ��Ϊ���
num_samples = size(res, 1);                  % ��������
res = res(randperm(num_samples), :);         % �������ݼ�����ϣ������ʱ��ע�͸��У�
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��

%%  ����ѵ�����Ͳ��Լ�
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ����ƽ��
%   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
%   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
%   ����Ӧ��ʼ�պ���������ݽṹ����һ��
p_train =  double(reshape(p_train, f_, 1, 1, M));
p_test  =  double(reshape(p_test , f_, 1, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';

%%  ���ݸ�ʽת��
for i = 1 : M
    Lp_train{i, 1} = p_train(:, :, 1, i);
end

for i = 1 : N
    Lp_test{i, 1}  = p_test( :, :, 1, i);
end
    
%%  ����ģ��
lgraph = layerGraph();                                                 % �����հ�����ṹ

tempLayers = [
    sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % ��������㣬�������ݽṹΪ[f_, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                          % ���������۵���
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = convolution2dLayer([1, 1], 32, "Name", "conv_1");         % ����� �����[1, 1] ����[1, 1] ͨ���� 32
lgraph = addLayers(lgraph,tempLayers);                                 % ����������ṹ����հ׽ṹ��
 
tempLayers = [
    reluLayer("Name", "relu_1")                                        % �����
    convolution2dLayer([1, 1], 64, "Name", "conv_2")                   % ����� �����[1, 1] ����[1, 1] ͨ���� 64
    reluLayer("Name", "relu_2")];                                      % �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")                      % ȫ��ƽ���ػ���
    fullyConnectedLayer(16, "Name", "fc_2")                            % SEע�������ƣ�ͨ������1 / 4
    reluLayer("Name", "relu_3")                                        % �����
    fullyConnectedLayer(64, "Name", "fc_3")                            % SEע�������ƣ���Ŀ��ͨ������ͬ
    sigmoidLayer("Name", "sigmoid")];                                  % �����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % ��˵�ע����
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % �������з��۵���
    flattenLayer("Name", "flatten")                                    % ������ƽ��
    gruLayer(50,'Name','gru1',"OutputMode", "last")                    % gru��
    fullyConnectedLayer(1, "Name", "fc")                               % ȫ���Ӳ�
    regressionLayer("Name", "regressionoutput")];                      % �ع��
lgraph = addLayers(lgraph, tempLayers);                                % ����������ṹ����հ׽ṹ��

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % �۵������ ���� ���������;
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                       % �۵������ ���� ���۵�������  
lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % �������� ���� �����
lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % �������� ���� ȫ��ƽ���ػ�
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % �������� ���� ��˲�
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % ȫ������� ���� ��˲�
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % ������

%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
    'MaxEpochs', 500, ...                 % ����������
    'InitialLearnRate', 1e-2, ...          % ��ʼѧϰ��Ϊ0.01
    'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
    'LearnRateDropFactor', 0.1, ...        % ѧϰ���½����� 0.5
    'LearnRateDropPeriod', 400, ...        % ����700��ѵ���� ѧϰ��Ϊ 0.01 * 0.1
    'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
    'Plots', 'training-progress', ...      % ��������
    'Verbose', false);

%%  ѵ��ģ��
net = trainNetwork(Lp_train, t_train, lgraph, options);

%%  ģ��Ԥ��
t_sim1 = predict(net, Lp_train);
t_sim2 = predict(net, Lp_test );

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1', ps_output);
T_sim2 = mapminmax('reverse', t_sim2', ps_output);
T_sim1=double(T_sim1);
T_sim2=double(T_sim2);
%%  ��ʾ����ṹ
% analyzeNetwork(net)

%% ���Լ����
figure;
plotregression(T_test,T_sim2,['�ع�ͼ']);
figure;
ploterrhist(T_test-T_sim2,['���ֱ��ͼ']);
%%  ��������� RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%%
%����ϵ��
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%%
%������� MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;
%%
%RPD ʣ��Ԥ��в�
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;

SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;
%% ƽ���������MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));
%% ƽ�����԰ٷֱ����MAPE
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));
%%  ѵ������ͼ
figure
%plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1)
plot(1:M,T_train,'r-',1:M,T_sim1,'b-','LineWidth',1.5)
legend('��ʵֵ','CNN-GRU-AttentionԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'ѵ����Ԥ�����Ա�';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
title(string)
%% Ԥ�⼯��ͼ
figure
plot(1:N,T_test,'r-',1:N,T_sim2,'b-','LineWidth',1.5)
legend('��ʵֵ','CNN-GRU-AttentionԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'���Լ�Ԥ�����Ա�';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)

%% ���Լ����ͼ
figure  
ERROR3=T_test-T_sim2;
plot(T_test-T_sim2,'b-*','LineWidth',1.5)
xlabel('���Լ��������')
ylabel('Ԥ�����')
title('���Լ�Ԥ�����')
grid on;
legend('Ԥ��������')
%% �����������ͼ
%% ѵ�������Ч��ͼ
figure
plot(T_train,T_sim1,'*r');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string = {'ѵ����Ч��ͼ';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% Ԥ�⼯���Ч��ͼ
figure
plot(T_test,T_sim2,'ob');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string1 = {'���Լ�Ч��ͼ';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% ��ƽ��
R3=(R1+R2)./2;
error3=(error1+error2)./2;
%% ����������Ԥ�����ͼ
tsim=[T_sim1,T_sim2]';
S=[T_train,T_test]';
figure
plot(S,tsim,'ob');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string1 = {'�����������Ԥ��ͼ';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% ��ӡ������ָ��
disp(['-----------------------������--------------------------'])
disp(['���۽��������ʾ��'])
disp(['ƽ���������MAEΪ��',num2str(MAE2)])
disp(['�������MSEΪ��       ',num2str(mse2)])
disp(['���������RMSEΪ��  ',num2str(error2)])
disp(['����ϵ��R^2Ϊ��  ',num2str(R2)])
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD2)])
disp(['ƽ�����԰ٷֱ����MAPEΪ��  ',num2str(MAPE2)])
grid
