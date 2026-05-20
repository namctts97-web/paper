clc;
clear all;
close all;

%% 初始化变量
n_ue = 80;     % 设备/任务数量
U = randi([100, 500], 1, n_ue);  %N个任务传输数据大小kb
D =randi([100, 1000], 1, n_ue);  %N个任务计算资源量mHz 1mHZ=10^6

p_no = 2.0e-13;  % 高斯信道噪声𝜎2 w
p_ue = 0.5;      % 用户设备传输功率w
W_max = 4 * 1e7;       %系统通信带宽Hz

k=1 * 10^(-27);          %芯片的能耗系数
r=2;
d_matrix = rand(n_ue, 1) * 1000;    % 用户设备到基站的距离m

r_eo=  1 * 1e4;     % MEC到异地的传输速率kb/s
r_ec = 6 * 1e2;    % MEC到云端的传输速率kb/s

f_local = 1 * 1e8 ;  % 本地计算的计算频率
f_mec_max = 30 * 1e9;    % 本地MEC计算的总计算频率
f_offsite_max = 50 * 1e9;    % 异地地MEC计算的总计算频率
f_cloud = 100 * 1e9;  % 云计算的计算频率

pt=0.5;     % 时延权重
pw=1-pt;   % 能耗权重

%% 上行链路速率
user_inter=zeros(1,n_ue);%用于存储每个用户与基站之间的干扰。
for i = 1:n_ue
    user_inter(i) = 127 + 30 * log10(d_matrix(i));
end

R_up=zeros(1,n_ue);%用于存储每个用户与基站之间的上行链路速率。
for i = 1:n_ue
    % 将 Table II 的 dB 损耗转化为线性信道增益 h_i (由于 d 是米，先除以 1000 转 km)
    h_i_linear = 10^(-(127 + 30 * log10(d_matrix(i)/1000)) / 10);
    % 严格套用公式 (1)
    R_up(i) = (W_max/n_ue)*log2(1 + h_i_linear * p_ue / p_no);
end

%% 任务的能耗
W = zeros(n_ue,4);
for i = 1:n_ue %遍历行数，每个任务
    for j = 1:4 %遍历列数，每种决策
        if j == 1
            W(i, j) = D(1,i)* 1e6* k *(1 * 1e8)^r;
        elseif j == 2
            W(i, j) = (U(1,i)*8*1e3/R_up(1,i))*p_ue;
        elseif j == 3
            W(i, j) = (U(1,i)*8*1e3/R_up(1,i))*p_ue;
        elseif j == 4
            W(i, j) = (U(1,i)*8*1e3/R_up(1,i))*p_ue;
        end
    end
end

%% 粒子群  参数设置
NP = 70;      % 种群个数
n_uer=n_ue;   %任务数
G = 150;       % 迭代次数 
c1 = 1.5;      % 学习因子
c2 = 1.5;
w_max = 0.8;   % 惯性权重
w_min = 0.6;
w1=0.5;
v_max = 3;     % 粒子的速度限制
v_min = -3;
penality = 2;  %惩罚系数
num_servers = 3;

% 调用优化算法PSO (传统)
[final_global_best2, final_fitness2,final_global_fitness2] = PSO(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,NP, n_ue, num_servers, v_min, v_max, w1,c1,c2 ,G, R_up, D, U, W,pt,pw);

% 调用优化算法CM_CAPSO (原版柯西)
[final_global_best3, final_fitness3,final_global_fitness3] = CM_CAPSO(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,NP, n_ue, num_servers, v_min, v_max, w_min, w_max, G, R_up, D, U, W,pt,pw);

% 🔥调用新增的优化算法: CM_CAPSO_Levy (莱维飞行改进版)🔥
[final_global_best_levy, final_fitness_levy, final_global_fitness_levy] = CM_CAPSO_Levy(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,NP, n_ue, num_servers, v_min, v_max, w_min, w_max, G, R_up, D, U, W,pt,pw);

% 绘制适应度值比较图
figure;
plot(0:G, final_fitness_levy, 'g-', 'LineWidth', 2, 'DisplayName', 'ACORAS-Levy (改进版)'); % 莱维飞行画绿色粗线
hold on;
plot(0:G, final_fitness3, 'LineWidth', 1.5, 'DisplayName', 'ACORAS-Cauchy (论文原版)');
plot(0:G, final_fitness2, 'LineWidth', 1.5, 'DisplayName', 'DPSAO (传统基准)');

xlabel('Number of iterations');
ylabel('Total system cost');
legend('show');
title('Convergence Comparison (with Lévy Flight)');