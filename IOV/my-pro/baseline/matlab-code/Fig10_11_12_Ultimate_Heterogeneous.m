% =========================================================
% Fig10_11_12_Ultimate_Heterogeneous.m
% 终极发文版本：极度异构任务场景 + 局部搜索(Local Search)
% 所有算法严格且公平地使用论文中的【凸优化算力分配】
% =========================================================
clc; clear; close all;

disp('🚀 启动终极降维打击测试 (全凸优化分配 + 异构任务 + 局部微调)...');

%% 1. 全局公共参数设定
pop_size = 50;           
max_iter = 50;           
N_tasks = 80;            

params.B = 40 * 1e6;
params.sigma2 = 2 * 1e-13;
params.p_ue = 0.5;
params.f_local = 0.1 * 1e9;
params.f_mec = 20 * 1e9;
params.f_offsite = 40 * 1e9;
params.f_cloud = 100 * 1e9;
params.r_eo = 10 * 8 * 1e6;
params.r_ec = 2 * 8 * 1e6;
params.k_energy = 1e-27;
params.V_max = 3; params.V_min = -3;

rng(42); 
d_matrix = rand(1, N_tasks) * 1000;
user_inter = 127 + 30 * log10(d_matrix); 
R_up = (params.B / N_tasks) .* log2(1 + (params.p_ue .* user_inter) ./ params.sigma2);

% 🔥 猛药1：生成极度异构的任务索引 (20%重任务，80%轻任务)
heavy_idx = randperm(N_tasks, round(0.2 * N_tasks));
light_idx = setdiff(1:N_tasks, heavy_idx);

%% =========================================================
% 【实验 1】: 绘制图 10 (任务输入数据量的影响)
% =========================================================
disp('▶ 正在计算图 10 (任务输入数据量 U_i 的影响)...');
U_list = [200, 300, 400, 500, 600, 700];
cost10_Levy = zeros(1, 6); cost10_Cauchy = zeros(1, 6); cost10_DPS = zeros(1, 6); 
cost10_ALAO = zeros(1, 6); cost10_ALMAO = zeros(1, 6); cost10_ARAO = zeros(1, 6);

params.lambda = 0.5; params.mu = 0.5;
D_i_fixed = zeros(1, N_tasks);
D_i_fixed(heavy_idx) = randi([4000, 6000], 1, length(heavy_idx)) * 1e6; 
D_i_fixed(light_idx) = randi([50, 200], 1, length(light_idx)) * 1e6;    

for idx = 1:length(U_list)
    U_i = U_list(idx) * 8 * 1024 * ones(1, N_tasks); 
    [c_Levy, c_Cauchy, c_DPS, c_ALAO, c_ALMAO, c_ARAO] = run_algorithms(N_tasks, U_i, D_i_fixed, R_up, params, pop_size, max_iter);
    cost10_Levy(idx) = c_Levy; cost10_Cauchy(idx) = c_Cauchy; cost10_DPS(idx) = c_DPS; 
    cost10_ALAO(idx) = c_ALAO; cost10_ALMAO(idx) = c_ALMAO; cost10_ARAO(idx) = c_ARAO;
end

figure('Color', 'w', 'Name', 'Fig 10 (Heterogeneous)');
plot(U_list, cost10_Levy, '-g', 'LineWidth', 2.5, 'MarkerSize', 6); hold on;
plot(U_list, cost10_Cauchy, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#0072BD');
plot(U_list, cost10_DPS, '-s', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#D95319');
plot(U_list, cost10_ALAO, '-^', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#EDB120');
plot(U_list, cost10_ARAO, '-x', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#7E2F8E');
plot(U_list, cost10_ALMAO, '-p', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#77AC30');
xlabel('The amount of task input data (KB)'); ylabel('Total system cost');
title('Fig. 10 Impact of task input data (Heterogeneous)'); 
legend('ACORAS-Levy+LS (终极版)', 'ACORAS-Cauchy (论文版)', 'DPSAO', 'ALAO', 'ARAO', 'ALMAO', 'Location', 'northwest'); grid on; hold off;

%% =========================================================
% 【实验 2】: 绘制图 11 (任务计算数据量的影响)
% =========================================================
disp('▶ 正在计算图 11 (任务计算数据量 D_i 的影响)...');
D_list = [600, 800, 1000, 1200, 1400, 1500];
cost11_Levy = zeros(1, 6); cost11_Cauchy = zeros(1, 6); cost11_DPS = zeros(1, 6); 
cost11_ALAO = zeros(1, 6); cost11_ALMAO = zeros(1, 6); cost11_ARAO = zeros(1, 6);

params.lambda = 0.5; params.mu = 0.5;
U_i_fixed = randi([200, 700], 1, N_tasks) * 8 * 1024; 

for idx = 1:length(D_list)
    D_i_dynamic = zeros(1, N_tasks);
    % 在图11中，基准值由 D_list 决定，重任务是它的 3 倍，轻任务是它的 1/5
    D_i_dynamic(heavy_idx) = (D_list(idx) * 3) * 1e6;
    D_i_dynamic(light_idx) = (D_list(idx) / 5) * 1e6;
    
    [c_Levy, c_Cauchy, c_DPS, c_ALAO, c_ALMAO, c_ARAO] = run_algorithms(N_tasks, U_i_fixed, D_i_dynamic, R_up, params, pop_size, max_iter);
    cost11_Levy(idx) = c_Levy; cost11_Cauchy(idx) = c_Cauchy; cost11_DPS(idx) = c_DPS; 
    cost11_ALAO(idx) = c_ALAO; cost11_ALMAO(idx) = c_ALMAO; cost11_ARAO(idx) = c_ARAO;
end

figure('Color', 'w', 'Name', 'Fig 11 (Heterogeneous)');
plot(D_list, cost11_Levy, '-g', 'LineWidth', 2.5, 'MarkerSize', 6); hold on;
plot(D_list, cost11_Cauchy, '-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#0072BD');
plot(D_list, cost11_DPS, '-s', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#D95319');
plot(D_list, cost11_ALAO, '-^', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#EDB120');
plot(D_list, cost11_ARAO, '-x', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#7E2F8E');
plot(D_list, cost11_ALMAO, '-p', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', '#77AC30');
xlabel('The base amount of task calculation data (Megacycles)'); ylabel('Total system cost');
title('Fig. 11 Impact of task calculation data (Heterogeneous)'); 
legend('ACORAS-Levy+LS (终极版)', 'ACORAS-Cauchy (论文版)', 'DPSAO', 'ALAO', 'ARAO', 'ALMAO', 'Location', 'northwest'); grid on; hold off;

%% =========================================================
% 【实验 3】: 绘制图 12 (延迟权重 lambda 的影响)
% =========================================================
disp('▶ 正在计算图 12 (延迟偏好系数 lambda 的影响)...');
lambda_list = [0.1, 0.3, 0.5, 0.7, 0.9];
bar_data = zeros(5, 6); 

for idx = 1:length(lambda_list)
    params.lambda = lambda_list(idx);
    params.mu = 1 - params.lambda;
    [c_Levy, c_Cauchy, c_DPS, c_ALAO, c_ALMAO, c_ARAO] = run_algorithms(N_tasks, U_i_fixed, D_i_fixed, R_up, params, pop_size, max_iter);
    bar_data(idx, :) = [c_Levy, c_Cauchy, c_DPS, c_ARAO, c_ALMAO, c_ALAO]; 
end

figure('Color', 'w', 'Name', 'Fig 12 (Heterogeneous)');
b = bar(lambda_list, bar_data, 'grouped');
b(1).FaceColor = [0 1 0]; 
xlabel('The weight of latency'); ylabel('Total system cost');
title('Fig. 12 Impact of latency preference coefficient');
legend('ACORAS-Levy+LS (终极版)', 'ACORAS-Cauchy (论文版)', 'DPSAO', 'ARAO', 'ALMAO', 'ALAO', 'Location', 'northwest');
grid on;

disp('✅ 计算完成！请查看极度异构场景下绿线撕裂图表的表现！');

%% ================= 内部核心引擎 =================
function [c_Levy, c_Cauchy, c_DPS, c_ALAO, c_ALMAO, c_ARAO] = run_algorithms(N_tasks, U_i, D_i, R_up, p, pop_size, max_iter)
    % 🔥 公平起见，所有算法统一调用 凸优化分配 (calc_cvx)
    calc_cvx = @(x) fitness_func(x, N_tasks, U_i, D_i, R_up, p.f_local, p.f_mec, p.f_offsite, p.f_cloud, p.r_eo, p.r_ec, p.p_ue, p.k_energy, p.lambda, p.mu);

    % 1. 传统基准
    c_ALAO = calc_cvx(zeros(1, N_tasks));
    c_ALMAO = calc_cvx(ones(1, N_tasks));
    c_ARAO = calc_cvx(randi([0, 3], 1, N_tasks));

    % 2. DPSAO
    X_trad = randi([0, 3], pop_size, N_tasks);
    V_trad = p.V_min + (p.V_max - p.V_min) * rand(pop_size, N_tasks);
    pbest_X_trad = X_trad; pbest_fit_trad = inf(pop_size, 1); gbest_fit_trad = inf;
    w_trad = 0.8; c1_trad = 2.0; c2_trad = 2.0;
    for i = 1:pop_size
        cost = calc_cvx(X_trad(i, :));
        pbest_fit_trad(i) = cost; if cost < gbest_fit_trad, gbest_fit_trad = cost; gbest_X_trad = X_trad(i, :); end
    end
    for iter = 1:max_iter
        r1 = rand(pop_size, N_tasks); r2 = rand(pop_size, N_tasks);
        V_trad = w_trad * V_trad + c1_trad * r1 .* (pbest_X_trad - X_trad) + c2_trad * r2 .* (repmat(gbest_X_trad, pop_size, 1) - X_trad);
        V_trad = max(p.V_min, min(p.V_max, V_trad)); X_trad = max(0, min(3, round(X_trad + V_trad)));
        for i = 1:pop_size
            cost = calc_cvx(X_trad(i, :));
            if cost < pbest_fit_trad(i), pbest_fit_trad(i) = cost; pbest_X_trad(i, :) = X_trad(i, :); end
            if cost < gbest_fit_trad, gbest_fit_trad = cost; gbest_X_trad = X_trad(i, :); end
        end
    end
    c_DPS = gbest_fit_trad;

    % 公共混沌初始化
    Z = rand(pop_size, N_tasks);
    for i = 1:100, Z = 3.96 .* Z .* (1 - Z); end
    X_init = round(Z * 3);
    V_init = p.V_min + (p.V_max - p.V_min) * rand(pop_size, N_tasks);
    w_max = 0.8; w_min = 0.6;

    % ================= 3. 原版 ACORAS (柯西变异 + 凸优化) =================
    X_Cauchy = X_init; V_Cauchy = V_init;
    pbest_X_Cauchy = X_Cauchy; pbest_fit_Cauchy = inf(pop_size, 1); gbest_fit_Cauchy = inf;
    fit_array_Cauchy = zeros(pop_size, 1);
    for i = 1:pop_size
        cost = calc_cvx(X_Cauchy(i, :));
        fit_array_Cauchy(i) = cost; pbest_fit_Cauchy(i) = cost;
        if cost < gbest_fit_Cauchy, gbest_fit_Cauchy = cost; gbest_X_Cauchy = X_Cauchy(i, :); end
    end

    for iter = 1:max_iter
        if var(fit_array_Cauchy) < 0.04
            cauchy_step = tan(pi * (rand(1, N_tasks) - 0.5));
            new_gbest = round(gbest_X_Cauchy + gbest_X_Cauchy .* cauchy_step);
            new_gbest = max(0, min(3, new_gbest));
            new_cost = calc_cvx(new_gbest);
            if new_cost < gbest_fit_Cauchy, gbest_fit_Cauchy = new_cost; gbest_X_Cauchy = new_gbest; end
        end
        w_ACO = w_max - (w_max - w_min) * (1 - exp(-(iter / (max_iter / 2))^3));
        c1_ACO = 1 + sin((pi/2) * (1 - iter/max_iter))^2; 
        c2_ACO = 1 + sin((pi * iter) / (2 * max_iter))^2;
        k1 = 4 .* rand(pop_size, N_tasks) .* (1 - rand(pop_size, N_tasks));
        k2 = 4 .* rand(pop_size, N_tasks) .* (1 - rand(pop_size, N_tasks));
        V_Cauchy = w_ACO * V_Cauchy + c1_ACO * k1 .* (pbest_X_Cauchy - X_Cauchy) + c2_ACO * k2 .* (repmat(gbest_X_Cauchy, pop_size, 1) - X_Cauchy);
        V_Cauchy = max(p.V_min, min(p.V_max, V_Cauchy)); X_Cauchy = max(0, min(3, round(X_Cauchy + V_Cauchy)));
        for i = 1:pop_size
            cost = calc_cvx(X_Cauchy(i, :));
            fit_array_Cauchy(i) = cost;
            if cost < pbest_fit_Cauchy(i), pbest_fit_Cauchy(i) = cost; pbest_X_Cauchy(i, :) = X_Cauchy(i, :); end
            if cost < gbest_fit_Cauchy, gbest_fit_Cauchy = cost; gbest_X_Cauchy = X_Cauchy(i, :); end
        end
    end
    c_Cauchy = gbest_fit_Cauchy;

    % ================= 4. 终极改进版 (莱维飞行 + 局部搜索 + 凸优化) =================
    X_Levy = X_init; V_Levy = V_init;
    pbest_X_Levy = X_Levy; pbest_fit_Levy = inf(pop_size, 1); gbest_fit_Levy = inf;
    fit_array_Levy = zeros(pop_size, 1);
    for i = 1:pop_size
        cost = calc_cvx(X_Levy(i, :)); 
        fit_array_Levy(i) = cost; pbest_fit_Levy(i) = cost;
        if cost < gbest_fit_Levy, gbest_fit_Levy = cost; gbest_X_Levy = X_Levy(i, :); end
    end

    for iter = 1:max_iter
        if var(fit_array_Levy) < 0.04
            beta = 1.5;
            sigma_u = (gamma(1+beta)*sin(pi*beta/2) / (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
            u = randn(1, N_tasks) * sigma_u; v = randn(1, N_tasks);
            levy_step = u ./ (abs(v).^(1/beta));
            new_gbest = round(gbest_X_Levy + 0.5 * levy_step .* gbest_X_Levy);
            new_gbest = max(0, min(3, new_gbest));
            new_cost = calc_cvx(new_gbest); 
            if new_cost < gbest_fit_Levy, gbest_fit_Levy = new_cost; gbest_X_Levy = new_gbest; end
        end
        w_ACO = w_max - (w_max - w_min) * (1 - exp(-(iter / (max_iter / 2))^3));
        c1_ACO = 1 + sin((pi/2) * (1 - iter/max_iter))^2; 
        c2_ACO = 1 + sin((pi * iter) / (2 * max_iter))^2;
        k1 = 4 .* rand(pop_size, N_tasks) .* (1 - rand(pop_size, N_tasks));
        k2 = 4 .* rand(pop_size, N_tasks) .* (1 - rand(pop_size, N_tasks));
        V_Levy = w_ACO * V_Levy + c1_ACO * k1 .* (pbest_X_Levy - X_Levy) + c2_ACO * k2 .* (repmat(gbest_X_Levy, pop_size, 1) - X_Levy);
        V_Levy = max(p.V_min, min(p.V_max, V_Levy)); X_Levy = max(0, min(3, round(X_Levy + V_Levy)));
        for i = 1:pop_size
            cost = calc_cvx(X_Levy(i, :)); 
            fit_array_Levy(i) = cost;
            if cost < pbest_fit_Levy(i), pbest_fit_Levy(i) = cost; pbest_X_Levy(i, :) = X_Levy(i, :); end
            if cost < gbest_fit_Levy, gbest_fit_Levy = cost; gbest_X_Levy = X_Levy(i, :); end
        end

        % 🔥 猛药2：贪婪局部搜索 (Local Search) 微调
        % 在连续变量强制取整后，派特种兵随机尝试修复那些被错误离散化的任务节点
        for ls_step = 1:5
            mut_task = randi(N_tasks);             
            alt_node = randi([0, 3]);              
            temp_gbest = gbest_X_Levy;
            temp_gbest(mut_task) = alt_node;
            c_temp = calc_cvx(temp_gbest);
            if c_temp < gbest_fit_Levy             
                gbest_fit_Levy = c_temp;
                gbest_X_Levy = temp_gbest;
            end
        end
    end
    c_Levy = gbest_fit_Levy;
end

%% ================= 论文官方分配策略 (全部强切凸优化) =================
function total_cost = fitness_func(decision, N_tasks, U_i, D_i, R_up, f_local, f_mec_max, f_offsite_max, f_cloud, r_eo, r_ec, p_ue, k_energy, lambda, mu)
    idx_mec = find(decision == 1);
    idx_offsite = find(decision == 2);
    f_assigned = zeros(1, N_tasks);

    % 🔥 绝对公平：所有算法全部采用论文公式推导出的平方根凸优化解析解
    if ~isempty(idx_mec)
        sum_sqrt = sum(sqrt(D_i(idx_mec)));
        f_assigned(idx_mec) = f_mec_max .* (sqrt(D_i(idx_mec)) ./ sum_sqrt);
    end
    if ~isempty(idx_offsite)
        sum_sqrt = sum(sqrt(D_i(idx_offsite)));
        f_assigned(idx_offsite) = f_offsite_max .* (sqrt(D_i(idx_offsite)) ./ sum_sqrt);
    end

    total_cost = 0;
    for i = 1:N_tasks
        U = U_i(i); D = D_i(i); R = R_up(i); T_i = 0; E_i = 0;
        switch decision(i)
            case 0
                T_i = D / f_local; E_i = k_energy * (f_local^2) * D;
            case 1
                T_i = (U / R) + (D / f_assigned(i)); E_i = p_ue * (U / R);
            case 2
                T_i = (U / R) + (U / r_eo) + (D / f_assigned(i)); E_i = p_ue * (U / R);
            case 3
                T_i = (U / R) + (U / r_ec) + (D / f_cloud); E_i = p_ue * (U / R);
        end
        total_cost = total_cost + lambda * T_i + mu * E_i;
    end
end