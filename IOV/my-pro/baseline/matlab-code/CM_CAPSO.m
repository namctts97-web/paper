function [global_best, fitness_best_xiugai,final_global_fitness] = CM_CAPSO(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,NP, n_uer, num_servers, v_min, v_max, w_min, w_max, G, R_up, D, U, W,pt,pw)


% 设置 Logistic Map 参数
r = 3.96;    %收敛率
num_iterations = 100;

% 初始化粒子群位置矩阵
x = zeros(NP, n_uer);

for i1 = 1:NP
    % 设置不同的随机数生成种子
    rng('shuffle');

    for k = 1:n_uer
        % 初始状态可以在 [0, 1] 范围内随机选择
        initial_state = rand();

        for j1 = 1:num_iterations
            % 利用 Logistic Map 进行迭代
            initial_state = logistic_map(initial_state, r);
        end

        % 最终状态作为粒子的初始位置的一个元素
        x(i1, k) = round(initial_state * 3);
    end
end

% 速度进行初始化
v = round(v_min + rand(NP, n_uer) * (v_max - v_min));
%disp(v);


%% 初始化适宜度
% 初始化个体最优
individual_best = x;       %  每个个体的历史最优
pbest = zeros(NP, 1);      %  个体最优位置对应的适应度值

% 初始化个体最优
for k=1:NP
    T=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
    pbest(k, 1) = func(individual_best(k,:),T,W,pt,pw);
end
% 初始化全局最优
global_best = zeros(1, n_uer);
global_best_fit = realmax;
for k=1:NP
    T=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
    temp = func(individual_best(k,:),T,W,pt,pw);
    %disp(temp);
    if temp < global_best_fit
        global_best = individual_best(k,:);
        global_best_fit = temp;
    end
end



% 初始化个体最优和 fitness_best_optimized(1)
fitness_best_xiugai = zeros(1, G);
fitness_best_xiugai(1) = global_best_fit; % 初始值可以设置为适当的值


%%  进行迭代（基础）
for gen = 1:G
    
    for k = 1:NP
        
        w = w_min + (w_max - w_min) * (1 - exp(-(gen /(G/2)).^3));
        c1 = 1 + 2 * sin(pi/2 * (1 - gen/G)).^2;
        c2 = 1 + 2 * sin(pi/2 * gen/G).^2;

        % 更新速度
        v(k, :) = w * v(k, :) + c1 * rand() * (individual_best(k, :) - x(k, :)) +c2 * rand() * (global_best - x(k, :));

        % 边界条件处理（速度边界设置）
        v(k, v(k, :) < -3) = -3;
        v(k, v(k, :) > 3) = 3;

        % 更新位置
        x(k, :) = x(k, :) + v(k, :);

        % 边界条件处理（位置边界设置）
        x(k, x(k, :) < 0) = 0;
        x(k, x(k, :) > num_servers) = num_servers;
        x = round(x);

        % 适应度值
        T0=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,x(k,:),R_up,D,U);
        fitness(k) = func(x(k, :), T0,W,pt,pw);
        %disp(fitness(k));

        % 更新个体最优
        T1=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
        if fitness(k) < func(individual_best(k, :),T1,W,pt,pw)   
            individual_best(k, :) = x(k, :);
        end

        % 更新全局最优
        T1=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
        T2=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,global_best,R_up,D,U);
        if func(individual_best(k, :), T1,W,pt,pw) < func(global_best, T2,W,pt,pw)
            global_best = individual_best(k, :);
            %disp(global_best)
        end
        
        T1=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
        particle_fitness(k) = func(individual_best(k, :),T1,W,pt,pw);
        % 输出速度和位置
        %fprintf('第 %d 代, 粒子 %d - 速度: %s, 位置: %s\n', gen, k, mat2str(v(k, :)), mat2str(x(k, :)));
    end

       
        % 计算当前迭代粒子群适应度的方差
    iteration_variance(gen) = var(particle_fitness);
    % 如果适应度方差小于0.04，进行柯西变异
    if iteration_variance(gen) < 0.04
        % 生成服从标准柯西分布的随机数
         cauchy_random_numbers = tan(pi * (rand(size(global_best)) - 0.5));

        %disp("变异前");
        %disp(global_best)
        % 柯西变异
         cauchy_mutated_particle = global_best + global_best .* cauchy_random_numbers;
        %disp("变异后");
        %disp(cauchy_mutated_particle)
        % 对柯西变异后的粒子进行位置边界处理
        cauchy_mutated_particle(cauchy_mutated_particle < 0) = 0;
        cauchy_mutated_particle(cauchy_mutated_particle > num_servers) = num_servers;
        cauchy_mutated_particle = round(cauchy_mutated_particle);
        
        
        % 计算柯西变异后的适应度值
        T_cauchy = time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,cauchy_mutated_particle, R_up, D, U);
        fitness_cauchy = func(cauchy_mutated_particle, T_cauchy, W,pt,pw);
        
        % 更新全局最优
        if fitness_cauchy < func(global_best, T2, W,pt,pw)
            global_best = cauchy_mutated_particle;
        end
        % disp(iteration_variance(gen));
    end
   
    
    
    
    
    %disp(iteration_variance(gen));
    T3=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,global_best,R_up,D,U);
    fitness_best_xiugai(gen + 1) = func(global_best, T3, W,pt,pw);
    %disp(fitnessbest(gen));
end  


% 最终全局最优适应度
    T_final = time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,global_best, R_up, D, U);
    final_global_fitness = func(global_best, T_final, W,pt,pw);
  
   
%% Logistic Map 函数
function x_next = logistic_map(x, r)
    x_next = r * x * (1 - x);
end    
    
%% 定义时延矩阵函数
function T = time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,x,R_up,D,U)
    n_ue = length(x);
    n1 = sum(x == 1);
    n2 = sum(x == 2);
        %% 任务的时延
    T=zeros(n_ue,4);
    for i = 1:n_ue %遍历行数，每个任务
        for j = 1:4 %遍历列数，每种决策
                    % 根据不同的列数选择不同的计算方式
            if j == 1
                T(i, j) = D(1,i)* 1e6/f_local;
            elseif j == 2
                %T(i, j) = U(1,i)*8*1e3/R_up(1,i)
                %                                                                                       
                T(i, j) = U(1,i)*8*1e3/R_up(1,i)+D(1,i)* 1e6/(f_mec_max/n1);
            elseif j == 3
                %T(i, j) = U(1,i)*8*1e3/R_up(1,i)+U(1,i)/r_eo
                T(i, j) = U(1,i)*8*1e3/R_up(1,i)+U(1,i)/r_eo+D(1,i)* 1e6/(f_offsite_max/n2);
            elseif j == 4
                %T(i, j) = U(1,i)*8*1e3/R_up(1,i)+U(1,i)/r_ec
                T(i, j) = U(1,i)*8*1e3/R_up(1,i)+U(1,i)/r_ec+D(1,i)* 1e6/f_cloud;
            end
        end
    end
end

%% 定义适应度函数
function res = func(x,T,W,pt,pw)

   % 确定新矩阵的大小
    rows = length(x);

    % 创建一个全零的矩阵
    binaryMatrix = zeros(rows, 4);

    % 将相应的位置设为1
    for i = 1:rows
        binaryMatrix(i, x(i) + 1) = 1;
    end
    result_sum1 = sum(binaryMatrix .* T, 'all');
    result_sum2 = sum(binaryMatrix .* W, 'all');
    % 初始化和为零
    res = pt*result_sum1+pw*result_sum2;
end

end