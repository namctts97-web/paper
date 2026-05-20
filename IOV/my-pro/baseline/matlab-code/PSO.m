function [global_best, fitness_best_basic, final_global_fitness] = PSO(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,NP, n_uer, num_servers, v_min, v_max,  w1,c1,c2, G, R_up, D, U, W,pt,pw)
 

%% 初始化位置和速度


% 初始化粒子群位置矩阵
x = zeros(NP, n_uer); 

% 生成卸载决策向量的初始位置
for i1 = 1:NP
    % 为每个任务生成随机的卸载决策
    x(i1, :) = randi([0, num_servers], 1, n_uer); % 生成 0 到 num_servers 之间的整数
end

% 显示初始化的粒子群位置矩阵
%disp('初始粒子群位置矩阵:');
%disp(x);

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



%% 存储每一代的适应度值
fitness_best_basic = zeros(1, G);
fitness_best_basic(1) = global_best_fit; % 初始值可以设置为适当的值

%%  进行迭代（基础）
for gen = 1:G
    
    for k = 1:NP

        % 更新速度
        v(k, :) = w1 * v(k, :) + c1 * rand() * (individual_best(k, :) - x(k, :)) + c2 * rand() * (global_best - x(k, :));

        % 边界条件处理（速度边界设置）
        v(k, v(k, :) < -3) = -3;
        v(k, v(k, :) > 3) = 3;
        %v = round(v);

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

    
    
    
    
    %disp(iteration_variance(gen));
    T3=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,global_best,R_up,D,U);
    fitness_best_basic(gen + 1) = func(global_best, T3, W,pt,pw);
    %disp(fitnessbest(gen));
end


%{
plot(fitness_best_optimized);
xlabel('迭代次数');
ylabel('适应度值');
title('适应度优化过程');
%}

% 最终全局最优适应度
    T_final = time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,global_best, R_up, D, U);
    final_global_fitness = func(global_best, T_final, W,pt,pw);
    
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