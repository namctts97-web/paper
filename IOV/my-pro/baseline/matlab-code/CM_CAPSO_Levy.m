function [global_best, fitness_best_xiugai,final_global_fitness] = CM_CAPSO_Levy(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,NP, n_uer, num_servers, v_min, v_max, w_min, w_max, G, R_up, D, U, W,pt,pw)

% 设置 Logistic Map 参数
r = 3.96;    %收敛率
num_iterations = 100;

% 初始化粒子群位置矩阵
x = zeros(NP, n_uer);

for i1 = 1:NP
    rng('shuffle');
    for k = 1:n_uer
        initial_state = rand();
        for j1 = 1:num_iterations
            initial_state = logistic_map(initial_state, r);
        end
        x(i1, k) = round(initial_state * 3);
    end
end

v = round(v_min + rand(NP, n_uer) * (v_max - v_min));

%% 初始化适宜度
individual_best = x;       
pbest = zeros(NP, 1);      

for k=1:NP
    T=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
    pbest(k, 1) = func(individual_best(k,:),T,W,pt,pw);
end

global_best = zeros(1, n_uer);
global_best_fit = realmax;
for k=1:NP
    T=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
    temp = func(individual_best(k,:),T,W,pt,pw);
    if temp < global_best_fit
        global_best = individual_best(k,:);
        global_best_fit = temp;
    end
end

fitness_best_xiugai = zeros(1, G);
fitness_best_xiugai(1) = global_best_fit; 

%%  进行迭代（基础）
for gen = 1:G
    for k = 1:NP
        w = w_min + (w_max - w_min) * (1 - exp(-(gen /(G/2)).^3));
        c1 = 1 + 2 * sin(pi/2 * (1 - gen/G)).^2;
        c2 = 1 + 2 * sin(pi/2 * gen/G).^2;

        v(k, :) = w * v(k, :) + c1 * rand() * (individual_best(k, :) - x(k, :)) +c2 * rand() * (global_best - x(k, :));
        v(k, v(k, :) < -3) = -3;
        v(k, v(k, :) > 3) = 3;

        x(k, :) = x(k, :) + v(k, :);
        x(k, x(k, :) < 0) = 0;
        x(k, x(k, :) > num_servers) = num_servers;
        x = round(x);

        T0=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,x(k,:),R_up,D,U);
        fitness(k) = func(x(k, :), T0,W,pt,pw);

        T1=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
        if fitness(k) < func(individual_best(k, :),T1,W,pt,pw)   
            individual_best(k, :) = x(k, :);
        end

        T1=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
        T2=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,global_best,R_up,D,U);
        if func(individual_best(k, :), T1,W,pt,pw) < func(global_best, T2,W,pt,pw)
            global_best = individual_best(k, :);
        end
        
        T1=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,individual_best(k,:),R_up,D,U);
        particle_fitness(k) = func(individual_best(k, :),T1,W,pt,pw);
    end

    iteration_variance(gen) = var(particle_fitness);
    
    % ========================================================
    % 🔥 核心替换：如果适应度方差小于0.04，进行 莱维飞行 变异
    % ========================================================
    if iteration_variance(gen) < 0.04
        beta = 1.5;
        sigma_u = (gamma(1+beta)*sin(pi*beta/2) / (gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
        u = randn(size(global_best)) * sigma_u;
        v_levy = randn(size(global_best));
        levy_step = u ./ (abs(v_levy).^(1/beta));
        
        % 加上莱维扰动 (缩放因子设为 0.5)
        cauchy_mutated_particle = global_best + 0.5 * levy_step .* global_best;
        
        cauchy_mutated_particle(cauchy_mutated_particle < 0) = 0;
        cauchy_mutated_particle(cauchy_mutated_particle > num_servers) = num_servers;
        cauchy_mutated_particle = round(cauchy_mutated_particle);
        
        T_cauchy = time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,cauchy_mutated_particle, R_up, D, U);
        fitness_cauchy = func(cauchy_mutated_particle, T_cauchy, W,pt,pw);
        
        if fitness_cauchy < func(global_best, T2, W,pt,pw)
            global_best = cauchy_mutated_particle;
        end
    end
    % ========================================================
   
    T3=time(r_eo,r_ec,f_local,f_mec_max,f_offsite_max,f_cloud,global_best,R_up,D,U);
    fitness_best_xiugai(gen + 1) = func(global_best, T3, W,pt,pw);
end  

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
    T=zeros(n_ue,4);
    for i = 1:n_ue 
        for j = 1:4 
            if j == 1
                T(i, j) = D(1,i)* 1e6/f_local;
            elseif j == 2                                                                                       
                T(i, j) = U(1,i)*8*1e3/R_up(1,i)+D(1,i)* 1e6/(f_mec_max/n1);
            elseif j == 3
                T(i, j) = U(1,i)*8*1e3/R_up(1,i)+U(1,i)/r_eo+D(1,i)* 1e6/(f_offsite_max/n2);
            elseif j == 4
                T(i, j) = U(1,i)*8*1e3/R_up(1,i)+U(1,i)/r_ec+D(1,i)* 1e6/f_cloud;
            end
        end
    end
end

%% 定义适应度函数
function res = func(x,T,W,pt,pw)
    rows = length(x);
    binaryMatrix = zeros(rows, 4);
    for i = 1:rows
        binaryMatrix(i, x(i) + 1) = 1;
    end
    result_sum1 = sum(binaryMatrix .* T, 'all');
    result_sum2 = sum(binaryMatrix .* W, 'all');
    res = pt*result_sum1+pw*result_sum2;
end

end