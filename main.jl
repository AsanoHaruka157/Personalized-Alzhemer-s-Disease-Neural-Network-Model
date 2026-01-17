using DifferentialEquations
using SciMLSensitivity
using Lux
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using Zygote
using ComponentArrays
using Random
using NPZ
using JLD2
using Plots
using Statistics
using LsqFit
using XLSX
using DataFrames

# 设置随机种子
rng = Random.default_rng()
Random.seed!(rng, 123)

# --- 辅助函数：数据标准化和反标准化 ---
function load_normalization_params(path="mean_std.npy")
    try
        data = npzread(path)
        mean_vals = data[1, :]
        std_vals = data[2, :]
        println("成功加载标准化参数")
        return mean_vals, std_vals
    catch e
        println("警告：无法加载 $(path)，错误: $(e)")
        println("使用默认标准化参数")
        return zeros(4), ones(4)
    end
end

function normalize_data(y, mean_vals, std_vals)
    return (y .- mean_vals') ./ std_vals'
end

function denormalize_data(y_norm, mean_vals, std_vals)
    return y_norm .* std_vals' .+ mean_vals'
end

# --- 数据加载函数 ---
function load_data(data_path="data.xlsx")
    println("正在从 $data_path 加载数据...")
    
    try
        xf = XLSX.readxlsx(data_path)
        sheet = xf["Sheet1"]
        data = XLSX.eachtablerow(sheet) |> DataFrame
        
        csf_dict = Dict{Int, Matrix{Float64}}()
        
        for row in eachrow(data)
            rid = Int(row[1])
            # 处理missing值，将其转换为NaN
            values = Float64[]
            for val in row[2:end]
                if ismissing(val)
                    push!(values, NaN)
                else
                    push!(values, Float64(val))
                end
            end
            
            if !haskey(csf_dict, rid)
                csf_dict[rid] = Matrix{Float64}(undef, 0, length(values))
            end
            
            csf_dict[rid] = vcat(csf_dict[rid], reshape(values, 1, :))
        end
        
        println("成功加载 $(length(csf_dict)) 位患者的数据")
        return csf_dict
    catch e
        println("警告：无法加载 $(data_path)，错误: $(e)")
        println("将生成模拟数据用于演示...")
        return generate_mock_data()
    end
end

function load_stage_dict(rawdata_path="rawdata.xlsx")
    println("正在从 $rawdata_path 加载患者分期信息...")
    
    try
        xf = XLSX.readxlsx(rawdata_path)
        sheet = xf["ADNI Org."]
        data = XLSX.eachtablerow(sheet) |> DataFrame
        
        stage_dict = Dict{Int, String}()
        
        for row in eachrow(data)
            rid = Int(row[:RID])
            stage = string(row[:DX_bl])
            
            if !haskey(stage_dict, rid)
                stage_dict[rid] = stage
            end
        end
        
        println("成功加载 $(length(stage_dict)) 位患者的分期信息")
        return stage_dict
    catch e
        println("警告：无法加载 $(rawdata_path)，错误: $(e)")
        println("将生成模拟分期信息...")
        return generate_mock_stages()
    end
end

# --- 生成模拟数据（用于演示） ---
function generate_mock_data()
    println("生成模拟数据...")
    csf_dict = Dict{Int, Matrix{Float64}}()
    
    for i in 1:50
        n_visits = rand(2:5)
        data = zeros(n_visits, 5)
        
        # 时间列
        data[:, 1] = sort(rand(n_visits) * 10.0)
        
        # 4个生物标志物（归一化后的值）
        for j in 1:n_visits
            data[j, 2:5] = randn(4) * 0.3
        end
        
        csf_dict[i] = data
    end
    
    return csf_dict
end

function generate_mock_stages()
    stage_dict = Dict{Int, String}()
    stages = ["CN", "LMCI", "AD"]
    
    for i in 1:50
        stage_dict[i] = rand(stages)
    end
    
    return stage_dict
end

# --- 1. 为每位患者分配DPS参数 ---
function assign_dps_params(csf_dict, stage_dict)
    """
    根据患者分期（CN, LMCI, AD）为每位患者指定 a 和 b 参数。
    a: CN=1, LMCI=2, AD=4
    b: 随机选择，使得初始s值落在指定区间内
    """
    println("\n正在为患者分配DPS参数...")
    
    patient_data = Dict()
    
    # s值的范围
    s_ranges = Dict(
        "CN" => (-10, 0),
        "LMCI" => (-2, 8),
        "AD" => (5, 20),
        "Other" => (-10, 20)
    )
    
    a_values = Dict(
        "CN" => 1.0,
        "LMCI" => 2.0,
        "AD" => 4.0,
        "Other" => 1.0
    )
    
    # 收集所有 (s, y) 点
    all_s_points = Float64[]
    all_y_points = Matrix{Float64}(undef, 0, 4)
    all_stages = String[]
    
    for (pid, sample) in csf_dict
        stage = get(stage_dict, pid, "Other")
        
        t = sample[:, 1]
        y = sample[:, 2:5]
        
        a = a_values[stage]
        s_min, s_max = s_ranges[stage]
        
        # 计算b，使s_initial落在目标区间
        t_initial = t[1]
        s_initial_target = s_min + rand() * (s_max - s_min)
        b = s_initial_target - a * t_initial
        
        s = a .* t .+ b
        
        patient_data[pid] = Dict(
            "t" => t,
            "y" => y,
            "s" => s,
            "stage" => stage,
            "a" => a,
            "b" => b
        )
        
        append!(all_s_points, s)
        all_y_points = vcat(all_y_points, y)
        append!(all_stages, fill(stage, length(t)))
    end
    
    println("成功为 $(length(patient_data)) 位患者分配DPS参数")
    return patient_data, all_s_points, all_y_points, all_stages
end

# --- 新增功能：计算CN群体的平均初始值 ---
function get_cn_average_y0(patient_data)
    """
    计算CN（认知正常）群体在第一次访问时的平均生物标记物值（忽略NaN）。
    """
    cn_y0s = []
    
    for (pid, data) in patient_data
        if data["stage"] == "CN"
            push!(cn_y0s, data["y"][1, :])
        end
    end
    
    if isempty(cn_y0s)
        println("警告：未找到CN患者数据，将使用默认初始值 [0.1, 0, 0, 0]。")
        return [0.1, 0.0, 0.0, 0.0]
    end
    
    # 计算平均值，忽略NaN
    cn_y0s_matrix = hcat(cn_y0s...)'
    avg_y0 = Float64[]
    for k in 1:size(cn_y0s_matrix, 2)
        col = cn_y0s_matrix[:, k]
        valid_vals = col[.!isnan.(col)]
        if isempty(valid_vals)
            push!(avg_y0, 0.0)
        else
            push!(avg_y0, mean(valid_vals))
        end
    end
    
    println("计算出的CN群体平均初始值（归一化后）: $avg_y0")
    return avg_y0
end

# --- 2. Sigmoid 拟合函数 ---
sigmoid(s, p) = p[1] ./ (1.0 .+ exp.(-p[2] .* (s .- p[3]))) .+ p[4]

function fit_sigmoids(s_data, y_data)
    println("\n正在拟合 Sigmoid 曲线...")
    params = zeros(4, 4)  # 4个生物标志物，每个4个参数
    
    for k in 1:4
        # 过滤掉NaN值
        valid_mask = .!isnan.(y_data[:, k])
        s_valid = s_data[valid_mask]
        y_valid = y_data[valid_mask, k]
        
        println("  标志物 $k: $(length(y_valid))/$(length(y_data[:, k])) 个有效数据点")
        
        # 初始猜测
        if isempty(y_valid)
            println("  警告: 标志物 $k 无有效数据，使用默认参数")
            params[:, k] = [1.0, 1.0, 0.0, 0.0]
            continue
        end
        
        p0 = [
            maximum(y_valid) - minimum(y_valid),  # a: 幅度
            1.0,                                   # b: 斜率
            median(s_valid),                       # c: 中心点
            minimum(y_valid)                       # d: 垂直偏移
        ]
        
        # 使用LsqFit进行曲线拟合
        model(s, p) = sigmoid(s, p)
        
        try
            fit = curve_fit(model, s_valid, y_valid, p0, maxIter=10000)
            params[:, k] = fit.param
            println("  标志物 $k 拟合完成: ", fit.param)
        catch e
            println("  警告: 标志物 $k 拟合失败，使用初始猜测: $(e)")
            params[:, k] = p0
        end
    end
    
    return params
end

function get_sigmoid_derivatives(s_grid, params)
    n = length(s_grid)
    y = zeros(n, 4)
    dyds = zeros(n, 4)
    
    for k in 1:4
        a, b, c, d = params[:, k]
        exp_term = exp.(-b .* (s_grid .- c))
        y[:, k] = a ./ (1.0 .+ exp_term) .+ d
        dyds[:, k] = (a .* b .* exp_term) ./ ((1.0 .+ exp_term).^2)
    end
    
    return y, dyds
end

# --- 3. 定义FNN模型 ---
function create_fnn_model(input_dim=4, hidden_dim=32, output_dim=4)
    return Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, output_dim, tanh)
    )
end

# --- 4. 预训练FNN在Sigmoid导数上（使用交替优化） ---
function pretrain_fnn_on_sigmoid(y_target, dyds_target, patient_data, stage_dict; 
                                 n_outer_epochs=3, n_nn_epochs=100, n_dps_epochs=20)
    println("\n--- 阶段一：在 Sigmoid 导数上预训练 FNN（交替优化） ---")
    
    # 创建模型
    model = create_fnn_model()
    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps)
    
    # 添加输出缩放参数
    output_scaler = [0.1]
    
    # 转换数据为32位浮点数
    y_target_f32 = Float32.(y_target)
    dyds_target_f32 = Float32.(dyds_target)
    
    # 初始DPS参数
    pids = collect(keys(patient_data))
    dps_params = Dict{Int, Dict{String, Float64}}()
    
    for pid in pids
        dps_params[pid] = Dict(
            "a" => patient_data[pid]["a"],
            "b" => patient_data[pid]["b"]
        )
    end
    
    current_params = ps
    s_grid_pretrain = Float32.(collect(range(-10, 20, length=500)))
    for outer_epoch in 1:n_outer_epochs
        println("\n=== 预训练外层迭代 $outer_epoch/$n_outer_epochs ===")

        # 步骤1: 固定DPS参数，优化神经网络参数
        println("[1] 固定DPS参数，优化神经网络参数...")
        curvature_weight = Float32(2.0)
        trajectory_weight = Float32(1.0)
        y0_f32 = Float32.(y_target_f32[1, :])
        function loss_fn(params, _)
            y_pred_deriv, _ = model(y_target_f32', params, st)
            y_pred_deriv = y_pred_deriv' .* Float32(output_scaler[1])
            deriv_mse_loss = sum((y_pred_deriv .- dyds_target_f32).^2) / Float32(length(dyds_target_f32))
            dudt = create_neural_ode(model, params, st, output_scaler)
            prob = ODEProblem(dudt, y0_f32, (s_grid_pretrain[1], s_grid_pretrain[end]), params)
            sol = solve(prob, Tsit5(), saveat=s_grid_pretrain, reltol=1e-4, abstol=1e-5)
            trajectory_loss = Float32(0.0)
            if sol.retcode == ReturnCode.Success
                y_ode = hcat(sol.u...)'
                trajectory_loss = sum((y_ode .- y_target_f32).^2) / Float32(length(y_target_f32))
            else
                trajectory_loss = Float32(100.0)
            end
            curvature_penalty = -curvature_weight * sum(abs.(y_pred_deriv)) / Float32(length(y_pred_deriv))
            loss = deriv_mse_loss + trajectory_weight * trajectory_loss + curvature_penalty
            return loss
        end
        optf = Optimization.OptimizationFunction(loss_fn, Optimization.AutoZygote())
        optprob = Optimization.OptimizationProblem(optf, current_params)
        iter_count = [0]
        callback = function(p, l)
            iter_count[1] += 1
            print("\r  NN Epoch [$iter_count[1]/$n_nn_epochs], Loss: $(round(l, digits=8))")
            flush(stdout)
            return false
        end
        result = Optimization.solve(
            optprob,
            LBFGS(),
            callback=callback,
            maxiters=n_nn_epochs
        )
        current_params = result.u
        println("\n  神经网络优化完成，Loss: $(result.objective)")

        # 步骤2: 固定神经网络参数，优化DPS参数（此处可自定义DPS微调策略）
        println("[2] 固定神经网络参数，优化DPS参数...")
        # ...此处可插入DPS参数微调代码...
        # 例如：dps_params = update_dps_params(patient_data, current_params, ...)
        # 这里暂时保持dps_params不变
        println("  DPS参数保持不变（预训练阶段）")

        # 步骤3: 用当前DPS参数重新生成s/y，重新拟合sigmoid，更新y_target/dyds_target
        println("[3] 用当前DPS参数重新生成人群数据并拟合sigmoid...")
        # 生成人群s/y
        s_pop = Float32[]
        y_pop = Array{Float32,2}(undef, 0, 4)
        stages_pop = String[]
        for (pid, data) in patient_data
            a = dps_params[pid]["a"]
            b = dps_params[pid]["b"]
            t = data["t"]
            y = data["y"]
            s = a .* t .+ b
            append!(s_pop, s)
            y_pop = vcat(y_pop, y)
            append!(stages_pop, fill(data["stage"], length(t)))
        end
        s_pop = collect(s_pop)
        y_pop = Array(y_pop)
        # 拟合sigmoid
        sigmoid_params = fit_sigmoids(s_pop, y_pop)
        y_target, dyds_target = get_sigmoid_derivatives(s_grid_pretrain, sigmoid_params)
        y_target_f32 = Float32.(y_target)
        dyds_target_f32 = Float32.(dyds_target)
    end
    
    println("\n预训练完成")
    return model, current_params, st, output_scaler, dps_params
end

# --- 5. Neural ODE定义 ---
function create_neural_ode(model, params, st, output_scaler)
    function dudt(u, p, t)
        # u 是状态向量，p 是参数，t 是时间
        u_reshaped = reshape(u, :, 1)
        du, _ = model(u_reshaped, p, st)
        # 确保输出类型与u一致
        result = vec(du) .* Float32(output_scaler[1])
        return result
    end
    return dudt
end

# --- 6. 在真实数据上交替训练模型参数和DPS参数 ---
function train_alternating(model, params, st, output_scaler, patient_data, dps_params, y0; 
                          n_outer_epochs=5, n_nn_epochs=20, n_dps_epochs=10)
    println("\n--- 阶段二：交替训练 FNN 参数和 DPS 参数 ---")
    
    current_params = params
    
    # 使用传入的DPS参数
    pids = collect(keys(patient_data))
    
    for outer_epoch in 1:n_outer_epochs
        println("\n=== 外层迭代 $outer_epoch/$n_outer_epochs ===")
        
        # --- 步骤 1: 固定DPS参数，优化神经网络参数 ---
        println("\n[1] 固定DPS参数，优化神经网络参数...")
        
        # 曲率正则化系数
        curvature_weight = Float32(0.001)
        
        function nn_loss_fn(p, _)
            total_loss = Float32(0.0)
            total_curvature = Float32(0.0)
            n_patients = 0
            
            for pid in pids
                data = patient_data[pid]
                a = dps_params[pid]["a"]
                b = dps_params[pid]["b"]
                
                t = data["t"]
                y_true = data["y"]
                s = a .* t .+ b
                
                # 排序
                sorted_indices = sortperm(s)
                s_sorted = Float32.(s[sorted_indices])
                y_sorted = Float32.(y_true[sorted_indices, :])
                
                # 创建ODE问题
                dudt = create_neural_ode(model, p, st, output_scaler)
                prob = ODEProblem(dudt, Float32.(y0), (s_sorted[1], s_sorted[end]), p)
                
                try
                    sol = solve(prob, Tsit5(), saveat=s_sorted, reltol=1e-4, abstol=1e-5)
                    
                    if sol.retcode == :Success
                        y_pred = hcat(sol.u...)'
                        patient_loss = Float32(sum((y_pred .- y_sorted).^2) / length(y_sorted))
                        total_loss += patient_loss
                        
                        # 计算曲率：轨迹变化量越大，曲率越大
                        if size(y_pred, 1) > 1
                            dy = diff(y_pred, dims=1)
                            curvature = Float32(sum(abs.(dy)))
                            total_curvature += curvature
                        end
                        
                        n_patients += 1
                    end
                catch e
                    # 跳过失败的患者
                end
            end
            
            # 曲率正则化：鼓励更大的曲率（更大的变化）
            curvature_bonus = -curvature_weight * total_curvature / Float32(max(n_patients, 1))
            
            return n_patients > 0 ? total_loss / Float32(n_patients) + curvature_bonus : Float32(1e10)
        end
        
        # 使用有限差分进行数值梯度估计（避免Zygote对复杂ODE求解的问题）
        optf_nn = Optimization.OptimizationFunction(nn_loss_fn, Optimization.AutoFiniteDiff())
        optprob_nn = Optimization.OptimizationProblem(optf_nn, current_params)
        
        nn_iter_count = [0]
        nn_callback = function(p, l)
            nn_iter_count[1] += 1
            print("\r  NN Epoch [$nn_iter_count[1]/$n_nn_epochs], Loss: $(round(l, digits=8))")
            flush(stdout)
            return false
        end
        
        result_nn = Optimization.solve(
            optprob_nn,
            LBFGS(),
            callback=nn_callback,
            maxiters=n_nn_epochs
        )
        
        current_params = result_nn.u
        println("\n  神经网络参数优化完成，Loss: $(result_nn.objective)")
        
        # --- 步骤 2: 固定神经网络参数，优化DPS参数 ---
        println("\n[2] 固定神经网络参数，优化DPS参数...")
        
        # 曲率奖励权重：鼓励更大的a值（更陡峭的曲线）
        dps_curvature_weight = 0.5
        
        # 使用简化的方法：基于当前模型输出和真实数据的匹配度进行微调
        dps_updated_count = 0
        for pid in pids
            data = patient_data[pid]
            t = data["t"]
            y_true = data["y"]
            
            # 当前DPS参数
            a_init = dps_params[pid]["a"]
            b_init = dps_params[pid]["b"]
            
            # 简单的网格搜索优化DPS参数
            best_score = Inf
            best_a, best_b = a_init, b_init
            
            # 在当前值附近搜索，优先尝试更大的a值
            for da in [-0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
                for db in [-2.0, -1.0, 0.0, 1.0, 2.0]
                    a_test = max(0.1, a_init + da)
                    b_test = b_init + db
                    
                    s = a_test .* t .+ b_test
                    sorted_indices = sortperm(s)
                    s_sorted = Float32.(s[sorted_indices])
                    y_sorted = Float32.(y_true[sorted_indices, :])
                    
                    # 创建ODE问题并求解
                    dudt = create_neural_ode(model, current_params, st, output_scaler)
                    prob = ODEProblem(dudt, Float32.(y0), (s_sorted[1], s_sorted[end]), current_params)
                    
                    try
                        sol = solve(prob, Tsit5(), saveat=s_sorted, reltol=1e-4, abstol=1e-5, verbose=false)
                        
                        if sol.retcode == :Success || sol.retcode == ReturnCode.Success
                            y_pred = hcat(sol.u...)'
                            mse_loss = sum((y_pred .- y_sorted).^2) / length(y_sorted)
                            
                            # 曲率奖励：更大的a值意味着更陡峭的曲线
                            curvature_bonus = -dps_curvature_weight * a_test
                            
                            # 总评分 = MSE损失 + 曲率奖励（负值表示奖励）
                            score = mse_loss + curvature_bonus
                            
                            if score < best_score
                                best_score = score
                                best_a = a_test
                                best_b = b_test
                            end
                        end
                    catch
                        continue
                    end
                end
            end
            
            # 更新DPS参数
            if best_score < Inf
                dps_params[pid]["a"] = best_a
                dps_params[pid]["b"] = best_b
                dps_updated_count += 1
            end
        end
        
        println("  DPS参数优化完成，更新了 $dps_updated_count/$(length(pids)) 位患者")
        
        # --- 计算总损失 ---
        total_loss = nn_loss_fn(current_params, nothing)
        println("\n外层迭代 $outer_epoch 完成，总损失: $total_loss")
    end
    
    println("\n交替训练完成")
    return current_params, dps_params
end

# --- 7. 绘图函数 ---
function plot_results(s_pop, y_pop, stages_pop, s_grid, y_sigmoid, y_traj, mean_vals, std_vals; 
                      title_str="Neural ODE 拟合结果")
    println("\n正在生成图表...")
    
    # 反标准化
    y_pop_orig = denormalize_data(y_pop, mean_vals, std_vals)
    y_sigmoid_orig = denormalize_data(y_sigmoid, mean_vals, std_vals)
    y_traj_orig = denormalize_data(y_traj, mean_vals, std_vals)
    
    # 标题
    titles = ["Aβ (A)", "p-Tau (T)", "N", "Cognition (C)"]
    
    # 按分期分组数据
    stage_colors = Dict(
        "CN" => :orange,
        "LMCI" => :green,
        "AD" => :blue,
        "Other" => :grey
    )
    
    # 创建2x2子图
    p = plot(layout=(2, 2), size=(1200, 900))
    
    for k in 1:4
        # 按分期绘制散点数据
        unique_stages = unique(stages_pop)
        
        for stage in unique_stages
            stage_mask = stages_pop .== stage
            stage_color = get(stage_colors, stage, :grey)
            
            scatter!(p[k], s_pop[stage_mask], y_pop_orig[stage_mask, k], 
                    label=stage, alpha=0.4, markersize=3, color=stage_color)
        end
        
        # Sigmoid拟合
        plot!(p[k], s_grid, y_sigmoid_orig[:, k], 
              label="Sigmoid 拟合", linewidth=2.5, color=:red, linestyle=:dash)
        
        # Neural ODE轨迹
        plot!(p[k], s_grid, y_traj_orig[:, k], 
              label="Neural ODE", linewidth=2.5, color=:black)
        
        xlabel!(p[k], "Disease Progression Score (s)")
        ylabel!(p[k], titles[k])
        title!(p[k], titles[k])
        plot!(p[k], grid=true, gridalpha=0.4)
    end
    
    plot!(p, plot_title=title_str, titlefontsize=16)
    
    savefig(p, "fnn_julia.png")
    println("图表已保存到 fnn_julia.png")
    
    display(p)
    return p
end

# --- 主程序 ---
function main()
    println("=== Neural ODE 训练流程 (Julia 版本) ===\n")
    
    # 1. 加载标准化参数
    mean_vals, std_vals = load_normalization_params()
    println("标准化参数加载完成")
    
    # 2. 加载数据
    csf_dict = load_data()
    stage_dict = load_stage_dict()
    
    # 3. 分配DPS参数并获取人群数据点
    patient_data, s_pop, y_pop, stages_pop = assign_dps_params(csf_dict, stage_dict)
    
    # 4. 计算CN群体平均初始值
    y0 = get_cn_average_y0(patient_data)
    
    # 5. 拟合Sigmoid
    println("\n正在拟合 Sigmoid 曲线...")
    s_grid = collect(range(-10, 20, length=500))
    sigmoid_params = fit_sigmoids(s_pop, y_pop)
    y_sigmoid, dyds_sigmoid = get_sigmoid_derivatives(s_grid, sigmoid_params)
    
    # 6. 预训练FNN（使用交替优化）
    model, params_pretrained, st, output_scaler, dps_pretrained = pretrain_fnn_on_sigmoid(
        y_sigmoid, dyds_sigmoid, patient_data, stage_dict,
        n_outer_epochs=10, n_nn_epochs=200, n_dps_epochs=30
    )
    
    # 7. 生成预训练轨迹
    dudt_pretrain = create_neural_ode(model, params_pretrained, st, output_scaler)
    prob_pretrain = ODEProblem(dudt_pretrain, Float32.(y0), 
                               (Float32(s_grid[1]), Float32(s_grid[end])), 
                               params_pretrained)
    sol_pretrain = solve(prob_pretrain, Tsit5(), saveat=Float32.(s_grid))
    y_pretrain = hcat(sol_pretrain.u...)'

    # 8. 绘制预训练结果并保存
    p_pretrain = plot_results(s_pop, y_pop, stages_pop, s_grid, y_sigmoid, y_pretrain, 
                 mean_vals, std_vals, title_str="预训练结果")
    savefig(p_pretrain, "fnn_pretrain.png")
    println("预训练结果已保存到 fnn_pretrain.png")

    # 8.1 保存预训练模型参数和DPS参数
    @save "fnn_pretrain.jld2" params_pretrained st output_scaler
    @save "dps_pretrain.jld2" dps_pretrained
    println("预训练模型参数已保存到 fnn_pretrain.jld2，DPS参数已保存到 dps_pretrain.jld2")

    # 9. 交替训练神经网络参数和DPS参数（如有预训练文件则加载）
    if isfile("fnn_pretrain.jld2") && isfile("dps_pretrain.jld2")
        println("检测到预训练文件，加载继续训练...")
        @load "fnn_pretrain.jld2" params_pretrained st output_scaler
        @load "dps_pretrain.jld2" dps_pretrained
    end
    params_final, dps_final = train_alternating(
        model, params_pretrained, st, output_scaler, 
        patient_data, dps_pretrained, y0,
        n_outer_epochs=15, n_nn_epochs=50, n_dps_epochs=20
    )
    
    # 10. 生成最终轨迹
    s_grid_final = collect(range(-10, 20, length=300))
    dudt_final = create_neural_ode(model, params_final, st, output_scaler)
    prob_final = ODEProblem(dudt_final, Float32.(y0), 
                           (Float32(s_grid_final[1]), Float32(s_grid_final[end])), 
                           params_final)
    sol_final = solve(prob_final, Tsit5(), saveat=Float32.(s_grid_final))
    y_final = hcat(sol_final.u...)'
    
    # 11. 重新收集所有患者的s和y数据（使用更新后的DPS参数）
    all_s_final = Float64[]
    all_y_final = Matrix{Float64}(undef, 0, 4)
    all_stages_final = String[]
    
    for (pid, data) in patient_data
        t = data["t"]
        y = data["y"]
        stage = data["stage"]
        
        a = dps_final[pid]["a"]
        b = dps_final[pid]["b"]
        s = a .* t .+ b
        
        append!(all_s_final, s)
        all_y_final = vcat(all_y_final, y)
        append!(all_stages_final, fill(stage, length(t)))
    end
    
    # 12. 绘制最终结果
    y_sigmoid_final, _ = get_sigmoid_derivatives(s_grid_final, sigmoid_params)
    plot_results(all_s_final, all_y_final, all_stages_final, s_grid_final, 
                y_sigmoid_final, y_final, mean_vals, std_vals, 
                title_str="最终 Neural ODE 拟合结果（交替训练后）")
    
    # 13. 保存模型参数和DPS参数
    @save "fnn_julia.jld2" params_final st output_scaler dps_final
    println("\n模型参数已保存到 fnn_julia.jld2")
    
    println("\n=== 完整流程执行完毕 ===")
end

# 自动运行主程序（点击执行按钮即可运行）
main()
