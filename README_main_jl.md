# main.jl - Neural ODE训练程序（Julia版本）

这个程序是 `fnn.py` 的Julia移植版本，用于训练Neural ODE模型拟合阿尔茨海默病生物标志物数据。

## 主要功能

1. **数据加载**：从Excel文件加载患者数据和分期信息
2. **DPS参数预设**：根据患者分期（CN, LMCI, AD）自动分配疾病进展评分（DPS）参数
   - CN: a=1, s ∈ [-10, 0]
   - LMCI: a=2, s ∈ [-2, 8]
   - AD: a=4, s ∈ [5, 20]
3. **Sigmoid拟合**：拟合4个生物标志物的sigmoid曲线
4. **Neural ODE训练**：
   - 阶段一：在sigmoid导数上预训练FNN
   - 阶段二：交替优化神经网络参数和DPS参数
5. **可视化**：绘制数据点、sigmoid拟合和Neural ODE轨迹

## 依赖包安装

```julia
using Pkg
Pkg.add([
    "DifferentialEquations",
    "Lux",
    "Optimization",
    "OptimizationOptimJL",
    "ComponentArrays",
    "NPZ",
    "JLD2",
    "Plots",
    "LsqFit",
    "XLSX",
    "DataFrames"
])
```

## 使用方法

### 方法1：使用真实数据

确保工作目录包含以下文件：
- `data.xlsx`: 患者数据（包含Sheet1）
- `rawdata.xlsx`: 患者分期信息（包含"ADNI Org."工作表）
- `mean_std.npy`: 标准化参数

然后运行：
```julia
julia main.jl
```

### 方法2：使用模拟数据

如果没有真实数据文件，程序会自动生成模拟数据：
```julia
julia main.jl
```

## 程序流程

1. **数据准备**
   - 加载患者数据和分期信息
   - 根据分期分配DPS参数（a, b）
   - 计算CN群体的平均初始值

2. **预训练阶段**
   - 拟合sigmoid曲线到人群数据
   - 计算sigmoid导数
   - 训练FNN拟合sigmoid导数

3. **交替训练阶段**（默认5轮外层迭代）
   - 固定DPS参数，优化神经网络参数（20次迭代）
   - 固定神经网络参数，优化每位患者的DPS参数（10次迭代）

4. **结果保存**
   - 模型参数保存到 `fnn_julia.jld2`
   - 图表保存到 `fnn_julia.png`

## 输出文件

- `fnn_julia.jld2`: 包含训练好的模型参数和DPS参数
- `fnn_julia.png`: 可视化结果图

## 与fnn.py的主要区别

1. **不包含不确定性量化（UQ）**：按照要求移除了不确定性分析
2. **交替训练**：实现了DPS参数和神经网络参数的交替优化
3. **语言差异**：使用Julia生态系统（Lux.jl, DifferentialEquations.jl）

## 参数调整

可以在 `main()` 函数中调整以下参数：

```julia
# 预训练迭代次数
n_epochs=100

# 交替训练参数
n_outer_epochs=5      # 外层迭代次数
n_nn_epochs=20        # 神经网络优化迭代次数
n_dps_epochs=10       # DPS参数优化迭代次数
```

## 注意事项

1. 首次运行可能需要较长时间编译Julia包
2. 如果ODE求解失败，程序会跳过该患者并继续训练
3. 建议使用真实数据以获得最佳结果
4. DPS参数的初始值是根据分期规则随机生成的，每次运行结果可能略有不同

## 故障排除

**问题**：无法加载数据文件  
**解决**：程序会自动切换到模拟数据模式

**问题**：ODE求解失败  
**解决**：减小 `n_dps_epochs` 或调整学习率

**问题**：内存不足  
**解决**：减小 `n_outer_epochs` 或减少患者数量
