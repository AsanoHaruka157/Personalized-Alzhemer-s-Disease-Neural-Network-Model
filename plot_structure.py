# -*- coding: utf-8 -*-
"""
绘制 PopulationGatedRNN 的详细结构图（竖版、比例适中）
- 展开 GatedRNNCell 的四个门（i, f, c~, o）
- 标注每个线性映射的维度与激活函数
- 标注输入/输出、Dropout、ODE 动力学与 A 参数
- 同时导出 SVG 和 PNG
"""

from graphviz import Digraph
from pathlib import Path

def build_population_gated_rnn_graph(
    hidden_dim: int = 32,
    num_layers: int = 2,
    filename: str = "PopulationGatedRNN_detailed",
    directory: str = ".",
    fmt: str = "svg",
):
    H = hidden_dim

    dot = Digraph("PopulationGatedRNN_detailed", format=fmt)
    # 纵向布局，避免又长又宽
    dot.attr(rankdir="TB")
    # 控制整体比例（可以自行微调）
    dot.graph_attr.update(
        size="7,9",       # 画布目标尺寸（英寸）；调整这里可改变“长宽”
        ratio="compress", # 根据内容自动压缩比例
        nodesep="0.3",
        ranksep="0.5",
        dpi="120"
    )
    dot.node_attr.update(
        shape="box",
        style="rounded,filled",
        fillcolor="white",
        color="#374151",
        fontname="Helvetica",
        fontsize="11"
    )
    # ---------- 输入 ----------
    dot.node("y", "y ∈ ℝ⁴ (4 biomarkers)", shape="folder", fillcolor="#EFF6FF")
    dot.node("s", "s ∈ ℝ¹ (scalar)", shape="folder", fillcolor="#EFF6FF")
    dot.node("concat_z", "concat([y, s]) → z ∈ ℝ⁵", fillcolor="#F3F4F6")
    dot.edge("y", "concat_z")
    dot.edge("s", "concat_z")

    # ---------- 输入层 ----------
    with dot.subgraph(name="cluster_input") as c:
        c.attr(label="Input block", color="#93C5FD", style="rounded")
        c.node("lin_in", f"Linear(5 → {H})", fillcolor="#E0F2FE")
        c.node("act_in", "Tanh", fillcolor="#E0F2FE")
        c.node("drop_in", "Dropout(p=0.1)", fillcolor="#E0F2FE")
        c.edge("lin_in", "act_in")
        c.edge("act_in", "drop_in")
    dot.edge("concat_z", "lin_in")

    dot.node("x0", f"x₀ ∈ ℝ¹×{H}", shape="note", fillcolor="#FEF3C7")
    dot.edge("drop_in", "x0")

    # 初始隐藏状态
    dot.node("h0", f"h₀ = zeros(1, {H})", shape="note", fillcolor="#FEF3C7")

    prev_x = "x0"
    prev_h = "h0"

    # ---------- 多层 GatedRNNCell ----------
    for i in range(1, num_layers + 1):
        with dot.subgraph(name=f"cluster_grnn_{i}") as c:
            c.attr(label=f"GatedRNNCell L{i}", color="#A7F3D0", style="rounded")

            # concat(x, h)
            c.node(f"concat_{i}", f"concat(x, h) ∈ ℝ¹×(2H)\n(此处 H={H})", fillcolor="#ECFDF5")

            # 四个门的线性层与激活（维度：2H→H）
            # input gate
            c.node(f"lin_i_{i}", f"Linear(2H → H)", fillcolor="#D1FAE5")
            c.node(f"sig_i_{i}", "Sigmoid → i_t", fillcolor="#D1FAE5")
            c.edge(f"lin_i_{i}", f"sig_i_{i}")

            # forget gate
            c.node(f"lin_f_{i}", f"Linear(2H → H)", fillcolor="#D1FAE5")
            c.node(f"sig_f_{i}", "Sigmoid → f_t", fillcolor="#D1FAE5")
            c.edge(f"lin_f_{i}", f"sig_f_{i}")

            # candidate
            c.node(f"lin_c_{i}", f"Linear(2H → H)", fillcolor="#D1FAE5")
            c.node(f"tanh_c_{i}", "Tanh → c̃_t", fillcolor="#D1FAE5")
            c.edge(f"lin_c_{i}", f"tanh_c_{i}")

            # output gate
            c.node(f"lin_o_{i}", f"Linear(2H → H)", fillcolor="#D1FAE5")
            c.node(f"sig_o_{i}", "Sigmoid → o_t", fillcolor="#D1FAE5")
            c.edge(f"lin_o_{i}", f"sig_o_{i}")

            # 更新方程（注意：你的实现中未使用 i_t, f_t；这里也标注“未参与更新”以与代码一致）
            c.node(
                f"update_{i}",
                "更新：h_t = o_t ⊙ tanh(c̃_t)\n(i_t, f_t 在当前实现中未参与 h_t)",
                fillcolor="#ECFDF5"
            )

            # 维度提示
            c.node(f"dim_{i}", f"x, h ∈ ℝ¹×{H}；concat(x,h) ∈ ℝ¹×(2H)", shape="note", fillcolor="#FEF3C7")

        # 连接到该层
        dot.edge(prev_x, f"concat_{i}", label="x", fontsize="10")
        dot.edge(prev_h, f"concat_{i}", label="h", fontsize="10")

        # 从 concat 到四个门
        dot.edge(f"concat_{i}", f"lin_i_{i}")
        dot.edge(f"concat_{i}", f"lin_f_{i}")
        dot.edge(f"concat_{i}", f"lin_c_{i}")
        dot.edge(f"concat_{i}", f"lin_o_{i}")

        # 门的输出到更新节点
        dot.edge(f"tanh_c_{i}", f"update_{i}")
        dot.edge(f"sig_o_{i}", f"update_{i}")
        # （可选）把没用到的门连到一个“未参与”提示
        dot.edge(f"sig_i_{i}", f"update_{i}", style="dashed", color="#9CA3AF")
        dot.edge(f"sig_f_{i}", f"update_{i}", style="dashed", color="#9CA3AF")

        # 本层输出 h_i
        dot.node(f"h{i}", f"h{i} ∈ ℝ¹×{H}", shape="note", fillcolor="#FEF3C7")
        dot.edge(f"update_{i}", f"h{i}", label="h_t", fontsize="10")

        # 下一层：x = h；h_prev = h
        prev_x = f"h{i}"
        prev_h = f"h{i}"

    # ---------- 输出层 ----------
    with dot.subgraph(name="cluster_output") as c:
        c.attr(label="Output block", color="#93C5FD", style="rounded")
        c.node("lin_out", f"Linear({H} → 4)", fillcolor="#E0F2FE")
        c.node("sig_out", "Sigmoid", fillcolor="#E0F2FE")
        c.node("drop_out", "Dropout(p=0.1)", fillcolor="#E0F2FE")
        c.edge("lin_out", "sig_out")
        c.edge("sig_out", "drop_out")
    dot.edge(prev_x, "lin_out")

    dot.node("net_out", "net_output ∈ ℝ⁴", shape="note", fillcolor="#FEF3C7")
    dot.edge("drop_out", "net_out")

    # ---------- ODE 动力学 ----------
    dot.node("Aparam", "A ∈ ℝ⁴（可学习参数）", shape="ellipse", fillcolor="#FAE8FF", color="#7C3AED")

    ode_label = (
        "ODE dynamics:\n"
        "f(y,s) = 1e-2 · σ(y) · (A − σ(y)) · net_output\n"
        "其中 σ 为 Sigmoid，逐分量作用于 y"
    )
    dot.node("ode", ode_label, fillcolor="#EDE9FE")
    dot.edge("y", "ode", label="σ(y)", fontsize="10")
    dot.edge("Aparam", "ode")
    dot.edge("net_out", "ode")

    dot.node("dy", "输出：f(y, s) = dy/dt ∈ ℝ⁴", shape="component", fillcolor="#E5E7EB")
    dot.edge("ode", "dy")

    # ---------- 渲染 ----------
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = (out_dir / filename).with_suffix(".svg")
    png_path = (out_dir / filename).with_suffix(".png")

    # 保存主格式
    svg_bytes = dot.pipe(format="svg")
    svg_path.write_bytes(svg_bytes)

    # 额外保存 PNG（便于插图）
    png_bytes = dot.pipe(format="png")
    png_path.write_bytes(png_bytes)

    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")
    return str(svg_path), str(png_path)

if __name__ == "__main__":
    # 按你的默认参数生成：hidden_dim=32, num_layers=2
    build_population_gated_rnn_graph(
        hidden_dim=32,
        num_layers=2,
        filename="PopulationGatedRNN_detailed",
        directory="."
    )

