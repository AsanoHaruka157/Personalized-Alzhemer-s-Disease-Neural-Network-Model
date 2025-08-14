# -*- coding: utf-8 -*-
# Vertical (top→bottom) Graphviz rendering for your PopulationODE + ResidualBlock
# Requirements: system Graphviz + `pip install graphviz`

from graphviz import Digraph
from typing import Literal

def build_population_ode_graph_vertical(
    hidden_dim: int = 32,
    num_residual_blocks: int = 2,
    fmt: Literal["png", "svg", "pdf"] = "png",
    filename: str = "population_ode_vertical",
):
    g = Digraph("PopulationODE", format=fmt, filename=filename)
    # 竖版：自上而下
    g.attr(rankdir="TB", labelloc="t",
           label=f"PopulationODE  (hidden_dim={hidden_dim}, num_residual_blocks={num_residual_blocks})",
           fontsize="12", outputorder="edgesfirst",
           nodesep="0.25", ranksep="0.35")  # 更紧凑
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#FFFFFF", fontsize="11")
    g.attr("edge", arrowsize="0.8")

    # ---------- Inputs ----------
    g.node("Yin", "y ∈ R^4")
    g.node("Sin", "s ∈ R^1")
    g.node("Concat", "concat → z ∈ R^5", fillcolor="#EEF7FF")
    # 让 y,s 并排同一层
    g.body.append("{rank=same; Yin; Sin}")
    g.edge("Yin", "Concat")
    g.edge("Sin", "Concat")

    # ---------- 1) Input Layer ----------
    with g.subgraph(name="cluster_input") as c:
        c.attr(label="1) Input Layer", color="#C7DFFF", style="rounded")
        c.node("In_Linear", f"Linear(5→{hidden_dim})")
        c.node("In_Tanh", "Tanh")
        c.node("In_Drop", "Dropout(p=0.1)")
        c.edge("In_Linear", "In_Tanh")
        c.edge("In_Tanh", "In_Drop")
    g.edge("Concat", "In_Linear")

    prev = "In_Drop"

    # ---------- 2) Residual Blocks ----------
    with g.subgraph(name="cluster_res") as ctop:
        ctop.attr(label=f"2) Residual Blocks ×{num_residual_blocks} (dim={hidden_dim})",
                  color="#FFD6A5", style="rounded")

        for i in range(1, num_residual_blocks+1):
            p = f"RB{i}"
            with ctop.subgraph(name=f"cluster_{p}") as cb:
                cb.attr(label=f"ResidualBlock #{i}", color="#FFB870", style="rounded")
                cb.node(f"{p}_L1", f"Linear({hidden_dim}→{hidden_dim})")
                cb.node(f"{p}_Act", "Tanh")
                cb.node(f"{p}_Drop", "Dropout(p=0.1)")
                cb.node(f"{p}_L2", f"Linear({hidden_dim}→{hidden_dim})")

                cb.edge(f"{p}_L1", f"{p}_Act")
                cb.edge(f"{p}_Act", f"{p}_Drop")
                cb.edge(f"{p}_Drop", f"{p}_L2")

                # 残差：x + block(x)
                cb.node(f"{p}_X", f"{p}: x", shape="ellipse", fillcolor="#FFFFFF")
                cb.node(f"{p}_ADD", "add: x + block(x)", shape="diamond", fillcolor="#FFF0E6")

            # 主干进入 Block
            g.edge(prev, f"{p}_L1")

            # skip：把输入 x 并行送到加法
            # 为避免拉长布局，使用 constraint=false 的虚线边
            g.edge(prev, f"{p}_X", style="dashed", label="x", constraint="false")
            g.edge(f"{p}_L2", f"{p}_ADD", label="block(x)")
            g.edge(f"{p}_X", f"{p}_ADD", style="dashed", label="x", constraint="false")

            prev = f"{p}_ADD"

    # ---------- 3) Output Layer ----------
    with g.subgraph(name="cluster_out") as c:
        c.attr(label="3) Output Layer", color="#B7EFC5", style="rounded")
        c.node("Out_Linear", f"Linear({hidden_dim}→4)")
        c.node("Out_Sigmoid", "Sigmoid")
        c.node("Out_Drop", "Dropout(p=0.1)")
        c.edge("Out_Linear", "Out_Sigmoid")
        c.edge("Out_Sigmoid", "Out_Drop")

    g.edge(prev, "Out_Linear")
    g.node("NetOut", "net_output ∈ R^4", fillcolor="#F0FFF4")
    g.edge("Out_Drop", "NetOut")

    # ---------- Learnable Param A ----------
    g.node("ParamA", "A ∈ R^4 (learnable, init = zeros)", fillcolor="#FFF7CC")

    # ---------- ODE RHS ----------
    g.node("SigY", "σ(y)", fillcolor="#F0FFF4")
    g.node("Aminus", "A - σ(y)", fillcolor="#F0FFF4")
    g.node("Scale", "× 1e-2", fillcolor="#F0FFF4")
    g.node("ODE", "f(y,s) = 1e-2 · σ(y) · (A - σ(y)) · net_output", fillcolor="#EAFDEA")

    # 放置关系（不拉长主干）
    g.body.append("{rank=same; ParamA; SigY; NetOut}")
    g.edge("NetOut", "ODE", constraint="false")
    g.edge("SigY", "Aminus", constraint="false")
    g.edge("ParamA", "Aminus", constraint="false")
    g.edge("Aminus", "ODE", constraint="false")
    g.edge("Scale", "ODE", constraint="false")

    # y → σ(y)（虚线）
    g.edge("Yin", "SigY", style="dashed", constraint="false")

    return g

if __name__ == "__main__":
    g = build_population_ode_graph_vertical(hidden_dim=32, num_residual_blocks=2, fmt="png",
                                            filename="population_ode_vertical")
    path = g.render(cleanup=True, view=False)
    print("Saved to:", path)
    # SVG（更清晰可缩放）：
    # g = build_population_ode_graph_vertical(fmt="svg", filename="population_ode_vertical_svg")
    # print("Saved to:", g.render(cleanup=True, view=False))
