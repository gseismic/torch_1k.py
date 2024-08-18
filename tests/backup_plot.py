
    def plot(self):
        # future: TODO
        import networkx as nx
        import matplotlib.pyplot as plt

        # 创建一个有向无环图
        DAG = nx.DiGraph()

        # 添加节点和边
        DAG.add_edges_from([(1, 2), (1, 3), (3, 4), (4, 5), (2, 5)])

        # 检查是否为DAG
        print("Is DAG:", nx.is_directed_acyclic_graph(DAG))

        # 使用 spring 布局绘制 DAG
        pos = nx.spring_layout(DAG)

        # 绘制节点
        nx.draw_networkx_nodes(DAG, pos, node_size=700, node_color="lightgreen")

        # 绘制边（使用箭头表示方向）
        nx.draw_networkx_edges(DAG, pos, edgelist=DAG.edges(), arrowstyle="->", arrowsize=20)

        # 绘制标签
        nx.draw_networkx_labels(DAG, pos, font_size=20, font_family='sans-serif')

        # 显示图形
        plt.title("Directed Acyclic Graph (DAG)")
        plt.axis('off')  # 关闭坐标轴
        plt.show()
