# ========================
# Imports and Dependencies
# ========================
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ast
import re
import random
import numpy as np

# ========================
# TreeRAG: Hierarchical Text Structure Management and Visualization
# ========================
class TreeRAG:
    def __init__(self, path1, path2=None, show_plots=True):
        self.path1 = path1
        self.show_plots = show_plots
        self.forest = {}
        self.load_data()
        self.build_forest()
        if self.show_plots:
            self.plot_forest()

    # --------------------
    # Load and Prepare Data
    # --------------------
    def load_data(self):
        self.df1 = pd.read_csv(self.path1)
        self.merged_df = self.df1

    def processing_text(self, s):
        data = ast.literal_eval(s)
        return data[0][0]

    def clean_text(self, text):
        text = text.replace('\xa0', ' ')
        text = text.replace('\xad', '')
        text = re.sub(r'\n+', '\n\n', text)
        return text.strip()

    # --------------------
    # Build Hierarchical Forest Structure
    # --------------------
    def build_forest(self):
        for _, row in self.merged_df.iterrows():
            chapter = row['Chapter'].lower()
            section = self.clean_text(row['Big_Chunk_Text'])
            chunk = self.clean_text(row['Chunk_Text'])
            self.forest.setdefault(chapter, {}).setdefault(section, []).append(chunk)

    # --------------------
    # Plot Forest Hierarchy (2D)
    # --------------------
    def plot_forest(self):
        G = nx.DiGraph()
        for chapter, sections in self.forest.items():
            G.add_node(chapter, label=chapter)
            for section, chunks in sections.items():
                G.add_node(section, label=section)
                G.add_edge(chapter, section)
                for i, chunk in enumerate(chunks, 1):
                    chunk_node = f"{section}_chunk{i}"
                    G.add_node(chunk_node, label=chunk[:5] + "...")
                    G.add_edge(section, chunk_node)

        plt.figure(figsize=(20, 10))
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_color="lightblue",
                node_size=10, font_size=5, edge_color="gray", arrows=True)
        plt.title("Hierarchical Tree of Chapters, Sections, and Phrases")
        plt.show()

    # --------------------
    # Plot Forest with Highlighted Path
    # --------------------
    def plot_forest_with_highlight(self, selected_chapter, selected_section, selected_chunk):
        G = nx.DiGraph()
        selected_nodes = set()

        if selected_chapter in self.forest:
            selected_nodes.add(selected_chapter)

        for chapter, sections in self.forest.items():
            G.add_node(chapter, label=chapter)
            for sec_index, (section, chunks) in enumerate(sections.items(), 1):
                section_label = f"Section {sec_index}"
                G.add_node(section, label=section_label)
                G.add_edge(chapter, section)
                if chapter == selected_chapter and section == selected_section:
                    selected_nodes.add(section)

                for chunk_index, chunk in enumerate(chunks, 1):
                    chunk_node = f"{section}_chunk{chunk_index}"
                    chunk_label = f"Chunk {chunk_index}"
                    G.add_node(chunk_node, label=chunk_label)
                    G.add_edge(section, chunk_node)
                    if chapter == selected_chapter and section == selected_section and chunk == selected_chunk:
                        selected_nodes.add(chunk_node)

        node_colors = ["red" if node in selected_nodes else "lightblue" for node in G.nodes()]
        edge_colors = ["red" if edge[0] in selected_nodes and edge[1] in selected_nodes else "gray" for edge in G.edges()]

        plt.figure(figsize=(20, 10))
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        labels = nx.get_node_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors,
                node_size=50, font_size=5, edge_color=edge_colors, arrows=True)
        plt.title("Hierarchical Tree with Highlighted Selected Path")
        plt.show()

    # --------------------
    # Accessors for Tree Data
    # --------------------
    def get_phrases_by_chapter(self, chapter_name):
        return self.forest.get(chapter_name, {})

    def get_phrases_by_section(self, chapter_name, section_name):
        chapter = {k.strip().lower(): v for k, v in self.forest.items()}.get(chapter_name.strip().lower())
        if not chapter:
            print(f"Chapter '{chapter_name}' not found.")
            return []

        section = {k.strip().lower(): v for k, v in chapter.items()}.get(section_name.strip().lower())
        if not section:
            print(f"Section '{section_name}' not found in chapter '{chapter_name}'.")
            return []

        return section

    def get_section_by_chapter(self, chapter):
        normalized_chapter = chapter.strip().lower()
        for ch in self.forest:
            if ch.strip().lower() == normalized_chapter:
                return list(self.forest[ch].keys())
        return []

    def get_forest(self):
        return self.forest

    # --------------------
    # Plot Forest in 3D (Subset)
    # --------------------
    def plot_3d_forest(self):
        G_sub = nx.DiGraph()
        for chapter, sections in self.forest.items():
            G_sub.add_node(chapter)
            selected_sections = random.sample(list(sections.keys()), max(1, len(sections) // 2))
            for section in selected_sections:
                G_sub.add_node(section)
                G_sub.add_edge(chapter, section)
                for i, chunk in enumerate(sections[section][:max(1, len(sections[section]) // 5)], 1):
                    chunk_node = f"{section}_chunk{i}"
                    G_sub.add_node(chunk_node)
                    G_sub.add_edge(section, chunk_node)

        pos_3d = nx.spring_layout(G_sub, dim=3, k=0.5, seed=42)
        self._plot_3d_network(G_sub, pos_3d, title="3D Semantic Network (Dark Mode)")

    # --------------------
    # Convert Forest to NetworkX Graph
    # --------------------
    def to_networkx(self):
        G = nx.DiGraph()
        for chapter, sections in self.forest.items():
            G.add_node(chapter, label=chapter)
            for sec_index, (section, chunks) in enumerate(sections.items(), 1):
                section_label = f"Section {sec_index}"
                G.add_node(section, label=section_label)
                G.add_edge(chapter, section)
                for chunk_index, chunk in enumerate(chunks, 1):
                    chunk_node = f"{section}_chunk{chunk_index}"
                    chunk_label = f"Chunk {chunk_index}"
                    G.add_node(chunk_node, label=chunk_label)
                    G.add_edge(section, chunk_node)
        return G

    # --------------------
    # Plot 3D Forest v2 (Full Structure, Customizable)
    # --------------------
    def plot_3d_forest_v2_plotly(self, selected_nodes=[], downsample_factor=1):
        G, pos_3d = self._build_3d_graph_structure(selected_nodes, downsample_factor)
        self._plot_3d_network(G, pos_3d, selected_nodes=selected_nodes, title="3D Forest Hierarchy (Chapters → Sections → Chunks)", animate=True)

    # --------------------
    # Internal: Build 3D Structure Layout
    # --------------------
    def _build_3d_graph_structure(self, selected_nodes, downsample_factor):
        G = nx.DiGraph()
        pos_3d = {}
        z_chapter, z_section, z_chunk = 100, 75, 50
        spacing_x, offset_section_y, offset_chunk_y = 750, 550, 550

        for chapter_idx, (chapter, sections) in enumerate(self.forest.items()):
            if chapter_idx % downsample_factor != 0:
                continue
            x = chapter_idx * spacing_x
            pos_3d[chapter] = (x, 0, z_chapter)
            G.add_node(chapter)

            for sec_idx, (section, chunks) in enumerate(sections.items()):
                if sec_idx % downsample_factor != 0:
                    continue
                section_y = (sec_idx - (len(sections) - 1) / 2) * offset_section_y
                pos_3d[section] = (x, section_y, z_section)
                G.add_node(section)
                G.add_edge(chapter, section)

                for chunk_idx, chunk in enumerate(chunks):
                    if chunk_idx % downsample_factor != 0:
                        continue
                    chunk_y = section_y + (chunk_idx - (len(chunks) - 1) / 2) * offset_chunk_y
                    chunk_node = f"{section}_chunk{chunk_idx + 1}"
                    pos_3d[chunk_node] = (x, chunk_y, z_chunk)
                    G.add_node(chunk_node)
                    G.add_edge(section, chunk_node)
        return G, pos_3d

    # --------------------
    # Internal: Plot 3D Graph with Optional Animation
    # --------------------
    def _plot_3d_network(self, G, pos_3d, selected_nodes=[], title="", animate=False):
        # Edge plotting
        edge_trace = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(width=1, color='gray'), hoverinfo='none')
        for u, v in G.edges():
            if u in pos_3d and v in pos_3d:
                x0, y0, z0 = pos_3d[u]
                x1, y1, z1 = pos_3d[v]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
                edge_trace['z'] += (z0, z1, None)

        # Node plotting
        node_trace = go.Scatter3d(
            x=[], y=[], z=[], text=[], mode='markers+text', textposition="top center",
            marker=dict(symbol='circle', size=4, line=dict(color='white', width=0.5)),
            textfont=dict(color="white")
        )

        node_colors, node_texts = [], []
        for node in pos_3d:
            x, y, z = pos_3d[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['z'] += (z,)

            if node in selected_nodes:
                node_texts.append("Big Text Unit" if z == 75 else node)
                node_colors.append("red")
            elif z == 50 and any(node.startswith(f"{s}_chunk") for s in selected_nodes):
                node_texts.append("stu")
                node_colors.append("red")
            else:
                node_texts.append("")
                node_colors.append("deepskyblue")

        node_trace['text'] = node_texts
        node_trace['marker']['color'] = node_colors

        # Plot assembly
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            title=title,
            title_font=dict(color="white", size=22),
            showlegend=False,
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, color="white"),
                yaxis=dict(showbackground=False, showgrid=False, color="white"),
                zaxis=dict(showbackground=False, showgrid=False, color="white"),
            ),
            margin=dict(b=20, l=20, r=20, t=50)
        ))

        # Optional: animate camera
        if animate:
            frames = []
            num_frames = 200
            radius, z_eye = 2, 0.5
            for t in range(num_frames):
                theta = 2 * np.pi * t / num_frames
                camera = dict(eye=dict(x=radius * np.cos(theta), y=radius * np.sin(theta), z=z_eye * radius))
                frames.append(go.Frame(layout=dict(scene_camera=camera)))
            fig.frames = frames
            fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, buttons=[
                dict(label="Play", method="animate", args=[None, dict(
                    frame=dict(duration=1, redraw=True), fromcurrent=True, mode='immediate')])
            ])])

        fig.show()
