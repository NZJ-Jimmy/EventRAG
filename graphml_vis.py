import networkx as nx
import matplotlib.pyplot as plt

# Configure Matplotlib for proper font rendering
plt.rcParams['font.sans-serif'] = 'Noto Sans CJK SC' # Support Chinese characters
plt.rcParams['axes.unicode_minus'] = False # Avoid issues with negative signs

# Load the GraphML file
graph = nx.read_graphml("surfilter/event_graph_chunk_entity_relation.graphml")

# Convert to a simple graph if necessary
if not isinstance(graph, nx.Graph):
    graph = nx.Graph(graph)

# Generate layout for nodes
positions = nx.spring_layout(graph)

# Draw the graph
nx.draw(
    graph,
    pos=positions,
    with_labels=True,
    node_color='skyblue',
    node_size=1500,
    font_size=10
)

# Display the visualization
plt.title("Graph Visualization")
plt.axis('off')
plt.show()