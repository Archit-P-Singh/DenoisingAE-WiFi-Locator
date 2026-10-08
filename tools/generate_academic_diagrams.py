import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_block_diagram(filename, title, layers):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Starting positions
    x = 0.1
    y = 0.4
    box_width = 0.12
    box_height = 0.25
    gap = 0.08
    
    # Draw blocks
    for i, layer in enumerate(layers):
        name = layer['name']
        dims = layer['dims']
        act = layer.get('act', '')
        
        # Color based on layer type
        if 'Input' in name:
            color = '#E8F5E9' # Light green
            edge = '#2E7D32'
        elif 'Output' in name:
            color = '#FFF3E0' # Light orange
            edge = '#E65100'
        elif 'Noise' in name:
            color = '#FFEBEE' # Light red
            edge = '#C62828'
        else:
            color = '#E3F2FD' # Light blue
            edge = '#1565C0'
            
        # Draw box
        rect = patches.Rectangle((x, y), box_width, box_height, linewidth=2, edgecolor=edge, facecolor=color, zorder=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + box_width/2, y + box_height*0.75, name, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)
        ax.text(x + box_width/2, y + box_height*0.45, dims, ha='center', va='center', fontsize=9, zorder=3)
        if act:
            ax.text(x + box_width/2, y + box_height*0.15, act, ha='center', va='center', fontsize=9, fontstyle='italic', color='#555555', zorder=3)
            
        # Draw arrow to next block
        if i < len(layers) - 1:
            ax.arrow(x + box_width, y + box_height/2, gap - 0.02, 0, head_width=0.03, head_length=0.02, fc='black', ec='black', zorder=1)
            
        x += box_width + gap

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 1. Denoising Autoencoder
ae_layers = [
    {'name': 'Input', 'dims': '520'},
    {'name': 'Corrupt', 'dims': 'Noise 20%'},
    {'name': 'Encoder 1', 'dims': '256', 'act': 'ReLU'},
    {'name': 'Encoder 2', 'dims': '128', 'act': 'ReLU'},
    {'name': 'Latent\n(Enc 3)', 'dims': '64', 'act': 'ReLU'},
    {'name': 'Decoder 1', 'dims': '128', 'act': 'ReLU'},
    {'name': 'Decoder 2', 'dims': '256', 'act': 'ReLU'},
    {'name': 'Output', 'dims': '520', 'act': 'Linear'}
]

# 2. Optimized NN Classifier
nn_layers = [
    {'name': 'Input', 'dims': '520'},
    {'name': 'Pre-trained\nEncoder', 'dims': '256 \u2192 128 \u2192 64', 'act': 'Frozen/Tuned'},
    {'name': 'Dense 1', 'dims': '128', 'act': 'ReLU + L2'},
    {'name': 'Dense 2', 'dims': '128', 'act': 'ReLU + L2'},
    {'name': 'Output', 'dims': '118', 'act': 'Softmax'}
]

draw_block_diagram('results/Denoising_AE_Structure.png', 'Denoising Autoencoder Architecture', ae_layers)
draw_block_diagram('results/Optimized_NN_Structure.png', 'Optimized Neural Network Classifier', nn_layers)

print("Generated academic block diagrams in results/ folder.")
