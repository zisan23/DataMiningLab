import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def parse_gsp_ps_files(gsp_path, ps_path):
    # Read GSP file
    with open(gsp_path, 'r') as f:
        gsp_lines = f.readlines()
    # Read PrefixSpan file
    with open(ps_path, 'r') as f:
        ps_lines = f.readlines()

    # Parse GSP
    gsp_data = []
    for i in range(len(gsp_lines)):
        if "(Min_Sup" in gsp_lines[i]:
            min_sup = float(re.search(r"= ([\d.]+)", gsp_lines[i]).group(1))
            mem = float(re.search(r"([\d.]+) MB", gsp_lines[i+1]).group(1))
            time = float(re.search(r"([\d.]+) seconds", gsp_lines[i+2]).group(1))
            freq = int(re.search(r"(\d+)", gsp_lines[i+3]).group(1))
            gsp_data.append((min_sup, mem, time, freq))

    # Parse PrefixSpan
    ps_data = []
    for i in range(len(ps_lines)):
        if "PrefixSpan Results" in ps_lines[i]:
            min_sup = float(re.search(r"= ([\d.]+)", ps_lines[i]).group(1))
            mem = float(re.search(r"([\d.]+) MB", ps_lines[i+1]).group(1))
            time = float(re.search(r"([\d.]+) seconds", ps_lines[i+2]).group(1))
            freq = int(re.search(r"(\d+)", ps_lines[i+3]).group(1))
            ps_data.append((min_sup, mem, time, freq))

    # Sort by min_sup descending (optional, for consistent table)
    gsp_data.sort(reverse=True)
    ps_data.sort(reverse=True)
    return gsp_data, ps_data

def generate_clean_table_gsp_ps(dataset_name, gsp_data, ps_data):
    # Prepare rows for table
    rows = []
    for i in range(len(gsp_data)):
        min_sup = gsp_data[i][0]
        # First row: min_sup, runtime
        rows.append([f"{min_sup}", 'Runtime (s)', f"{gsp_data[i][2]:.2f}", f"{ps_data[i][2]:.2f}"])
        # Second row: memory
        rows.append(['', 'Memory (MB)', f"{gsp_data[i][1]:.2f}", f"{ps_data[i][1]:.2f}"])
        # Third row: frequent patterns
        rows.append(['', 'Frequent Patterns', f"{gsp_data[i][3]}", f"{ps_data[i][3]}"])

    table_df = pd.DataFrame(rows, columns=['Min_sup', 'Metric', 'GSP', 'PrefixSpan'])

    print("\n📋 GSP vs PrefixSpan Comparison Table:\n")
    print(table_df.to_string(index=False))

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, len(rows)*0.4 + 1))
    ax.axis('off')

    header_color = '#4472C4'
    alt_color = '#E6F0FF'

    cell_colors = []
    for i in range(len(rows)):
        group = i // 3
        if group % 2 == 0:
            cell_colors.append(['white', 'white', 'white', 'white'])
        else:
            cell_colors.append([alt_color, alt_color, alt_color, alt_color])

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors
    )

    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    table.auto_set_column_width(col=list(range(len(table_df.columns))))

    for key, cell in table._cells.items():
        cell.set_edgecolor('lightgray')

    plt.suptitle(f'Performance Comparison: GSP vs PrefixSpan', fontsize=14, y=0.98)
    plt.title(f'Dataset: {dataset_name}', fontsize=12)

    plt.tight_layout()
    outname = f"{dataset_name}_GSP_vs_PrefixSpan_table.png"
    plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"\n✅ Clean table saved as: {outname}")

def main():
    # Example usage for BIKE.txt
    gsp_path = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab6\Output_GSP\SIGN.txt"
    ps_path = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab6\Output_PS\SIGN.txt"
    dataset_name = os.path.splitext(os.path.basename(gsp_path))[0]

    gsp_data, ps_data = parse_gsp_ps_files(gsp_path, ps_path)
    if len(gsp_data) == 0 or len(ps_data) == 0:
        print("❌ Error: No data extracted from files. Please check the file format.")
        return
    generate_clean_table_gsp_ps(dataset_name, gsp_data, ps_data)

if __name__ == "__main__":
    main()