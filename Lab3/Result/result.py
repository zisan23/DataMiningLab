import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ---- Step 1: Read and Parse the File ----
def parse_file(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    filename = os.path.splitext(os.path.basename(filepath))[0]

    # Extract sections
    apriori_match = re.search(r"Apriori:(.*?)FP_Growth:", text, re.S)
    if not apriori_match:
        print("Warning: Couldn't find the Apriori section properly. Trying alternative pattern.")
        apriori_match = re.search(r"Apriori:\s*(.*?)FP_Growth:", text, re.S)
    
    fp_growth_match = re.search(r"FP_Growth:(.*)", text, re.S)
    
    if not apriori_match or not fp_growth_match:
        print("⚠️ Section extraction failed. Debug info:")
        print(f"File content preview: {text[:200]}...")
        raise ValueError("Failed to extract algorithm sections from file")
    
    apriori_section = apriori_match.group(1)
    fp_growth_section = fp_growth_match.group(1)
    
    # First try D_Chess.txt format for Apriori (with Computation time)
    apriori_data = re.findall(
        r"\(Min_sup = (\d+\.?\d*)%\).*?Total frequent itemsets: (\d+).*?Runtime: ([\d.]+) seconds.*?Memory Usage: ([\d.]+) MB",
        apriori_section,
        re.S
    )
    
    # If no matches, try D_Mushroom.txt format for Apriori (without Computation time)
    if not apriori_data:
        apriori_data = re.findall(
            r"\(Min_sup = (\d+\.?\d*)%\).*?Total frequent itemsets: (\d+).*?Runtime: ([\d.]+) seconds.*?Memory Usage: ([\d.]+) MB",
            apriori_section,
            re.S
        )
        
    # If still no matches, try S_T10.txt format for Apriori
    if not apriori_data:
        apriori_data = re.findall(
            r"\(Min_Sup = (\d+\.?\d*)%\).*?Total memory used by algorithm: ([\d.]+) MB.*?Total execution time: ([\d.]+) seconds.*?Total number of frequent items: (\d+)",
            apriori_section,
            re.S
        )
    
    # Process data if using D_Chess.txt or D_Mushroom.txt format
    if apriori_data and len(apriori_data[0]) == 4 and "Total frequent itemsets" in apriori_section:
        # For these formats, we need to rearrange: [min_sup, freq_items, time, memory] to [min_sup, memory, time, freq_items]
        temp_data = []
        for entry in apriori_data:
            temp_data.append((entry[0], entry[3], entry[2], entry[1]))
        apriori_data = temp_data
    
    # Try both FP-Growth formats (uppercase and lowercase Min_sup)
    fp_growth_data = re.findall(
        r"\(Min_[Ss]up = (\d+\.?\d*)%\s*(?:/\s*[\d.]+)?\).*?Total memory usage: ([\d.]+) MB.*?Total execution time: ([\d.]+) seconds.*?Total number of frequent items: (\d+)",
        fp_growth_section,
        re.S
    )
    
    # Debugging information
    print(f"Found {len(apriori_data)} Apriori entries and {len(fp_growth_data)} FP-Growth entries")
    
    if len(apriori_data) != len(fp_growth_data):
        print("⚠️ Warning: Different number of entries found for Apriori and FP-Growth")
    
    if not apriori_data or not fp_growth_data:
        print("⚠️ No data extracted. Preview of sections:")
        print(f"Apriori section (first 300 chars): {apriori_section[:300]}")
        print(f"FP-Growth section (first 300 chars): {fp_growth_section[:300]}")
    
    return filename, apriori_data, fp_growth_data

# ---- Step 2: Organize the Data ----
def organize_data(apriori_data, fp_growth_data):
    min_sup = []
    apriori_time, apriori_memory, apriori_freq = [], [], []
    fp_growth_time, fp_growth_memory, fp_growth_freq = [], [], []

    for entry in apriori_data:
        min_sup.append(float(entry[0]))
        apriori_memory.append(float(entry[1]))
        apriori_time.append(float(entry[2]))
        apriori_freq.append(int(entry[3]))

    for entry in fp_growth_data:
        fp_growth_memory.append(float(entry[1]))
        fp_growth_time.append(float(entry[2]))
        fp_growth_freq.append(int(entry[3]))

    return min_sup, apriori_time, apriori_memory, apriori_freq, fp_growth_time, fp_growth_memory, fp_growth_freq

# ---- Step 3: Plot Graphs ----
def plot_graphs(filename, min_sup, apriori_time, fp_growth_time, apriori_memory, fp_growth_memory):
    # Line plot for execution time
    plt.figure(figsize=(8, 5))
    plt.plot(min_sup, apriori_time, marker='o', label='Apriori')
    plt.plot(min_sup, fp_growth_time, marker='s', label='FP-Growth')

    for i, v in enumerate(apriori_time):
        plt.text(min_sup[i], v + max(apriori_time)*0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(fp_growth_time):
        plt.text(min_sup[i], v + max(fp_growth_time)*0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

    plt.title('Execution Time vs Min Support')
    plt.xlabel('Min Support (%)')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename}_execution_time.png")
    plt.show()

    # Bar plot for memory usage
    x = np.arange(len(min_sup))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, apriori_memory, width, label='Apriori')
    bars2 = ax.bar(x + width/2, fp_growth_memory, width, label='FP-Growth')

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Min Support (%)')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(min_sup)
    ax.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{filename}_memory_usage.png")
    plt.show()

# ---- Step 4: Generate Clean Table ----
def generate_clean_table(filename, min_sup, apriori_time, fp_growth_time, apriori_memory, fp_growth_memory, apriori_freq, fp_growth_freq):
    # Create lists for table structure
    rows = []
    
    # For each min_sup value
    for i, sup in enumerate(min_sup):
        # First row in group - contains min_sup
        rows.append([f"{sup}%", 'Runtime (s)', f"{apriori_time[i]:.2f}", f"{fp_growth_time[i]:.2f}"])
        # Second and third rows - empty min_sup cell
        rows.append(['', 'Memory (MB)', f"{apriori_memory[i]:.2f}", f"{fp_growth_memory[i]:.2f}"])
        rows.append(['', 'Frequent Items', f"{apriori_freq[i]}", f"{fp_growth_freq[i]}"])

    table_df = pd.DataFrame(rows, columns=['Min_sup', 'Metric', 'Apriori', 'FP-Growth'])

    # Print clean table
    print("\n📋 Comparison Table:\n")
    print(table_df.to_string(index=False))

    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, len(rows)*0.4 + 1))
    ax.axis('off')
    
    # Define colors
    header_color = '#4472C4'  # Blue for header
    alt_color = '#E6F0FF'     # Light blue for alternating groups
    
    # Create cell colors for alternating groups
    cell_colors = []
    for i in range(len(rows)):
        group = i // 3  # Determine which min_sup group this row belongs to
        if group % 2 == 0:  # Alternating groups
            cell_colors.append(['white', 'white', 'white', 'white'])
        else:
            cell_colors.append([alt_color, alt_color, alt_color, alt_color])
    
    # Create table
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors
    )
    
    # Style the header
    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor(header_color)
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)  # Make rows a bit taller
    
    # Set column widths
    table.auto_set_column_width(col=list(range(len(table_df.columns))))

    # Add border lines
    for key, cell in table._cells.items():
        cell.set_edgecolor('lightgray')

    # Set title
    plt.suptitle(f'Performance Comparison: Apriori vs FP-Growth', fontsize=14, y=0.98)
    plt.title(f'Dataset: {filename}', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{filename}_clean_table.png", bbox_inches='tight', dpi=300)
    plt.show()

    print(f"\n✅ Clean table saved as: {filename}_clean_table.png")

# ---- Step 5: Main Execution ----
def main():
    try:
        filepath = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Final Result\D_Chess.txt"  # Changed to D_Chess.txt
        filename, apriori_data, fp_growth_data = parse_file(filepath)
        
        if len(apriori_data) == 0 or len(fp_growth_data) == 0:
            print("❌ Error: No data extracted from file. Please check the file format.")
            return
            
        min_sup, apriori_time, apriori_memory, apriori_freq, fp_growth_time, fp_growth_memory, fp_growth_freq = organize_data(apriori_data, fp_growth_data)

        plot_graphs(filename, min_sup, apriori_time, fp_growth_time, apriori_memory, fp_growth_memory)
        generate_clean_table(filename, min_sup, apriori_time, fp_growth_time, apriori_memory, fp_growth_memory, apriori_freq, fp_growth_freq)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

# ---- Execute ----
if __name__ == "__main__":
    main()
