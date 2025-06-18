import os
import re
import matplotlib.pyplot as plt

# Directories
gsp_dir = 'output_gsp'
ps_dir = 'output_ps'

# Filenames (should exist in both directories)
files = ['BIKE.txt', 'BMS1.txt', 'e_shop.txt', 'SIGN.txt']

def extract_data(filepath, prefix=False):
    """Extracts min_sup, memory usage, and execution time from a file."""
    with open(filepath, 'r') as file:
        content = file.read()

    pattern = re.compile(
        r'Min_Sup\s*=\s*(\d*\.?\d+).*?memory used(?: by algorithm)?:\s*([\d.]+)\s*MB.*?execution time.*?:\s*([\d.]+)\s*seconds',
        re.IGNORECASE | re.DOTALL
    )
    matches = pattern.findall(content)

    data = {}
    for min_sup, mem, time in matches:
        data[float(min_sup)] = {
            'memory': float(mem),
            'time': float(time)
        }
    return data

for filename in files:
    gsp_data = extract_data(os.path.join(gsp_dir, filename))
    ps_data = extract_data(os.path.join(ps_dir, filename))

    # Take intersection of common Min_Sup values in both files
    common_minsup = sorted(set(gsp_data.keys()) & set(ps_data.keys()))

    gsp_times = [gsp_data[s]['time'] for s in common_minsup]
    ps_times = [ps_data[s]['time'] for s in common_minsup]
    gsp_mems = [gsp_data[s]['memory'] for s in common_minsup]
    ps_mems = [ps_data[s]['memory'] for s in common_minsup]

    # Plot Execution Time (Bar Chart)
    x = range(len(common_minsup))
    x_labels = [f"{s:.2f}" for s in common_minsup]

    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], gsp_times, width=0.4, label='GSP')
    plt.bar([i + 0.2 for i in x], ps_times, width=0.4, label='PrefixSpan')
    plt.xlabel('Min Support')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time Comparison: {filename}')
    plt.xticks(x, x_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filename}_exec_time_comparison.png')
    plt.close()

    # Plot Memory Usage (Line Chart)
    plt.figure(figsize=(10, 6))
    plt.plot(common_minsup, gsp_mems, marker='o', label='GSP')
    plt.plot(common_minsup, ps_mems, marker='s', label='PrefixSpan')
    plt.xlabel('Min Support')
    plt.ylabel('Memory Usage (MB)')
    plt.title(f'Memory Usage Comparison: {filename}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{filename}_memory_usage_comparison.png')
    plt.close()

print("All graphs generated and saved.")
