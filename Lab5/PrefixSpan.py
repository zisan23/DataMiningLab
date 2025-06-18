import copy
import time
from collections import defaultdict
from memory_profiler import memory_usage
from math import ceil

def load_data(file_path):
    """Load sequence data from SPMF format file"""
    sequences = []
    current_sequence = []
    current_event = set()
    
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                num = int(token)
                
                if num == -1:  # End of event
                    if current_event:
                        current_sequence.append(sorted(current_event))
                        current_event = set()
                elif num == -2:  # End of sequence
                    if current_event:
                        current_sequence.append(sorted(current_event))
                        current_event = set()
                    if current_sequence:
                        sequences.append(current_sequence)
                        current_sequence = []
                else:  # Event ID
                    current_event.add(num)
    
    # Handle any remaining data
    if current_event:
        current_sequence.append(sorted(current_event))
    if current_sequence:
        sequences.append(current_sequence)
        
    return sequences

def prefix_span(sequences, min_support, max_pattern_length):
    """
    Main PrefixSpan function to mine sequential patterns
    
    Parameters:
    - sequences: list of sequences, each sequence is a list of itemsets
    - min_support: minimum support threshold (absolute count)
    - max_pattern_length: maximum length of patterns to mine
    
    Returns:
    - List of (pattern, support) tuples
    """
    # Results container
    frequent_patterns = []
    
    # Get frequent items
    all_items = set()
    item_support = defaultdict(int)
    
    # First scan to find frequent 1-items
    for sequence in sequences:
        # For each sequence, an item can be counted only once
        items_in_seq = set()
        
        for itemset in sequence:
            for item in itemset:
                items_in_seq.add(item)
                all_items.add(item)
        
        for item in items_in_seq:
            item_support[item] += 1
    
    # Filter to get only frequent items
    frequent_items = {item: support for item, support in item_support.items() 
                     if support >= min_support}
    
    # Sort frequent items for consistent output
    sorted_frequent_items = sorted(frequent_items.items())
    
    # Generate frequent 1-patterns
    for item, support in sorted_frequent_items:
        pattern = [item]
        frequent_patterns.append((pattern, support))
        
        # Project database for this item
        projected_db = []
        for sequence in sequences:
            # Find postfix starting from first occurrence of item
            postfix = []
            found = False
            
            for i, itemset in enumerate(sequence):
                if item in itemset:
                    found = True
                    # Add remaining items from this itemset (if any)
                    remaining = [x for x in itemset if x > item]
                    if remaining:
                        postfix.append(remaining)
                    # Add remaining itemsets
                    postfix.extend(sequence[i+1:])
                    break
            
            if found and postfix:
                projected_db.append(postfix)
        
        # Recursive pattern growth
        if len(pattern) < max_pattern_length and projected_db:
            prefix_span_rec(pattern, projected_db, min_support, max_pattern_length, frequent_patterns)
    
    return frequent_patterns

def prefix_span_rec(pattern, projected_db, min_support, max_pattern_length, frequent_patterns):
    """Recursive pattern growth function for PrefixSpan"""
    # Find all frequent items in the projected database
    item_support = defaultdict(int)
    
    # Count support
    for sequence in projected_db:
        # Track items found in this sequence to avoid counting duplicates
        found_items = set()
        
        for itemset in sequence:
            for item in itemset:
                if item not in found_items:
                    item_support[item] += 1
                    found_items.add(item)
    
    # Filter to get only frequent items
    frequent_items = {item: support for item, support in item_support.items() 
                     if support >= min_support}
    
    # Sort frequent items for consistent output
    sorted_frequent_items = sorted(frequent_items.items())
    
    # For each frequent item, extend pattern
    for item, support in sorted_frequent_items:
        new_pattern = pattern + [item]
        frequent_patterns.append((new_pattern, support))
        
        # If maximum pattern length not reached, project database
        if len(new_pattern) < max_pattern_length:
            new_projected_db = []
            for sequence in projected_db:
                # Find postfix starting from first occurrence of item
                postfix = []
                found = False
                
                for i, itemset in enumerate(sequence):
                    if item in itemset:
                        found = True
                        # Add remaining items from this itemset (if any)
                        remaining = [x for x in itemset if x > item]
                        if remaining:
                            postfix.append(remaining)
                        # Add remaining itemsets
                        postfix.extend(sequence[i+1:])
                        break
                
                if found and postfix:
                    new_projected_db.append(postfix)
            
            # Recursive call if there are sequences in projected database
            if new_projected_db:
                prefix_span_rec(new_pattern, new_projected_db, min_support, max_pattern_length, frequent_patterns)

def run_prefix_span(file_path, minsup, max_length=float('inf')):
    """Run PrefixSpan algorithm and track performance metrics"""
    # Track performance
    start_time = time.time()
    start_mem = memory_usage()[0]
    
    # Load data
    sequences = load_data(file_path)
    
    # Convert minsup to absolute count if it's a fraction
    if 0 < minsup < 1:
        minsup = ceil(minsup * len(sequences))
    
    # Use the corrected implementation
    frequent_patterns = prefix_span(sequences, minsup, max_length)
    
    # Sort by support (descending)
    frequent_patterns.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    
    # Calculate performance metrics
    end_time = time.time()
    end_mem = memory_usage()[0]
    memory_consumption = end_mem - start_mem
    execution_time = end_time - start_time
    
    return frequent_patterns, len(sequences), memory_consumption, execution_time

def format_pattern(pattern):
    """Format a pattern for display"""
    return ' -> '.join(map(str, pattern))

# Example usage
if __name__ == "__main__":
    # File paths
    file_path_name = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab5"
    file_name = r"a.txt"
    file_path = file_path_name + "\\" + file_name
    
    # Parameters
    min_sup = 4  # Minimum support threshold (50%)
    max_pattern_length = 1000000000  # Maximum length to mine
    
    print(f"Loading sequences from {file_name}...")
    patterns, total_sequences, memory_usage_val, execution_time = run_prefix_span(file_path, min_sup, max_pattern_length)
    
    # Print performance summary
    print("\n=== Performance Summary ===")
    print(f"Dataset: {file_name}")
    print(f"Minimum support: {min_sup} ({min_sup*100}%)")
    print(f"Total sequences: {total_sequences}")
    print(f"Total memory used: {memory_usage_val:.2f} MB")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total frequent patterns found: {len(patterns)}")
    
    # Print top patterns
    print("\nTop 20 frequent patterns:")
    for i, (pattern, support) in enumerate(patterns[:20], 1):
        print(f"{i}. {format_pattern(pattern)}: {support} ({support/total_sequences:.2%})")
    
    # Save results to output file
    O_file_path = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab5"
    O_file_name = O_file_path + "\\" + "results.txt"
    with open(O_file_name, "a") as f:
        f.write(f"PrefixSpan Results (Min_Sup = {min_sup})\n")
        f.write(f"Total memory used: {memory_usage_val:.2f} MB\n")
        f.write(f"Total execution time: {execution_time:.2f} seconds\n")
        f.write(f"Total frequent patterns found: {len(patterns)}\n\n")