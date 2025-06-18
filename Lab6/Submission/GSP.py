import copy
import time
from memory_profiler import memory_usage
from math import ceil

def load_data(file_path):
    sequences = []
    current_sequence = []
    current_event = set()
    
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                num = int(token)
                
                if num == -1:  # End of event
                    current_sequence.append(sorted(current_event))
                    current_event = set()
                elif num == -2:  # End of sequence
                    if current_sequence: # Only add non-empty sequences
                        sequences.append(current_sequence)
                        current_sequence = []
                else:  # Event ID
                    current_event.add(num)
    return sequences

def is_subsequence(main_sequence, subsequence):
    """Check if subsequence is contained in main_sequence"""
    def is_subsequence_recursive(subsequence_clone, start=0):
        if not subsequence_clone:
            return True
        first_elem = set(subsequence_clone.pop(0))
        for i in range(start, len(main_sequence)):
            if set(main_sequence[i]).issuperset(first_elem):
                return is_subsequence_recursive(subsequence_clone, i + 1)
        return False
    return is_subsequence_recursive(copy.deepcopy(subsequence))

def sequence_length(sequence):
    """Calculate the total number of items in the sequence"""
    return sum(len(i) for i in sequence)

def count_support(sequences, cand_seq):
    """Count how many sequences contain the candidate sequence"""
    return sum(1 for seq in sequences if is_subsequence(seq, cand_seq))

def gen_cands_for_pair(cand1, cand2): # This is the join step of GSP
    #Generate a new candidate from two candidates of length k-1
    #Eta level 3 theke use korte hbe for candidate generation
    cand1_clone = copy.deepcopy(cand1)
    cand2_clone = copy.deepcopy(cand2)
    
    if len(cand1[0]) == 1:
        cand1_clone.pop(0)
    else:
        cand1_clone[0] = cand1_clone[0][1:]
        
    if len(cand2[-1]) == 1:
        cand2_clone.pop(-1)
    else:
        cand2_clone[-1] = cand2_clone[-1][:-1]
        
    if not cand1_clone == cand2_clone:
        return []
    else:
        new_cand = copy.deepcopy(cand1)
        if len(cand2[-1]) == 1:
            new_cand.append(cand2[-1])
        else:
            new_cand[-1].extend([cand2[-1][-1]])
        return new_cand

def gen_cands(last_lvl_cands):
    #Generate candidate sequences of length k+1 from frequent sequences of length k
    #This is the main candidate generation step of GSP
    k = sequence_length(last_lvl_cands[0]) + 1
    
    if k == 2:
        flat_short_cands = [item for sublist2 in last_lvl_cands 
                          for sublist1 in sublist2 
                          for item in sublist1]
        result = [[[a, b]] for a in flat_short_cands 
                          for b in flat_short_cands 
                          if b > a]
        result.extend([[[a], [b]] for a in flat_short_cands 
                                     for b in flat_short_cands])
        return result
    else:
        cands = []
        for i in range(len(last_lvl_cands)):
            for j in range(len(last_lvl_cands)):
                new_cand = gen_cands_for_pair(last_lvl_cands[i], last_lvl_cands[j])
                if new_cand:
                    cands.append(new_cand)
        return cands


#Ekhan theke pruning shuru korbo...21May
def gen_direct_subsequences(sequence):
    #Generate all possible direct subsequences of length k-1
    result = []
    for i, itemset in enumerate(sequence):
        if len(itemset) == 1:
            seq_clone = copy.deepcopy(sequence)
            seq_clone.pop(i)
            result.append(seq_clone)
        else:
            for j in range(len(itemset)):
                seq_clone = copy.deepcopy(sequence)
                seq_clone[i].pop(j)
                result.append(seq_clone)
    return result

def prune_cands(last_lvl_cands, cands_gen):
    #Prune candidates that have any (k-1)-subsequence 
    # #that is not frequent
    return [cand for cand in cands_gen 
            if all(any(cand_subseq == freq_seq for freq_seq in last_lvl_cands) 
                 for cand_subseq in gen_direct_subsequences(cand))]


def gsp_print(dataset, min_sup, verbose, output_file="gsp_results.txt"):
    # Open output file
    f_out = open(output_file, 'w') if output_file else None
    
    # Helper function to print and write to file
    def print_and_write(text):
        print(text)
        if f_out:
            f_out.write(text + "\n")
    
    # Convert min_sup to absolute count if it's a fraction
    if 0 < min_sup < 1:
        min_sup = int(min_sup * len(dataset))
    
    # Initialize
    overall = []
    
    # Get all unique items
    items = sorted(set([item for sequence in dataset
                       for itemset in sequence
                       for item in itemset]))
    
    # Generate 1-sequences
    single_item_sequences = [[[item]] for item in items]
    single_item_counts = [(s, count_support(dataset, s)) 
                         for s in single_item_sequences]
    single_item_counts = [(i, count) for i, count in single_item_counts 
                         if count >= min_sup]
    overall.append(single_item_counts)
    
    # Print level 1 results
    if verbose:
        print_and_write("\n" + "="*80)
        print_and_write(f"LEVEL 1 PATTERNS")
        print_and_write("="*80)
        
        # C1 Table header
        print_and_write(f"C1: Candidate 1-sequences: {len(single_item_sequences)}")
        print_and_write("+---------+----------------+")
        print_and_write("| Index   | Sequence       |")
        print_and_write("+---------+----------------+")
        
        # C1 Table content
        for i, seq in enumerate(single_item_sequences, 1):
            print_and_write(f"| {i:<7} | {format_sequence(seq):<14} |")
        print_and_write("+---------+----------------+")
        
        # L1 Table header
        print_and_write(f"\nL1: Frequent 1-sequences: {len(single_item_counts)}")
        print_and_write("+---------+----------------+----------+")
        print_and_write("| Index   | Sequence       | Support  |")
        print_and_write("+---------+----------------+----------+")
        
        # L1 Table content
        for i, (seq, sup) in enumerate(single_item_counts, 1):
            print_and_write(f"| {i:<7} | {format_sequence(seq):<14} | {sup:<8} |")
        print_and_write("+---------+----------------+----------+")
    
    # Generate k-sequences for k > 1
    k = 1
    while overall[k - 1]:
        last_lvl_cands = [x[0] for x in overall[k - 1]]
        cands_gen = gen_cands(last_lvl_cands)
        cands_pruned = prune_cands(last_lvl_cands, cands_gen)
        cands_counts = [(s, count_support(dataset, s)) for s in cands_pruned]
        result_lvl = [(i, count) for i, count in cands_counts if count >= min_sup]
        
        if verbose:
            print_and_write("\n" + "="*80)
            print_and_write(f"LEVEL {k+1} PATTERNS")
            print_and_write("="*80)
            
            # C(k+1) before pruning table
            max_seq_len = max(len(format_sequence(seq)) for seq in cands_gen) if cands_gen else 14
            max_seq_len = max(max_seq_len, 14)  # Minimum width
            
            print_and_write(f"C{k+1}: Candidate {k+1}-sequences (before pruning): {len(cands_gen)}")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+")
            print_and_write(f"| Index   | {'Sequence':<{max_seq_len}} |")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+")
            
            for i, seq in enumerate(cands_gen, 1):
                print_and_write(f"| {i:<7} | {format_sequence(seq):<{max_seq_len}} |")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+")
            
            # C(k+1) after pruning table
            print_and_write(f"\nC{k+1}: Candidate {k+1}-sequences (after pruning): {len(cands_pruned)}")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+")
            print_and_write(f"| Index   | {'Sequence':<{max_seq_len}} |")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+")
            
            for i, seq in enumerate(cands_pruned, 1):
                print_and_write(f"| {i:<7} | {format_sequence(seq):<{max_seq_len}} |")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+")
            
            # L(k+1) frequent sequences table
            print_and_write(f"\nL{k+1}: Frequent {k+1}-sequences: {len(result_lvl)}")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+----------+")
            print_and_write(f"| Index   | {'Sequence':<{max_seq_len}} | Support  |")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+----------+")
            
            for i, (seq, sup) in enumerate(result_lvl, 1):
                print_and_write(f"| {i:<7} | {format_sequence(seq):<{max_seq_len}} | {sup:<8} |")
            print_and_write(f"+---------+{'-'*(max_seq_len+2)}+----------+")
        
        overall.append(result_lvl)
        k += 1
    
    # Flatten the results and sort by support (descending)
    overall = overall[:-1]  # Remove empty last level
    overall = [item for sublist in overall for item in sublist]
    overall.sort(key=lambda tup: (tup[1], -sequence_length(tup[0])), reverse=True)
    
    # Print summary of all frequent patterns
    if verbose:
        print_and_write("\n" + "="*80)
        print_and_write("SUMMARY OF ALL FREQUENT PATTERNS")
        print_and_write("="*80)
        
        max_seq_len = max(len(format_sequence(seq)) for seq, _ in overall) if overall else 14
        max_seq_len = max(max_seq_len, 14)  # Minimum width
        
        print_and_write(f"+---------+{'-'*(max_seq_len+2)}+----------+")
        print_and_write(f"| Index   | {'Sequence':<{max_seq_len}} | Support  |")
        print_and_write(f"+---------+{'-'*(max_seq_len+2)}+----------+")
        
        for i, (seq, sup) in enumerate(overall, 1):
            print_and_write(f"| {i:<7} | {format_sequence(seq):<{max_seq_len}} | {sup:<8} |")
        print_and_write(f"+---------+{'-'*(max_seq_len+2)}+----------+")
        
        print_and_write(f"\nTotal frequent sequences found: {len(overall)}")
    
    # Close output file
    if f_out:
        f_out.close()
        print(f"Results saved to {output_file}")
    
    return overall

def gsp(dataset, min_sup, verbose):
    start_time = time.time()
    start_mem = memory_usage()[0]
    # Convert min_sup to absolute count if it's a fraction
    if 0 < min_sup < 1:
        min_sup = ceil(min_sup * len(dataset))
    
    # Initialize
    overall = []
    
    # Get all unique items
    items = sorted(set([item for sequence in dataset
                       for itemset in sequence
                       for item in itemset]))
    
    # Generate 1-sequences
    single_item_sequences = [[[item]] for item in items]
    single_item_counts = [(s, count_support(dataset, s)) 
                         for s in single_item_sequences]
    single_item_counts = [(i, count) for i, count in single_item_counts 
                         if count >= min_sup]
    overall.append(single_item_counts)
    
    # Generate k-sequences for k > 1
    k = 1
    while overall[k - 1]:
        last_lvl_cands = [x[0] for x in overall[k - 1]]
        cands_gen = gen_cands(last_lvl_cands)
        cands_pruned = prune_cands(last_lvl_cands, cands_gen)
        cands_counts = [(s, count_support(dataset, s)) for s in cands_pruned]
        result_lvl = [(i, count) for i, count in cands_counts if count >= min_sup]
        
        if verbose :
            print('Candidates generated, lvl', k + 1, ':', cands_gen)
            print('\nCandidates pruned, lvl', k + 1, ':', cands_pruned)
            print('Result, Level', k + 1, ':', result_lvl)
            print('-' * 100)
        
        overall.append(result_lvl)
        k += 1
    
    # Flatten the results and sort by support (descending)
    overall = overall[:-1]  # Remove empty last level
    overall = [item for sublist in overall for item in sublist]
    overall.sort(key=lambda tup: (tup[1], -sequence_length(tup[0])), reverse=True)
    
    end_time = time.time()
    end_mem = memory_usage()[0]
    memory_consumption = end_mem - start_mem
    execution_time = end_time - start_time
    
    return overall, memory_consumption, execution_time

def format_sequence(sequence):
    """Format a sequence for display"""
    return str(sequence).replace('], [', ' -> ').replace('[', '').replace(']', '')

# Example usage:
if __name__ == "__main__":
    file_path_name = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab4"
    file_name = r"a.txt"

    file_path =  file_path_name+ r"\\" +file_name
    sequences = load_data(file_path)
    print(f"Loaded {len(sequences)} sequences from {file_name}")


    min_sup = .03 # support
    print(f"Running GSP with min_sup={min_sup}")
    
    #RUN GSP with output printing
    output_file = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab4\output.txt"
    results = gsp_print(sequences, min_sup, verbose=True, output_file=output_file)
    

    # Run GSP
    # results, memory_usages, execution_time = gsp(sequences, min_sup, verbose=False)
    # total_frequent_items = len(results)
    # # Print final performance summary
    # print("\n=== Final Performance Summary ===")
    # # print(f"Dataset Name: {file_name}")
    # print(f"(Min_Sup = {min_sup})")
    # print(f"Total memory used by algorithm: {memory_usages:.2f} MB")
    # print(f"Total execution time: {execution_time:.2f} seconds")
    # print(f"Total number of frequent items: {total_frequent_items}")
    
    # O_file_path = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab4\Output"
    # O_file_name = O_file_path + r"\\" + file_name
    # with open(O_file_name, "a") as f:
    #     f.write(f"(Min_Sup = {min_sup})\n")
    #     f.write(f"Total memory used by algorithm: {memory_usages:.2f} MB\n")
    #     f.write(f"Total execution time: {execution_time:.2f} seconds\n")
    #     f.write(f"Total number of frequent items: {total_frequent_items}\n")