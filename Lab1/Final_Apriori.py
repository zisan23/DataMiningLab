import time
from prettytable import PrettyTable
from memory_profiler import memory_usage
import os

def generate_candidates(itemset, length):
    candidates = []
    
    # For k=1, we handle separately in the apriori function
    if length == 1:
        return sorted(itemset)  # Ensure consistent ordering
        
    # For k=2, simply create pairs from 1-itemsets
    if length == 2:
        # Sort the itemset for consistent ordering
        sorted_itemset = sorted([tuple(item) for item in itemset])
        itemset = [list(item) for item in sorted_itemset]
        
        for i in range(len(itemset)):
            for j in range(i + 1, len(itemset)):
                # creating a unique 2-itemset
                candidate = sorted(list(set(itemset[i] + itemset[j])))
                if len(candidate) == length and candidate not in candidates:
                    candidates.append(candidate)
        return candidates
    
    # For k>2, use the standard Apriori join step
    sorted_itemset = sorted([tuple(sorted(items)) for items in itemset])
    itemset = [list(item) for item in sorted_itemset]
    
    # Create a set of tuples for faster lookups during pruning
    itemset_set = set(tuple(items) for items in itemset)
    
    # Join step
    for i in range(len(itemset)):
        for j in range(i + 1, len(itemset)):
            # First k-2 elements must be identical
            if itemset[i][:length-2] == itemset[j][:length-2]:
                # Last elements must be different and in order (to avoid duplicates)
                if itemset[i][length-2] < itemset[j][length-2]:
                    # Generate new candidate by joining the two itemsets
                    new_candidate = itemset[i][:length-2] + [itemset[i][length-2], itemset[j][length-2]]
                    
                    # Prune step: All (k-1) subsets must be frequent
                    should_add = True
                    # Check each possible (k-1) subset
                    for m in range(length):
                        subset = new_candidate.copy()
                        del subset[m]  # Remove one item to create a (k-1) subset
                        if tuple(subset) not in itemset_set:
                            should_add = False
                            break
                    
                    if should_add:
                        candidates.append(new_candidate)
    
    return candidates

def get_support(transactions, itemset):
    count = 0
    itemset_set = frozenset(itemset)  # Use frozenset for immutability and consistency
    for transaction in transactions:
        if itemset_set.issubset(set(transaction)):
            count += 1
    return count, (count / len(transactions)) * 100

def apriori(transactions, min_support_percentage):
    start_time = time.time()
    start_mem = memory_usage()[0]
    
    # Find all unique items in transactions and sort them for consistency
    unique_items = sorted(set(item for transaction in transactions for item in transaction))
    
    # Create 1-itemsets
    items = [[item] for item in unique_items]
    
    k = 1
    frequent_itemsets = []
    total_frequent_count = 0

    while items:
        print(f"\nLevel {k} Candidate Itemsets: {len(items)}")
        
        # Sort items for consistent processing order
        items = sorted([sorted(item) for item in items])
        
        # Find frequent itemsets at this level
        support_count = {}
        for item in items:
            abs_count, support = get_support(transactions, item)
            if support >= min_support_percentage:
                support_count[tuple(sorted(item))] = (abs_count, support)

        # If we found any frequent itemsets
        if support_count:
            frequent_itemsets.append(support_count)
            level_count = len(support_count)
            total_frequent_count += level_count
            
            # Print results in a table
            table = PrettyTable()
            table.field_names = ["Itemset", "Absolute Support", "Support (%)"]
            
            # Sort itemsets for consistent display
            sorted_itemsets = sorted(support_count.items())
            for itemset, (abs_count, support) in sorted_itemsets:
                table.add_row([list(itemset), abs_count, f"{support:.2f}"])
                
            print(f"\nLevel {k} Frequent Itemsets: {level_count}")
            # print(table)
            
            # Generate candidates for next level
            items = generate_candidates([list(k) for k in support_count.keys()], k + 1)
        else:
            print(f"No frequent itemsets found at level {k}")
            break
        
        k += 1

    # Calculate memory usage and execution time
    end_time = time.time()
    end_mem = memory_usage()[0]
    memory_consumption = end_mem - start_mem
    execution_time = end_time - start_time
    
    return frequent_itemsets, memory_consumption, execution_time, total_frequent_count



def run_apriori(file_path, min_support):
    transactions = []
    with open(file_path, 'r') as file:
        for line in file:
            transaction = line.strip().split()
            transactions.append(transaction)
    
    print(f"Total number of transactions: {len(transactions)}")
    print(f"(Min_Sup = {min_support}%)")
    
    frequent_sets, memory_usage, execution_time, total_frequent_items = apriori(transactions, min_support)
    
    frequent_sets_count = [0] * len(frequent_sets)
    for i, itemsets in enumerate(frequent_sets):
        print(f"\nFrequent {i+1}-itemsets:")
        table = PrettyTable()
        table.field_names = ["Itemset", "Absolute Support", "Support (%)"]
        for itemset, (abs_count, support) in itemsets.items():
            table.add_row([list(itemset), abs_count, f"{support:.2f}"])
        print(table)
        frequent_sets_count[i] = len(itemsets)
    
    for i in range(0, len(frequent_sets_count)):
        print(f"Total number of frequent {i+1}-itemsets: {frequent_sets_count[i]}")
        
    print("\n=== Final Performance Summary ===")
    print(f"Total memory used by algorithm: {memory_usage:.2f} MB")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total number of frequent items: {total_frequent_items}")
    
    
    
    return memory_usage, execution_time, total_frequent_items

if __name__ == "__main__":
    
    file_path = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab1\kosarak.txt"
    min_support = 50 # Minimum support realtive
    
    run_apriori(file_path, min_support)
    
