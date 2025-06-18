import time
import psutil
import os
from collections import defaultdict
from memory_profiler import memory_usage

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0
        self.itemset = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, itemset, transaction_id=None):
        node = self.root
        for item in sorted(itemset):
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.is_end = True
        node.count += 1
        node.itemset = tuple(sorted(itemset))
    
    def insert_transaction(self, transaction, transaction_id):
        for item in transaction:
            self.insert([item], transaction_id)
    
    def get_frequent_itemsets(self, min_support, transaction_count):
        frequent = {}
        
        def dfs(node, itemset):
            if node.is_end and node.count >= min_support:
                support_percentage = (node.count / transaction_count) * 100
                frequent[node.itemset] = (node.count, support_percentage)
            
            for item, child in node.children.items():
                dfs(child, itemset + [item])
        
        dfs(self.root, [])
        return frequent
    
    def get_candidates(self, k):
        candidates = []
        
        def dfs(node, current_itemset):
            if len(current_itemset) == k:
                candidates.append(current_itemset)
                return
            
            for item, child in sorted(node.children.items()):
                dfs(child, current_itemset + [item])
        
        dfs(self.root, [])
        return candidates
    
    def support_count(self, itemset, transactions):
        count = 0
        for transaction in transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        return count

def generate_candidates(itemset, length):
    candidates = []
    for i in range(len(itemset)):
        for j in range(i + 1, len(itemset)):
            union_set = set(itemset[i]).union(set(itemset[j]))
            if len(union_set) == length:
                candidates.append(list(union_set))
    return candidates

def get_support(transactions, itemset):
    count = 0
    for transaction in transactions:
        if set(itemset).issubset(set(transaction)):
            count += 1
    return count, count / len(transactions) * 100

def apriori_trie(transactions, min_support_percentage):
    start_time = time.time()
    min_support_count = (min_support_percentage / 100) * len(transactions)
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    frequent_items = {item: count for item, count in item_counts.items() 
                     if count >= min_support_count}
    if not frequent_items:
        return []
    trie = Trie()
    for item in frequent_items:
        trie.insert([item])
        trie.root.children[item].count = frequent_items[item]
    k = 1
    frequent_itemsets = []
    support_count_1 = {}
    for item, count in frequent_items.items():
        support_percentage = (count / len(transactions)) * 100
        support_count_1[tuple([item])] = (count, support_percentage)
    frequent_itemsets.append(support_count_1)
    while True:
        k += 1
        prev_frequent = [list(x) for x in frequent_itemsets[-1].keys()]
        candidates = generate_candidates(prev_frequent, k)
        if not candidates:
            break
        support_count_k = {}
        for candidate in candidates:
            count = 0
            for transaction in transactions:
                if set(candidate).issubset(set(transaction)):
                    count += 1
            if count >= min_support_count:
                support_percentage = (count / len(transactions)) * 100
                support_count_k[tuple(sorted(candidate))] = (count, support_percentage)
        if not support_count_k:
            break
        frequent_itemsets.append(support_count_k)
    end_time = time.time()
    return frequent_itemsets

def run_apriori(file_path, min_support):
    transactions = []
    with open(file_path, 'r') as file:
        for line in file:
            transaction = line.strip().split()
            transactions.append(transaction)
    start_mem = memory_usage()[0]
    start_time = time.time()
    frequent_itemsets = apriori_trie(transactions, min_support)
    end_time = time.time()
    end_mem = memory_usage()[0]
    memory_consumption = end_mem - start_mem
    execution_time = end_time - start_time
    total_frequent_count = sum(len(level) for level in frequent_itemsets)
    return memory_consumption, execution_time, total_frequent_count

if __name__ == "__main__":
    file_path = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab1\retail.dat"
    total_transactions = 88162
    min_support = 20 # Here the minimum supprot is relative
    memory_usage, execution_time, total_frequent_items = run_apriori(file_path, min_support)
    print(f"(Min_Sup = {min_support}%)")
    print(f"Total memory used by algorithm: {memory_usage:.2f} MB")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total number of frequent items: {total_frequent_items}")
