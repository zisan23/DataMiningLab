import time
from collections import defaultdict
from prettytable import PrettyTable
import psutil
import os

class FPTreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None

    def increment(self, count):
        self.count += count

    def display(self, ind=1):
        print('  ' * ind, f'{self.item}: {self.count}')
        for child in self.children.values():
            child.display(ind + 1)

def build_fp_tree(transactions, min_support):
    header_table = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            header_table[item] += 1
    header_table = {k: v for k, v in header_table.items() if v >= min_support}
    if not header_table:
        return None, None
    for key in header_table:
        header_table[key] = [header_table[key], None]
    root = FPTreeNode(None, 1, None)
    for transaction in transactions:
        filtered_transaction = [item for item in transaction if item in header_table]
        filtered_transaction.sort(key=lambda x: header_table[x][0], reverse=True)
        insert_tree(filtered_transaction, root, header_table)
    return root, header_table

def insert_tree(items, node, header_table):
    if items:
        first_item = items[0]
        if first_item in node.children:
            node.children[first_item].increment(1)
        else:
            new_node = FPTreeNode(first_item, 1, node)
            node.children[first_item] = new_node
            if header_table[first_item][1] is None:
                header_table[first_item][1] = new_node
            else:
                current = header_table[first_item][1]
                while current.link is not None:
                    current = current.link
                current.link = new_node
        insert_tree(items[1:], node.children[first_item], header_table)

def mine_fp_tree(header_table, min_support, prefix, frequent_itemsets):
    sorted_items = sorted(header_table.items(), key=lambda x: x[1][0])
    for base_item, (count, node) in sorted_items:
        new_prefix = prefix.copy()
        new_prefix.add(base_item)
        frequent_itemsets.append((new_prefix, count))
        conditional_pattern_base = []
        while node is not None:
            path = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            path.reverse()
            for _ in range(node.count):
                conditional_pattern_base.append(path)
            node = node.link
        conditional_tree, conditional_header = build_fp_tree(conditional_pattern_base, min_support)
        if conditional_header is not None:
            mine_fp_tree(conditional_header, min_support, new_prefix, frequent_itemsets)

def run_fp_growth(file_path, min_support):
    transactions = []
    with open(file_path, 'r') as file:
        for line in file:
            transactions.append(line.strip().split())
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)
    
    start_time = time.time()
    
    root, header_table = build_fp_tree(transactions, min_support)
    
    if root is not None:
        frequent_itemsets = []
        mine_fp_tree(header_table, min_support, set(), frequent_itemsets)
        end_time = time.time()
        
        final_memory = process.memory_info().rss / (1024 * 1024)
        
        memory_usage = final_memory - initial_memory
        execution_time = end_time - start_time
        total_frequent_items = len(frequent_itemsets)
        
        return memory_usage, execution_time, total_frequent_items
    else:
        return 0.0, 0.0, 0

if __name__ == "__main__":
    file_path = r"C:\Users\Zisan-23\OneDrive\Desktop\Data Mining Lab\Lab2\kosarak.dat"
    
    total_transactions = 990002
    min_support = 495001
    memory_usage, execution_time, total_frequent_items = run_fp_growth(file_path, min_support)
    print(f"(Min_Sup = {min_support*100 / total_transactions:.0f}% / {min_support})")    #(Min_sup = 20% / 1624.8))
    print(f"Total memory usage: {memory_usage:.2f} MB")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total number of frequent items: {total_frequent_items}")