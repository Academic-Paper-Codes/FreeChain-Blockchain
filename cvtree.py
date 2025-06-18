# cvtree.py (optimized implementation)
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, List
import time
import hmac

@dataclass
class CVNode:
    identifier: str  # 节点标识符
    children: Dict[str, 'CVNode']  # 子节点字典 
    hash_value: str  # 当前节点的哈希值
    file_pointer: Optional[str] = None  # 指向加密文件的指针
    # 新增优化字段
    min_cipher: Optional[str] = None  # 子树中最小密文
    max_cipher: Optional[str] = None  # 子树中最大密文
    cipher_count: int = 0  # 子树中密文数量

class CVTree:
    def __init__(self, freore_instance):
        """
        初始化CVTree
        Args:
            freore_instance: FreORE实例，用于加密操作
        """
        self.freore = freore_instance
        self.root = CVNode(identifier="", children={}, hash_value="")
        self.ciphertext_to_address = {}  # 存储密文到地址的映射
        self.plaintext_to_ciphertext = {}  # 存储明文到密文的映射
        self.node_count = 0  # 节点总数
        self.insertion_times = []  # 记录插入时间
        # 新增优化字段
        self._sorted_ciphers = []  # 排序后的密文列表
        self._index_dirty = False  # 索引是否需要重建

    def insert(self, plaintext: int, file_address: str):
        """插入明文数据，内部进行FreORE加密并构建前缀树"""
        start_time = time.time()
        
        # Step 1: 使用FreORE加密
        ciphertext = self.freore.data_encrypt(plaintext)
        
        # Step 2: 构建前缀树路径（基于三进制密文）并优化范围信息
        node = self.root
        path_nodes = [node]  # 记录路径上的所有节点
        
        for i, char in enumerate(ciphertext):
            if char not in node.children:
                node.children[char] = CVNode(
                    identifier=char, 
                    children={}, 
                    hash_value="",
                    min_cipher=ciphertext,
                    max_cipher=ciphertext,
                    cipher_count=0
                )
                self.node_count += 1
            
            node = node.children[char]
            path_nodes.append(node)
        
        # Step 3: 设置叶子节点的文件指针
        node.file_pointer = file_address
        self.ciphertext_to_address[ciphertext] = file_address
        self.plaintext_to_ciphertext[plaintext] = ciphertext
        
        # Step 4: 更新路径上所有节点的范围信息
        self._update_range_info_along_path(path_nodes, ciphertext)
        
        # Step 5: 重新计算哈希值（从叶子到根）
        self._update_hashes_along_path(ciphertext)
        
        # Step 6: 标记索引需要重建
        self._index_dirty = True
        
        end_time = time.time()
        insertion_time = (end_time - start_time) * 1000  # 转换为毫秒
        self.insertion_times.append(insertion_time)
        
        return insertion_time

    def _update_range_info_along_path(self, path_nodes: List[CVNode], ciphertext: str):
        """更新路径上所有节点的范围信息"""
        for node in path_nodes:
            # 更新最小值
            if node.min_cipher is None or self.freore.data_compare(ciphertext, node.min_cipher) < 0:
                node.min_cipher = ciphertext
            
            # 更新最大值
            if node.max_cipher is None or self.freore.data_compare(ciphertext, node.max_cipher) > 0:
                node.max_cipher = ciphertext
            
            # 增加计数
            node.cipher_count += 1

    def _rebuild_sorted_index(self):
        """重建排序索引"""
        if not self._index_dirty:
            return
        
        # 按照FreORE的比较规则排序
        cipher_list = list(self.ciphertext_to_address.keys())
        cipher_list.sort(key=lambda x: x)  # 简化为字典序排序
        self._sorted_ciphers = cipher_list
        self._index_dirty = False

    def range_query(self, low_plaintext: int, high_plaintext: int) -> List[str]:
        """范围查询，使用FreORE的陷阱门机制 - 优化版本"""
        # 生成查询陷阱门
        low_trapdoor = self.freore.trap_encrypt(low_plaintext)
        high_trapdoor = self.freore.trap_encrypt(high_plaintext)
        
        # 智能选择查询策略
        total_data = len(self.ciphertext_to_address)
        estimated_range = abs(high_plaintext - low_plaintext)
        
        # 如果查询范围较小或数据量较大，使用前缀树剪枝
        if estimated_range < total_data * 0.1 and total_data > 100:
            return self._range_query_with_pruning(low_trapdoor, high_trapdoor)
        else:
            # 否则使用索引查询
            return self._range_query_with_index(low_trapdoor, high_trapdoor)

    def _range_query_with_pruning(self, low_trapdoor, high_trapdoor) -> List[str]:
        """使用前缀树剪枝的范围查询"""
        results = []
        self._dfs_range_search(self.root, "", low_trapdoor, high_trapdoor, results)
        return results

    def _dfs_range_search(self, node: CVNode, current_path: str, 
                         low_trapdoor, high_trapdoor, results: List[str]):
        """深度优先搜索 + 剪枝优化"""
        # 剪枝条件：检查当前节点的范围是否与查询范围有交集
        if node.min_cipher and node.max_cipher:
            # 检查：low > node.max 或 high < node.min 则剪枝
            try:
                low_vs_max = self.freore.trap_compare(low_trapdoor, node.max_cipher)
                high_vs_min = self.freore.trap_compare(high_trapdoor, node.min_cipher)
                
                if low_vs_max > 0 or high_vs_min < 0:
                    return  # 剪枝！整个子树都不在范围内
            except:
                # 如果比较失败，继续搜索（保持正确性）
                pass
        
        # 如果是叶子节点，检查是否匹配
        if node.file_pointer:
            cipher = current_path
            try:
                low_cmp = self.freore.trap_compare(low_trapdoor, cipher)
                high_cmp = self.freore.trap_compare(high_trapdoor, cipher)
                
                if low_cmp <= 0 and high_cmp >= 0:
                    results.append(node.file_pointer)
            except:
                # 如果比较失败，跳过这个节点
                pass
            return
        
        # 递归搜索子节点
        for char in sorted(node.children.keys()):
            child = node.children[char]
            new_path = current_path + char
            self._dfs_range_search(child, new_path, low_trapdoor, high_trapdoor, results)

    def _range_query_with_index(self, low_trapdoor, high_trapdoor) -> List[str]:
        """使用排序索引的范围查询"""
        self._rebuild_sorted_index()
        
        results = []
        
        # 使用二分搜索找到起始位置
        left, right = 0, len(self._sorted_ciphers) - 1
        start_idx = len(self._sorted_ciphers)
        
        # 找到第一个 >= low 的位置
        while left <= right:
            mid = (left + right) // 2
            cipher = self._sorted_ciphers[mid]
            try:
                cmp_result = self.freore.trap_compare(low_trapdoor, cipher)
                if cmp_result <= 0:  # low <= cipher
                    start_idx = mid
                    right = mid - 1
                else:
                    left = mid + 1
            except:
                # 比较失败时保守处理
                left = mid + 1
        
        # 从起始位置开始扫描
        for i in range(start_idx, len(self._sorted_ciphers)):
            cipher = self._sorted_ciphers[i]
            try:
                high_cmp = self.freore.trap_compare(high_trapdoor, cipher)
                if high_cmp < 0:  # cipher > high，停止搜索
                    break
                results.append(self.ciphertext_to_address[cipher])
            except:
                # 比较失败时继续下一个
                continue
        
        return results

    def _update_hashes_along_path(self, ciphertext: str):
        """沿路径更新哈希值"""
        def _update_node_hash(node: CVNode, path_remaining: str) -> str:
            if not path_remaining:  # 叶子节点
                content = node.file_pointer or ""
                node.hash_value = hashlib.sha256(content.encode()).hexdigest()
                return node.hash_value
            
            # 递归更新子节点
            next_char = path_remaining[0]
            if next_char in node.children:
                child_hash = _update_node_hash(
                    node.children[next_char], 
                    path_remaining[1:]
                )
            
            # 计算当前节点哈希（基于所有子节点）
            child_hashes = []
            for char in sorted(node.children.keys()):
                child = node.children[char]
                child_hashes.append(child.hash_value)
            
            combined = f"{node.identifier}" + "".join(child_hashes)
            node.hash_value = hashlib.sha256(combined.encode()).hexdigest()
            return node.hash_value
        
        _update_node_hash(self.root, ciphertext)

    def compute_hashes(self) -> float:
        """计算所有节点的哈希值，并返回存储成本（KB）"""
        def _compute(node: CVNode, current_ciphertext: str) -> str:
            if not node.children:  # 叶子节点
                content = node.file_pointer or self.ciphertext_to_address.get(current_ciphertext, "")
                node.hash_value = hashlib.sha256(content.encode()).hexdigest()
                return node.hash_value
            
            # 非叶子节点：按identifier排序子节点
            child_hashes = []
            for char in sorted(node.children.keys()):
                child = node.children[char]
                child_hash = _compute(child, current_ciphertext + char)
                child_hashes.append(child_hash)
            
            combined = f"{node.identifier}" + "".join(child_hashes)
            node.hash_value = hashlib.sha256(combined.encode()).hexdigest()
            return node.hash_value

        _compute(self.root, current_ciphertext="")
        # 存储成本 = (节点数*32B + 地址数*32B) / 1024 → KB
        return (self.node_count * 32 + len(self.ciphertext_to_address) * 32) / 1024

    def _path_contains_range(self, path: str, low_cipher: str, high_cipher: str) -> bool:
        """判断路径是否可能包含范围内的密文（基于FreORE比较）"""
        # 简化判断：检查路径前缀是否可能产生范围内的结果
        if len(path) > len(low_cipher) or len(path) > len(high_cipher):
            return True
        
        # 构造最小和最大可能的密文（补充剩余位）
        min_possible = path + "0" * (len(low_cipher) - len(path))
        max_possible = path + "2" * (len(high_cipher) - len(path))
        
        try:
            # 使用FreORE比较规则
            low_cmp = self.freore.data_compare(min_possible, low_cipher)
            high_cmp = self.freore.data_compare(max_possible, high_cipher)
            
            # 如果路径的可能范围与查询范围有重叠
            return low_cmp <= 0 and high_cmp >= 0
        except:
            # 比较失败时保守返回True
            return True

    def generate_proof(self, plaintext: int) -> Optional[Dict]:
        """生成指定明文的Merkle证明"""
        if plaintext not in self.plaintext_to_ciphertext:
            return None
        
        ciphertext = self.plaintext_to_ciphertext[plaintext]
        address = self.ciphertext_to_address[ciphertext]
        
        proof = {'address': address, 'ciphertext': ciphertext, 'siblings': []}
        node = self.root
        current_path = ""
        
        for char in ciphertext:
            parent = node
            node = node.children.get(char)
            if not node:
                return None
            
            # 获取兄弟节点哈希
            sorted_children = sorted(parent.children.items(), key=lambda x: x[0])
            sorted_ids = [c for c, _ in sorted_children]
            sorted_hashes = [child.hash_value for _, child in sorted_children]
            
            try:
                index_in_siblings = sorted_ids.index(char)
            except ValueError:
                return None
                
            proof['siblings'].append({
                'parent_identifier': parent.identifier,
                'sorted_hashes': sorted_hashes,
                'index': index_in_siblings
            })
            current_path += char
            
        return proof

    @staticmethod
    def verify_proof(proof: Dict, root_hash: str) -> bool:
        """验证Merkle证明"""
        if not proof or 'address' not in proof or 'siblings' not in proof:
            return False
            
        current_hash = hashlib.sha256(proof['address'].encode()).hexdigest()
        
        for entry in reversed(proof['siblings']):
            parent_id = entry['parent_identifier']
            sorted_hashes = entry['sorted_hashes']
            index = entry['index']
            
            if index < 0 or index >= len(sorted_hashes) or sorted_hashes[index] != current_hash:
                return False
            
            combined = f"{parent_id}" + "".join(sorted_hashes)
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
            
        return current_hash == root_hash

    @property
    def root_hash(self) -> str:
        """获取根节点哈希值"""
        return self.root.hash_value

    def get_average_insertion_time(self) -> float:
        """获取平均插入时间"""
        return sum(self.insertion_times) / len(self.insertion_times) if self.insertion_times else 0

    def get_storage_size(self) -> float:
        """获取存储大小（KB）"""
        return self.compute_hashes()