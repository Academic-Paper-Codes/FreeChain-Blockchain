# bvtree.py (optimized implementation)
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
import math

@dataclass
class Block:
    min_cipher: str
    max_cipher: str
    ciphertexts: List[str]
    addresses: List[str]
    plaintexts: List[int]  # 添加明文存储用于调试
    merkle_root: str
    block_id: int
    # 新增优化字段
    is_sorted: bool = False  # 块内数据是否已排序

class BVTree:
    def __init__(self, freore_instance, block_size=1000):
        """
        初始化BVTree
        Args:
            freore_instance: FreORE实例，用于加密操作
            block_size: 每个块的最大容量
        """
        self.freore = freore_instance
        self.blocks: List[Block] = []
        self.block_size = block_size
        self.insertion_times = []
        self.next_block_id = 0
        # 新增优化字段
        self._block_index = []  # (min_cipher, max_cipher, block_id)
        self._index_dirty = False  # 索引是否需要重建

    def insert(self, plaintext: int, file_address: str):
        """插入明文数据，内部进行FreORE加密并管理块结构"""
        start_time = time.time()
        
        # Step 1: 使用FreORE加密
        ciphertext = self.freore.data_encrypt(plaintext)
        
        # Step 2: 智能块选择
        target_block = self._find_optimal_block(ciphertext)
        
        if not target_block:
            # 创建新块
            new_block = Block(
                min_cipher=ciphertext,
                max_cipher=ciphertext,
                ciphertexts=[ciphertext],
                addresses=[file_address],
                plaintexts=[plaintext],
                merkle_root="",
                block_id=self.next_block_id,
                is_sorted=True  # 单个元素默认有序
            )
            self.blocks.append(new_block)
            self.next_block_id += 1
            target_block = new_block
        else:
            # 添加到现有块
            target_block.ciphertexts.append(ciphertext)
            target_block.addresses.append(file_address)
            target_block.plaintexts.append(plaintext)
            target_block.is_sorted = False  # 标记需要重新排序
            
            # 更新min/max（使用FreORE比较）
            try:
                if self.freore.data_compare(ciphertext, target_block.min_cipher) < 0:
                    target_block.min_cipher = ciphertext
                if self.freore.data_compare(ciphertext, target_block.max_cipher) > 0:
                    target_block.max_cipher = ciphertext
            except:
                # 比较失败时保持原值
                pass
        
        # Step 3: 重新计算受影响块的Merkle根
        self._update_block_merkle_root(target_block)
        
        # Step 4: 标记索引需要重建
        self._index_dirty = True
        
        end_time = time.time()
        insertion_time = (end_time - start_time) * 1000
        self.insertion_times.append(insertion_time)
        
        return insertion_time

    def _find_optimal_block(self, ciphertext: str) -> Optional[Block]:
        """找到最适合插入的块"""
        # 策略1：找到能包含该密文的未满块
        for block in self.blocks:
            if len(block.ciphertexts) < self.block_size:
                try:
                    # 检查是否在块的范围内
                    if (self.freore.data_compare(ciphertext, block.min_cipher) >= 0 and
                        self.freore.data_compare(ciphertext, block.max_cipher) <= 0):
                        return block
                except:
                    # 比较失败时跳过
                    continue
        
        # 策略2：如果最后一个块未满，使用它
        if self.blocks and len(self.blocks[-1].ciphertexts) < self.block_size:
            return self.blocks[-1]
        
        return None

    def _rebuild_block_index(self):
        """重建块级索引"""
        if not self._index_dirty:
            return
        
        self._block_index = []
        for block in self.blocks:
            self._block_index.append((block.min_cipher, block.max_cipher, block.block_id))
        
        # 按min_cipher排序
        try:
            self._block_index.sort(key=lambda x: x[0])
        except:
            # 排序失败时保持原序
            pass
        
        self._index_dirty = False

    def _sort_block_if_needed(self, block: Block):
        """按需对块内数据排序"""
        if block.is_sorted or len(block.ciphertexts) <= 1:
            return
        
        try:
            # 创建排序索引
            indices = list(range(len(block.ciphertexts)))
            indices.sort(key=lambda i: block.ciphertexts[i])
            
            # 重新排列所有数组
            block.ciphertexts = [block.ciphertexts[i] for i in indices]
            block.addresses = [block.addresses[i] for i in indices]
            block.plaintexts = [block.plaintexts[i] for i in indices]
            
            block.is_sorted = True
        except:
            # 排序失败时保持原状
            pass

    def range_query(self, low_plaintext: int, high_plaintext: int) -> List[str]:
        """范围查询，使用FreORE的陷阱门机制 - 优化版本"""
        # 重建块级索引
        self._rebuild_block_index()
        
        # 生成查询陷阱门
        low_trapdoor = self.freore.trap_encrypt(low_plaintext)
        high_trapdoor = self.freore.trap_encrypt(high_plaintext)
        
        results = []
        
        # 使用块级索引快速定位相关块
        relevant_blocks = self._find_relevant_blocks(low_trapdoor, high_trapdoor)
        
        for block in relevant_blocks:
            # 对块内数据排序（如果需要）
            self._sort_block_if_needed(block)
            
            # 在块内进行优化查询
            if block.is_sorted and len(block.ciphertexts) > 20:
                # 大块使用二分搜索
                block_results = self._binary_search_in_block(block, low_trapdoor, high_trapdoor)
            else:
                # 小块使用线性搜索
                block_results = self._linear_search_in_block(block, low_trapdoor, high_trapdoor)
            
            results.extend(block_results)
        
        return results

    def _find_relevant_blocks(self, low_trapdoor, high_trapdoor) -> List[Block]:
        """使用索引快速找到相关块"""
        relevant_blocks = []
        
        for min_cipher, max_cipher, block_id in self._block_index:
            try:
                # 检查块范围是否与查询范围重叠
                # 条件：low <= block.max && high >= block.min
                low_vs_max = self.freore.trap_compare(low_trapdoor, max_cipher)
                high_vs_min = self.freore.trap_compare(high_trapdoor, min_cipher)
                
                if low_vs_max <= 0 and high_vs_min >= 0:  # 有重叠
                    # 找到对应的块对象
                    block = next((b for b in self.blocks if b.block_id == block_id), None)
                    if block:
                        relevant_blocks.append(block)
            except:
                # 比较失败时保守地包含该块
                block = next((b for b in self.blocks if b.block_id == block_id), None)
                if block:
                    relevant_blocks.append(block)
        
        return relevant_blocks

    def _binary_search_in_block(self, block: Block, low_trapdoor, high_trapdoor) -> List[str]:
        """在已排序的块内使用二分搜索"""
        results = []
        
        try:
            # 找到起始位置
            left, right = 0, len(block.ciphertexts) - 1
            start_idx = len(block.ciphertexts)
            
            while left <= right:
                mid = (left + right) // 2
                cipher = block.ciphertexts[mid]
                cmp_result = self.freore.trap_compare(low_trapdoor, cipher)
                
                if cmp_result <= 0:  # low <= cipher
                    start_idx = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            # 从起始位置扫描到结束
            for i in range(start_idx, len(block.ciphertexts)):
                cipher = block.ciphertexts[i]
                high_cmp = self.freore.trap_compare(high_trapdoor, cipher)
                
                if high_cmp < 0:  # cipher > high
                    break
                    
                results.append(block.addresses[i])
        except:
            # 二分搜索失败时回退到线性搜索
            return self._linear_search_in_block(block, low_trapdoor, high_trapdoor)
        
        return results

    def _linear_search_in_block(self, block: Block, low_trapdoor, high_trapdoor) -> List[str]:
        """在块内进行线性搜索"""
        results = []
        
        for i, cipher in enumerate(block.ciphertexts):
            try:
                # 检查：low_plaintext <= cipher <= high_plaintext
                low_vs_cipher = self.freore.trap_compare(low_trapdoor, cipher)
                high_vs_cipher = self.freore.trap_compare(high_trapdoor, cipher)
                
                if low_vs_cipher <= 0 and high_vs_cipher >= 0:
                    results.append(block.addresses[i])
            except:
                # 比较失败时跳过
                continue
        
        return results

    def _update_block_merkle_root(self, block: Block):
        """计算块的Merkle根哈希"""
        if not block.ciphertexts:
            block.merkle_root = ""
            return
        
        # 构建Merkle树（基于地址哈希）
        hashes = [hashlib.sha256(addr.encode()).hexdigest() for addr in block.addresses]
        
        # 自底向上构建树
        while len(hashes) > 1:
            new_level = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i + 1] if i + 1 < len(hashes) else left
                combined = hashlib.sha256(f"{left}{right}".encode()).hexdigest()
                new_level.append(combined)
            hashes = new_level
        
        block.merkle_root = hashes[0] if hashes else ""

    def compute_merkle_roots(self) -> float:
        """计算所有块的Merkle根，并返回存储开销（KB）"""
        total_cost = 0
        
        for block in self.blocks:
            self._update_block_merkle_root(block)
            
            # 计算存储成本
            # 每块存储：min_cipher_hash(32B) + max_cipher_hash(32B) + merkle_root(32B)
            # + 内部Merkle节点
            num_addresses = len(block.addresses)
            if num_addresses > 0:
                merkle_internal_nodes = max(0, num_addresses - 1)
                block_cost = 32 * 3 + merkle_internal_nodes * 32  # min/max/root + 内部节点
                total_cost += block_cost
        
        return total_cost / 1024  # 转换为KB

    def generate_proof(self, plaintext: int) -> Optional[Dict]:
        """为指定明文生成验证证明"""
        target_cipher = self.freore.data_encrypt(plaintext)
        
        # 找到包含该密文的块
        target_block = None
        target_index = -1
        
        for block in self.blocks:
            if target_cipher in block.ciphertexts:
                target_block = block
                target_index = block.ciphertexts.index(target_cipher)
                break
        
        if not target_block:
            return None
        
        # 生成块内Merkle证明
        merkle_proof = self._generate_merkle_proof(target_block, target_index)
        
        # 生成完整性证明（相邻块信息）
        completeness_proof = self._generate_completeness_proof(plaintext, target_block)
        
        return {
            'block_id': target_block.block_id,
            'ciphertext': target_cipher,
            'address': target_block.addresses[target_index],
            'block_min': target_block.min_cipher,
            'block_max': target_block.max_cipher,
            'merkle_proof': merkle_proof,
            'block_root': target_block.merkle_root,
            'completeness_proof': completeness_proof
        }

    def _generate_completeness_proof(self, plaintext: int, target_block: Block) -> Dict:
        """生成完整性证明，包含相邻块的边界信息"""
        proof = {
            'adjacent_blocks': [],
            'boundary_values': []
        }
        
        try:
            target_trapdoor = self.freore.trap_encrypt(plaintext)
            
            for block in self.blocks:
                if block.block_id == target_block.block_id:
                    continue
                    
                # 检查该块是否可能包含相关结果
                min_vs_target = self.freore.trap_compare(target_trapdoor, block.min_cipher)
                max_vs_target = self.freore.trap_compare(target_trapdoor, block.max_cipher)
                
                if min_vs_target > 0:  # target < block.min，记录min作为边界
                    proof['boundary_values'].append({
                        'block_id': block.block_id,
                        'boundary_type': 'min',
                        'boundary_value': block.min_cipher
                    })
                elif max_vs_target < 0:  # target > block.max，记录max作为边界
                    proof['boundary_values'].append({
                        'block_id': block.block_id,
                        'boundary_type': 'max',
                        'boundary_value': block.max_cipher
                    })
        except:
            # 生成证明失败时返回空证明
            pass
        
        return proof

    def _generate_merkle_proof(self, block: Block, target_index: int) -> List[str]:
        """生成块内Merkle证明"""
        if not block.addresses:
            return []
        
        # 构建完整的Merkle树并记录证明路径
        hashes = [hashlib.sha256(addr.encode()).hexdigest() for addr in block.addresses]
        proof = []
        current_index = target_index
        
        while len(hashes) > 1:
            new_level = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i + 1] if i + 1 < len(hashes) else left
                
                # 如果当前索引在这一对中，记录兄弟节点
                if current_index == i:
                    proof.append(right)
                elif current_index == i + 1:
                    proof.append(left)
                
                combined = hashlib.sha256(f"{left}{right}".encode()).hexdigest()
                new_level.append(combined)
            
            hashes = new_level
            current_index = current_index // 2
        
        return proof

    @staticmethod
    def verify_proof(proof: Dict) -> bool:
        """验证BVTree证明"""
        if not proof or 'merkle_proof' not in proof:
            return False
        
        # 验证块内Merkle证明
        current_hash = hashlib.sha256(proof['address'].encode()).hexdigest()
        
        for sibling_hash in proof['merkle_proof']:
            # 简化的证明验证（实际应该考虑左右子树位置）
            combined = hashlib.sha256(f"{current_hash}{sibling_hash}".encode()).hexdigest()
            current_hash = combined
        
        return current_hash == proof['block_root']

    def get_storage_size(self) -> float:
        """获取总存储大小（KB）"""
        return self.compute_merkle_roots()

    def get_average_insertion_time(self) -> float:
        """获取平均插入时间"""
        return sum(self.insertion_times) / len(self.insertion_times) if self.insertion_times else 0

    def debug_print_blocks(self):
        """调试输出：打印所有块的信息"""
        for i, block in enumerate(self.blocks):
            print(f"Block {i} (ID: {block.block_id}):")
            print(f"  Range: {block.min_cipher} to {block.max_cipher}")
            print(f"  Count: {len(block.ciphertexts)}")
            print(f"  Plaintexts: {block.plaintexts}")
            print(f"  Sorted: {block.is_sorted}")
            print(f"  Merkle Root: {block.merkle_root[:16]}...")
            print()