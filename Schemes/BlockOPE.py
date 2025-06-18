import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import random

# 加密工具类（AES对称加密）
class CryptoUtils:
    @staticmethod
    def encrypt(key, plaintext):
        plaintext_str = f"{float(plaintext):.10f}"
        cipher = AES.new(key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext_str.encode())
        return (cipher.nonce, ciphertext, tag)

    @staticmethod
    def decrypt(key, nonce, ciphertext, tag):
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return float(plaintext.decode())

# OPE树节点
class OPETreeNode:
    def __init__(self, ciphertext, code):
        self.nonce = ciphertext[0]
        self.ct = ciphertext[1]  # 密文
        self.tag = ciphertext[2]
        self.code = code              # 保序编码
        self.left = None               # 左子节点
        self.right = None              # 右子节点
        self.udz = []                  # 未决区（UDZ）
        self.udz_direction = None  

# BlockOPE核心类
class BlockOPE:
    def __init__(self, initial_code_space=2**30):
        self.root = None
        self.code_space = initial_code_space  # 编码空间大小（例如2^48）
        self.version = 0                       # 版本号（用于检测更新冲突）
        self.cache = {}                        # 轻量级查询缓存
        self.cache_enabled = False  # 是否启用缓存
        self.node_count = 0  # 节点计数器

    # ------------ 编码阶段 ------------
    def encode(self, plaintext, key):
        # 生成密文
        nonce, ciphertext, tag = CryptoUtils.encrypt(key, str(plaintext))
        # 交互式遍历OPE树生成编码
        code, path, version = self._interactive_traversal(plaintext, key)
        # 添加随机噪声
        # epsilon = random.randint(-10, 10)  # 论文中的随机编码噪声
        #final_code = code + epsilon
        final_code = code + 0 # 这里的epsilon可以根据需要调整
        # 返回编码结果（包含路径和版本）
        return {
            'ciphertext': (nonce, ciphertext, tag),
            'code': final_code,
            'path': path,
            'version': version,
        }
    
    def encrypt(self, plaintext, key):
        success = False
        attempts = 0
        while not success and attempts < 3:
            attempts += 1
            # 生成编码
            tx = self.encode(plaintext, key)
            # 尝试执行交易
            success = self.execute_transaction(tx)
            return tx
        
        if not success:
            raise Exception("Transaction failed due to version conflict or UDZ overflow.")

    def _interactive_traversal(self, plaintext, key):
        # 模拟DO与BCN的交互遍历过程
        path = []
        current = self.root
        parent_code = [0, self.code_space]  # 初始编码区间 [low, high]
        version = self.version

        while current is not None:
            # DO解密当前节点密文
            decrypted = float(CryptoUtils.decrypt(
                key, 
                current.nonce, 
                current.ct,
                current.tag
                ))
            # 比较明文大小
            if plaintext < decrypted:
                path.append('L')
                parent_code[1] = current.code
                current = current.left
            else:
                path.append('R')
                parent_code[0] = current.code
                current = current.right

        # 计算中间编码（论文中的Encode函数）
        code = parent_code[0] + (parent_code[1] - parent_code[0]) // 2
        return code, path, version
    
    def compare(self, c1, c2):
        """
        比较两个加密结果的大小
        :param c1: encode()返回的加密结果字典
        :param c2: encode()返回的加密结果字典
        :return: -1(c1 < c2), 0(c1 == c2), 1(c1 > c2)
        """
        # 先直接比较密文是否相同
        if c1['ciphertext'] == c2['ciphertext']:
            return 0
        
        # 获取路径序列
        path1 = c1['path']
        path2 = c2['path']
        
        # 逐级比较路径
        min_depth = min(len(path1), len(path2))
        for i in range(min_depth):
            if path1[i] != path2[i]:
                # 路径分叉点判断
                if path1[i] == 'L' and path2[i] == 'R':
                    return -1  # c1在左分支，c2在右分支 → c1 < c2
                else:
                    return 1   # 其他分叉情况均认为c1 > c2
        
        # 路径前缀相同，比较路径长度
        if len(path1) > len(path2):
            # c1路径更深，查看其下一步方向
            next_step = path1[min_depth] if len(path1) > min_depth else None
            return 1 if next_step == 'L' else -1
        elif len(path2) > len(path1):
            next_step = path2[min_depth] if len(path2) > min_depth else None
            return -1 if next_step == 'L' else 1
        else:
            # 路径完全一致，直接比较编码值
            if c1['code'] == c2['code']:
                return 0
            return -1 if c1['code'] < c2['code'] else 1

    # ------------ 执行阶段 ------------
    def execute_transaction(self, tx):
        # 检查版本冲突
        if tx['version'] != self.version:
            # 触发重平衡
            # TODO: 实现重平衡逻辑

            return False  # 更新冲突（UC）

        # 插入节点到OPE树
        node = OPETreeNode(tx['ciphertext'], tx['code'])
        self.node_count += 1  
        if not self.root:
            self.root = node
            return True

        current = self.root
        for direction in tx['path']:
            if direction == 'L':
                if not current.left:
                    current.left = node
                    return True
                current = current.left
            else:
                if not current.right:
                    current.right = node
                    return True
                current = current.right

        # 处理相同位置冲突（SCC），使用UDZ
        if len(current.udz) < 3:  # UDZ容量假设为3
            current.udz.append(node)
            return True
        else:
            # UDZ已满，触发重排
            self._rearrange_udz(current)
            return False  # 需要重新提交
        
    def _rearrange_udz(self, node):
        # 对UDZ中的节点重新编码
        sorted_nodes = sorted(node.udz, key=lambda x: x.code)
        # 清空UDZ并插入为子树
        node.udz = []
        self._build_subtree(node, sorted_nodes)
        self.version += 1 

    def _build_subtree(self, parent, nodes):
        if not nodes:
            return
        
        # 按编码排序
        sorted_nodes = sorted(nodes, key=lambda x: x.code)

        # 创建平衡子树
        root = self._create_balanced_subtree(sorted_nodes)

        self._reassign_codes(root, 0, self.code_space)

        self._attach_subtree(parent, root)

    def _create_balanced_subtree(self, nodes):
        """创建平衡的子树"""
        if not nodes:
            return None
        mid = len(nodes) // 2
        node = nodes[mid]
        node.left = self._create_balanced_subtree(nodes[:mid])
        node.right = self._create_balanced_subtree(nodes[mid+1:])
        return node

    def _reassign_codes(self, root, low, high):
        if not root:
            return
        if root.left:
            self._reassign_codes(root.left, low, root.code)
        
        root.code = low + (high - low) // 2  # 重新计算编码

        if root.right:
            self._reassign_codes(root.right, root.code, high)
    
    def _attach_subtree(self, parent, subtree):
        if parent.udz_direction == 'L':
            parent.left = subtree
        else:
            parent.right = subtree
        parent.udz_direction = None 

    # ------------ 查询阶段 ------------
    def query_range(self, low, high):
        # 使用缓存优化
        if (low, high) in self.cache:
            return self.cache[(low, high)]

        results = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node:
                if low <= node.code <= high:
                    results.append((node.nonce, node.ct, node.tag))
                    stack.append(node.left)
                    stack.append(node.right)
                elif node.code < low:
                    stack.append(node.right)
                else:
                    stack.append(node.left)
        # 更新缓存
        self.cache[(low, high)] = results
        return results
    def query_point(self, plaintext, key):
        """
        单点查询函数
        :param plaintext: 要查找的明文值
        :param key: 解密密钥
        :return: 字典包含 {'found': bool, 'path': list, 'node': OPETreeNode or None}
        """
        if self.root is None:
            return {'found': False, 'path': [], 'node': None}
        
        path = []
        current = self.root
        
        while current is not None:
            # 解密当前节点的值
            try:
                decrypted_value = CryptoUtils.decrypt(
                    key, 
                    current.nonce, 
                    current.ct, 
                    current.tag
                )
                
                # 比较明文值
                if abs(float(plaintext) - decrypted_value) < 1e-10:  # 浮点数精度比较
                    # 找到目标节点
                    return {'found': True, 'path': path, 'node': current}
                elif float(plaintext) < decrypted_value:
                    # 向左子树搜索
                    path.append('L')
                    current = current.left
                else:
                    # 向右子树搜索
                    path.append('R')
                    current = current.right
                    
            except Exception as e:
                # 解密失败，可能是密钥错误
                return {'found': False, 'path': path, 'node': None, 'error': str(e)}
        
        # 搜索完毕但未找到
        return {'found': False, 'path': path, 'node': None}

    def query_point_with_udz(self, plaintext, key):
        """
        带UDZ检查的单点查询函数
        :param plaintext: 要查找的明文值
        :param key: 解密密钥
        :return: 字典包含 {'found': bool, 'path': list, 'node': OPETreeNode or None, 'in_udz': bool}
        """
        if self.root is None:
            return {'found': False, 'path': [], 'node': None, 'in_udz': False}
        
        path = []
        current = self.root
        
        while current is not None:
            try:
                # 解密当前节点的值
                decrypted_value = CryptoUtils.decrypt(
                    key, 
                    current.nonce, 
                    current.ct, 
                    current.tag
                )
                
                # 比较明文值
                if abs(float(plaintext) - decrypted_value) < 1e-10:
                    # 找到目标节点
                    return {'found': True, 'path': path, 'node': current, 'in_udz': False}
                
                # 检查当前节点的UDZ
                for udz_node in current.udz:
                    udz_decrypted = CryptoUtils.decrypt(
                        key,
                        udz_node.nonce,
                        udz_node.ct,
                        udz_node.tag
                    )
                    if abs(float(plaintext) - udz_decrypted) < 1e-10:
                        # 在UDZ中找到目标节点
                        return {'found': True, 'path': path, 'node': udz_node, 'in_udz': True}
                
                # 继续向下搜索
                if float(plaintext) < decrypted_value:
                    path.append('L')
                    current = current.left
                else:
                    path.append('R')
                    current = current.right
                    
            except Exception as e:
                return {'found': False, 'path': path, 'node': None, 'in_udz': False, 'error': str(e)}
        
        # 搜索完毕但未找到
        return {'found': False, 'path': path, 'node': None, 'in_udz': False}

    # ------------ 工具方法 ------------
    def decrypt_value(self, key, ciphertext):
        nonce, ct, tag = ciphertext
        return float(CryptoUtils.decrypt(key, nonce, ct, tag))


    # 辅助方法 ----------------------------------------------------------------
    def _find_udz_node(self, node):
        """查找包含UDZ的节点（包括非叶子节点）"""
        if node is None:
            return None
        # 优先检查当前节点
        if node.udz:
            return node
        # 递归搜索左子树
        left_res = self._find_udz_node(node.left)
        if left_res:
            return left_res
        # 递归搜索右子树
        return self._find_udz_node(node.right)

    def _get_tree_height(self, node):
        """计算子树高度"""
        if node is None:
            return 0
        return 1 + max(self._get_tree_height(node.left), 
                      self._get_tree_height(node.right))
    
    def in_order_traversal(self, node):
        """中序遍历树，返回节点编码列表"""
        if node:
            self.in_order_traversal(node.left)
            print(f"Node Code: {node.code}, Nonce: {node.nonce.hex()}, CT: {node.ct.hex()}, Tag: {node.tag.hex()}")
            self.in_order_traversal(node.right)
    
    def get_storage_size(self):
        """计算OPE树的存储大小"""
        total_size = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node:
                # 每个节点存储：nonce(16B) + ct(16B) + tag(16B) + code(8B) + address * 2
                total_size += 16 + 16 + 16 + 8 + 16 + 16
                stack.append(node.left)
                stack.append(node.right)
        return total_size

if __name__ == "__main__":
    ope = BlockOPE()
    key = get_random_bytes(16)
    plaintexts = [i for i in range(10)]
    print("Plaintexts: ", plaintexts)
    for pt in plaintexts:
        ope.encrypt(pt, key)
    print("OPE Tree constructed with {} nodes.".format(ope.node_count))
    print("Root of the tree", ope.root.code if ope.root else None)
    ope.in_order_traversal(ope.root) 

    
    results = ope.query_range(0, 2**30)
    print("Query Results: ", results)


    ope = BlockOPE()
    key = get_random_bytes(16)
    plaintexts = [i for i in range(10)]
    print("Plaintexts: ", plaintexts)
    
    # 插入数据
    for pt in plaintexts:
        val = ope.encrypt(pt, key)
        print(val)

    
    print("OPE Tree constructed with {} nodes.".format(ope.node_count))
    
    # 测试单点查询
    test_values = [3, 7, 15, -1]  # 包含存在和不存在的值
    
    for val in test_values:
        result = ope.query_point(val, key)
        if result['found']:
            print(f"Found {val} at path: {' -> '.join(result['path']) if result['path'] else 'ROOT'}")
        else:
            print(f"Value {val} not found. Search path: {' -> '.join(result['path']) if result['path'] else 'No path'}")
        
        # 也可以测试带UDZ的版本
        result_udz = ope.query_point_with_udz(val, key)
        if result_udz['found']:
            location = "UDZ" if result_udz['in_udz'] else "Tree"
            print(f"  (UDZ check: Found in {location})")
