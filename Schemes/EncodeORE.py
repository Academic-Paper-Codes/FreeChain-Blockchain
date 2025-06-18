import hashlib
import hmac
import secrets
from functools import cmp_to_key

class EncodeORE:
    def __init__(self, l1=8, l2=4, key=None):
        self.l1 = l1    # V部分的二进制长度
        self.l2 = l2    # E部分的二进制长度
        self.total_bits = l1 + l2
        self.key = key if key else secrets.token_bytes(16)  # 16字节密钥
    

    def _split_data(self, m: int, d: int) -> tuple[float, int]:
        """将整数m分解为v和e，使得m = v * d^e，其中1 ≤ v < d"""
        if m == 0:
            return 0.0, 0
        if d <= 1:
            raise ValueError("d must be greater than 1")
        e = 0
        abs_m = float(m)
        # 调整e使得d^e ≤ abs_m < d^(e+1)
        if abs_m >= 1:
            while abs_m >= d:
                abs_m /= d
                e += 1
        else:
            while abs_m < 1:
                abs_m *= d
                e -= 1
        return abs_m, e

    def encode(self, m: int, d: int = 10) -> str:
        """编码整数为二进制字符串（E部分 + V部分）"""
        v, e = self._split_data(m, d)
        # 处理V部分：定点数（整数4位 + 小数4位）
        integer_part = int(v)
        decimal_part = v - integer_part
        # 整数部分4位，左补零
        integer_bin = bin(integer_part)[2:].zfill(4)[-4:]
        # 小数部分4位，每次乘2取整
        decimal_str = []
        remaining = decimal_part
        for _ in range(8):
            remaining *= 2
            bit = int(remaining)
            decimal_str.append(str(bit))
            remaining -= bit
        decimal_bin = ''.join(decimal_str)
        v_bin = integer_bin + decimal_bin
        # 处理E部分，填充到l2位
        e_bin = bin(e)[2:].zfill(self.l2)[-self.l2:]
        return e_bin + v_bin

    def _prf(self, index: int, prefix: str) -> int:
        """伪随机函数，使用HMAC-SHA256生成模3的值"""
        # 将index和prefix转换为字节流
        index_bytes = index.to_bytes(4, 'big')       # 4字节大端序
        prefix_bytes = prefix.encode('utf-8')       # 二进制字符串转字节
        # 使用HMAC-SHA256
        hmac_obj = hmac.new(self.key, index_bytes + prefix_bytes, hashlib.sha256)
        digest = hmac_obj.digest()
        # 转换为0-2的整数
        return int.from_bytes(digest, 'big') % 3

    def encrypt(self, m: int) -> list[int]:
        """加密明文，返回密文列表"""
        encoded = self.encode(m)
        ciphertext = []
        for i in range(1, self.total_bits + 1):
            # 构造前缀：前i-1位 + 后续补零到总长度
            prefix_part = encoded[:i-1] if i > 1 else ''
            zeros_part = '0' * (self.total_bits - (i-1))
            prefix = prefix_part + zeros_part
            # 获取当前bit值（0或1）
            bit = int(encoded[i-1]) if i <= len(encoded) else 0
            # 计算PRF并生成密文元素
            mask = self._prf(i, prefix)
            u_i = (mask + bit) % 3
            ciphertext.append(u_i)
        return ciphertext

    def compare(self, ct1: list[int], ct2: list[int]) -> int:
        """比较两个密文，返回-1（ct1<ct2）, 0（相等）, 1（ct1>ct2）"""
        for u1, u2 in zip(ct1, ct2):
            if u1 != u2:
                # 判断差异类型（根据模3运算）
                diff = (u2 - u1) % 3
                return -1 if diff == 1 else 1
        return 0

def sort_encrypted_data(ciphertexts: list[list[int]], compare_func) -> list[list[int]]:
    """使用自定义比较函数对密文排序"""
    return sorted(ciphertexts, key=cmp_to_key(compare_func))

# 示例使用
if __name__ == "__main__":
    # 初始化参数（范围4位，值8位）
    ore = EncodeORE(l1=8, l2=4)
    plaintexts = [i for i in range(100)]  # 明文数据范围0-99

    # 加密所有明文
    ciphertexts = [ore.encrypt(m) for m in plaintexts]
    #print("Ciphertexts:", ciphertexts)
    print("Original Order:", plaintexts)

    # 对密文排序
    sorted_ciphertexts = sort_encrypted_data(ciphertexts, ore.compare)

    # 获取排序后的明文（仅用于验证，实际无法解密）
    sorted_indices = [ciphertexts.index(ct) for ct in sorted_ciphertexts]
    sorted_plaintexts = [plaintexts[i] for i in sorted_indices]
    print("Sorted Order:", sorted_plaintexts)