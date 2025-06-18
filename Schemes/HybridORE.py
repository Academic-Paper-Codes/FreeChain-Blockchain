import hashlib
import os
from Crypto.Cipher import AES
from Crypto.Util import Counter
from Crypto.Util.Padding import pad, unpad
from EncodeORE import sort_encrypted_data
import hmac

def prf(key, data, output_bytes=16):
    return int.from_bytes(hmac.new(key, data, hashlib.sha256).digest()[:output_bytes], byteorder='big')


class CLWW:
    def __init__(self, key, bit_length=32):
        self.key = key
        self.bit_length = bit_length
        self.mod = 3 

    def encrypt(self, m):
        """加密整数m"""
        binary = bin(m)[2:].zfill(self.bit_length)
        ct = []
        for i in range(1, self.bit_length+1):
            prefix = binary[:i-1].ljust(self.bit_length-1, '0')
            mask = prf(self.key, f"{i}_{prefix}".encode(), 1) % self.mod
            bit = int(binary[i-1]) if i-1 < len(binary) else 0
            ct.append((mask + bit) % self.mod)
        return ct

    @staticmethod
    def compare(ct1, ct2):
        """比较两个密文 ct1 < ct2 返回 1, ct1 > ct2 返回 -1, ct1 == ct2 返回 0"""
        for i, (u1, u2) in enumerate(zip(ct1, ct2)):
            if u1 != u2:
                return 1 if (u2 - u1) % 3 == 1 else -1
        return 0
    

class LewiWuSmall:
    def __init__(self, key, domain_size=128):
        self.key = key
        self.N = domain_size

        # 1. 生成确定性排列
        self.perm = list(range(domain_size))
        for i in range(domain_size-1, 0, -1):
            j = prf(key, f"perm_{i}".encode()) % (i+1)
            self.perm[i], self.perm[j] = self.perm[j], self.perm[i]

        # 2. 固定 AES-ECB 对象，只建一次
        self.aes_key = prf(key, b'aes_key', 16).to_bytes(16, 'big')
        self._aes_ecb = AES.new(self.aes_key, AES.MODE_ECB)

        # 3. 逆排列
        self.inv_perm = [0] * domain_size
        for i in range(domain_size):
            self.inv_perm[self.perm[i]] = i

    def encrypt(self, x: int):
        pos = self.perm[x]
        right = []

        for i in range(self.N):
            # 比较值
            actual_j = self.inv_perm[i]
            cmp_val = 1 if actual_j < x else (-1 if actual_j > x else 0)
            cmp_bytes = cmp_val.to_bytes(4, 'big', signed=True)

            # keystream = AES_k(nonce||0)
            nonce = prf(self.key, f"nonce_{x}_{i}".encode(), 8).to_bytes(8, 'big')
            ks = self._aes_ecb.encrypt(nonce + b'\x00'*8)

            # XOR
            ct = bytes(a ^ b for a, b in zip(cmp_bytes, ks))
            right.append(nonce + ct)

        return (pos, right)

    def compare(self, ct1, ct2):
        pos1, _   = ct1
        _, right2 = ct2

        nonce_ct  = right2[pos1]
        nonce     = nonce_ct[:8]
        ct_bytes  = nonce_ct[8:]

        ks = self._aes_ecb.encrypt(nonce + b'\x00'*8)
        plain = bytes(a ^ b for a, b in zip(ct_bytes, ks))
        return int.from_bytes(plain, 'big', signed=True)


class HybridORE:
    def __init__(self, lw_key, clww_key, l1=4, l2=4):
        self.l1 = l1  # V部分的整数位数
        self.l2 = l2  # E部分的位数
        self.range_bits = l1 + l2  # 范围部分的位数
        self.range_enc = LewiWuSmall(lw_key, 2**self.range_bits)
        self.value_enc = CLWW(clww_key, bit_length=8)  # 调整为l1位


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


    def encode(self, m: int, d: int = 10) -> tuple[int, str]:
        """编码整数，返回 range_int（用于范围加密）和 value_bin（用于值加密）"""
        v, e = self._split_data(m, d)
        
        # 提取 V部分的整数部分（l1位）
        integer_part = int(v)
        integer_bin = bin(integer_part)[2:].zfill(self.l1)[-self.l1:]  # 固定 l1 位
        
        # 生成 range_int = E部分 + V的整数部分（二进制拼接后转整数）
        e_bin = bin(e)[2:].zfill(self.l2)[-self.l2:]  # E部分固定 l2 位
        range_bin = e_bin + integer_bin  # 拼接 E 和 V的整数部分
        range_int = int(range_bin, 2)  # 转换为整数
        
        # 生成 value_bin（包含整数+小数，供值加密使用）
        decimal_str = []
        remaining = v - integer_part
        for _ in range(4):  # 小数部分固定4位
            remaining *= 2
            bit = int(remaining)
            decimal_str.append(str(bit))
            remaining -= bit
        decimal_bin = ''.join(decimal_str)
        value_bin = integer_bin + decimal_bin  # 整数4位 + 小数4位
        
        return range_int, value_bin    
    
    def encrypt(self, m):
        """加密数值"""
        range_int, value_bin = self.encode(m)
        ct_range = self.range_enc.encrypt(range_int)
        ct_value = self.value_enc.encrypt(int(value_bin, 2))  # 值部分转整数加密
        return (ct_range, ct_value)
    
    def compare(self, ct1, ct2):
        """比较两个密文"""
        # 先比较范围部分
        range_cmp = self.range_enc.compare(ct1[0], ct2[0])
        if range_cmp != 0:
            return range_cmp
        # 范围相等时，比较值部分
        value_cmp = self.value_enc.compare(ct1[1], ct2[1])
        return value_cmp
    

# ========================
# 测试用例
# ========================
if __name__ == "__main__":
    # 生成密钥
    lw_key = os.urandom(16)
    clww_key = os.urandom(16)
    
    # # 测试CLWW
    # print("Testing CLWW:")
    # clww = CLWW(clww_key)
    # plains = [i for i in range(100)]
    # ct_list = [clww.encrypt(p) for p in plains]
    # print(f"Ciphertext list: {ct_list}")
    # for i in range(len(plains) - 1):
    #     print(f"Compare {plains[i]} vs {plains[i+1]}: {clww.compare(ct_list[i], ct_list[i+1])} (should be 1)") 

    # print("Testing LewiWuSmall:")
    # lw = LewiWuSmall(lw_key)
    # plains = [i for i in range(100)]
    # ct_list = [lw.encrypt(p) for p in plains]
    # for i in range(len(plains) - 1):
    #     print(f"Compare {plains[i]} vs {plains[i+1]}: {lw.compare(ct_list[i], ct_list[i+1])} (should be 1)")


    # # 测试HybridORE
    print("\nTesting HybridORE:")
    hore = HybridORE(lw_key, clww_key)
    num1 = 15  # 前8位: 00101101 = 45
    num2 = 16  # 前8位: 00101101 = 45 (相同范围)
    ct5 = hore.encrypt(num1)
    ct6 = hore.encrypt(num2)
    print(f"Compare {num1} vs {num2}: {hore.compare(ct5, ct6)} (should be 1)")
    print(f"ciphers: ct1 = {ct5}, ct2 = {ct6}")


    plains = [i for i in range(100)]
    ciphers = [hore.encrypt(p) for p in plains]
    
    sorted_ciphertexts = sort_encrypted_data(ciphers, hore.compare)

    sorted_indices = [ciphers.index(ct) for ct in sorted_ciphertexts]
    sorted_plaintexts = [plains[i] for i in sorted_indices]
    print("Sorted Order:", sorted_plaintexts)
    