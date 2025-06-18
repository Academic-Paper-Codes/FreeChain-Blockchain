import hashlib
import hmac
import random
from functools import cmp_to_key


class FreORE:
    def __init__(self, d: int, alpha: int, beta: int, gamma: int, pfk: bytes, nx: int, ny: int):
        self.d = d          
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pfk = pfk
        self.nx = nx
        self.ny = ny

    def _split_data(self, m: int, d: int) -> tuple[float, int]:
        """m = v * d^e，其中1 ≤ v < d"""
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


    def data_encrypt(self, m: int) -> str:
        """加密整数m为密文字符串"""
        # 步骤1: 数据分割 m = x * d^y
        x, y = self._split_data(m, self.d)


        integer_part = int(x)  # 整数部分
        decimal_part = x - integer_part  # 小数部分    
        integet_bin = bin(integer_part)[2:].zfill(self.nx)[-self.nx:]  # 整数部分4位，左补零
        decimal_str = []
        remaining = decimal_part
        for _ in range(8):
            remaining *= 2
            bit = int(remaining)
            decimal_str.append(str(bit))
            remaining -= bit
        decimal_bin = ''.join(decimal_str)  # 小数部分4位，每次乘2取整
        x_bin = integet_bin + decimal_bin  
        y_bin = bin(y)[2:].zfill(self.ny)[-self.ny:]  
        v = x_bin + y_bin  # 拼接整数和小数部分

        r = random.randint(0, self.gamma)  # 随机扰动值
        if len(v) == 0:
            v1, rest = 0, 0
        else:
            v1 = int(v[0])         # 取第一个字符作为v1
            rest = int(v[1:] or 0)

        v1_perturbed = v1 * self.alpha + rest * self.beta + r  
        perturbed_bits = bin(v1_perturbed)[2:].zfill(self.nx)[-self.nx:]
        new_x_bin = perturbed_bits + v[self.nx:]  # 更新x的值


        N = len(new_x_bin)  
        cipher = []

        for j in range(1, N+1):  # j从1到N
            prefix = new_x_bin[:j-1].ljust(N, '0')  # 前缀
            index_bytes = j.to_bytes(4, 'big')
            message = f"{prefix}".encode('utf-8')
            hmac_digest = hmac.new(self.pfk, index_bytes + message, hashlib.sha256).digest()
            F_val = int.from_bytes(hmac_digest, byteorder='big') % 3
            bj = int(new_x_bin[j-1])   
            cj = (F_val + bj) % 3
            cipher.append(str(cj))
        # print("cipher:", "".join(cipher))
        return "".join(cipher)
    
    def _compare(self, c1: str, c2: str) -> int:
        """比较两个密文，返回-1, 0, 1分别表示小于、等于、大于"""
        # 将密文字符串转换为整数列表，例如 "102" → [1, 0, 2]
        c1 = [int(bit) for bit in c1]
        c2 = [int(bit) for bit in c2]

        # 检查长度是否一致
        if len(c1) != len(c2):
            print(f"cur c1 = {c1} cur c2 ={c2} , len(c1) = {len(c1)}, len(c2) = {len(c2)}")
            raise ValueError("len(c1) != len(c2)")

        N = len(c1)
        b = 1

        # 从高位到低位逐位比较
        for i in range(N):
            if c1[i] == c2[i]:
                continue
            elif c1[i] == (c2[i] + 1) % 3:
                b = 2
                break
            elif c2[i] == (c1[i] + 1) % 3:
                b = 0
                break

        return b
    
    def trap_encrypt(self, m: int) -> tuple[str, str]:
        """加密整数m为陷阱门密文"""
        # 步骤1: 数据分割 m = x * d^y
        x, y = self._split_data(m, self.d)

        integer_part = int(x)
        decimal_part = x - integer_part
        integet_bin = bin(integer_part)[2:].zfill(self.nx)[-self.nx:]
        decimal_str = []
        remaining = decimal_part
        for _ in range(8):
            remaining *= 2
            bit = int(remaining)
            decimal_str.append(str(bit))
            remaining -= bit
        decimal_bin = ''.join(decimal_str)
        x_bin = integet_bin + decimal_bin
        y_bin = bin(y)[2:].zfill(self.ny)[-self.ny:]
        v = x_bin + y_bin

        if len(v) == 0:
            v1, rest = 0, 0
        else:
            v1 = int(v[0])
            rest = int(v[1:])

        v_low_1 = v1 * self.alpha + rest * self.beta + 0
        v_high_1 = v1 * self.alpha + rest * self.beta + self.gamma

        v_q_l = bin(v_low_1)[2:].zfill(self.nx)[-self.nx:] + v[self.nx:]
        v_q_h = bin(v_high_1)[2:].zfill(self.nx)[-self.nx:] + v[self.nx:]

        cipher_l = []
        cipher_h = []
        for j in range(1, len(v_q_l) + 1):
            prefix = v_q_l[:j - 1].ljust(len(v_q_l), '0')
            index_bytes = j.to_bytes(4, 'big')
            message = f"{prefix}".encode('utf-8')
            hmac_digest = hmac.new(self.pfk, index_bytes + message, hashlib.sha256).digest()
            F_val = int.from_bytes(hmac_digest, byteorder='big') % 3
            bj = int(v_q_l[j - 1])
            cj_l = (F_val + bj) % 3
            cipher_l.append(str(cj_l))

        for j in range(1, len(v_q_h) + 1):
            prefix = v_q_h[:j - 1].ljust(len(v_q_h), '0')
            index_bytes = j.to_bytes(4, 'big')
            message = f"{prefix}".encode('utf-8')
            hmac_digest = hmac.new(self.pfk, index_bytes + message, hashlib.sha256).digest()
            F_val = int.from_bytes(hmac_digest, byteorder='big') % 3
            bj = int(v_q_h[j - 1])
            cj_h = (F_val + bj) % 3
            cipher_h.append(str(cj_h))
        
        c_q_l = "".join(cipher_l)
        c_q_h = "".join(cipher_h)

        # print("trapdoor low:", c_q_l)
        # print("trapdoor high:", c_q_h)
        
        return (c_q_l, c_q_h)
    
    def trap_compare(self, c_q: tuple[str, str], c_d: str) -> int:
        """比较陷阱门密文和数据密文，返回-1, 0, 1分别表示小于、等于、大于"""
        c_q_l, c_q_h = c_q
        b = 1  # 默认相等
        c_d = [int(ch) for ch in c_d]  # 转换为整数列表
        c_q_l = [int(ch) for ch in c_q_l]
        c_q_h = [int(ch) for ch in c_q_h]

        for i in range(len(c_q_l)):
            if c_q_l[i] == c_d[i]:
                continue
            elif c_q_l[i] == (c_d[i] + 1) % 3:
                b = 2  # c_q_l > c_d
                break
            elif c_d[i] == (c_q_l[i] + 1) % 3:
                b = 0  # c_q_l < c_d
                break

        # 根据论文算法处理高值比较
        if b == 0:
            b = 1  # 重置为默认相等
            # 完整遍历所有bit重新比较
            for j in range(len(c_q_h)):
                if c_q_h[j] == c_d[j]:
                    continue
                elif c_q_h[j] == (c_d[j] + 1) % 3:
                    b = 2  # 高值 > 数据
                    break
                elif c_d[j] == (c_q_h[j] + 1) % 3:
                    b = 0  # 高值 < 数据
                    break
            # 论文算法中的特殊处理：当高值比较结果为2时设为1
            if b == 2:
                b = 1

        # 映射到最终三态结果
        if b == 0:
            return -1  # m_q < m_d
        elif b == 1:
            return 0   # m_q == m_d
        else:
            return 1   # m_q > m_d
    
    def data_compare(self, c1: str, c2: str) -> int:
        """比较两个密文，返回-1, 0, 1分别表示小于、等于、大于"""
        result = self._compare(c1, c2)
        # 将结果转换为标准比较值 (-1, 0, 1)
        return result - 1
    
    def sort_encrypted(self, ciphertexts: list[str]) -> list[str]:
        """使用实例的compare方法对密文排序"""
        return sorted(ciphertexts, key=cmp_to_key(self._compare))
    



# 示例用法
if __name__ == "__main__":

    ore = FreORE(d=2, alpha=1000, beta=10, gamma=5, pfk=b"secret_key", nx=8, ny=8)

    # 加密数字42
    p1 = 40
    c1 = ore.data_encrypt(p1)
    
    # 打印结果
    print(f"明文: {p1}")
    print(f"密文: {c1}")    
    print(f"密文长度: {len(c1)} 位")


    p2 = 50
    c2 = ore.data_encrypt(p2)
    print(f"明文: {p2}")
    print(f"密文: {c2}")

    print(f"compare result: {ore._compare(c1, c2)}")  # 比较两个密文

    plains = [i for i in range(100)]  # 明文数据范围0-99
    ciphers = [ore.data_encrypt(m) for m in plains]  # 加密所有明文
    print(ciphers[0])
    sorted_ciphertexts = ore.sort_encrypted(ciphers)  # 对密文排序

    sorted_indices = [ciphers.index(ct) for ct in sorted_ciphertexts]  # 获取排序后的明文索引
    sorted_plaintexts = [plains[i] for i in sorted_indices]  # 获取排序后的明文
    print("排序后的明文:", sorted_plaintexts)  # 打印排序后的明文

   
    # test_cases = [
    #     # (C_q_l, C_q_h), C_d, 预期结果
    #     (("1021", "1021"), "1021", 1),  # 完全相等
    #     (("0121", "1021"), "1021", 1),  # 低值 < 数据，高值 = 数据 → 相等
    #     (("0121", "1121"), "1021", 1),  # 低值 < 数据，高值 > 数据 → 数据在区间内 → 相等
    #     (("0121", "1121"), "0121", 1),  # 数据等于低值 → 相等
    #     (("0121", "1121"), "1121", 1),  # 数据等于高值 → 相等
    #     (("0121", "1121"), "2221", 2),  # 数据 > 高值 → m_q > m_d
    #     (("0121", "1121"), "0021", 0),  # 数据 < 低值 → m_q < m_d
    #     (("2102", "2102"), "2012", 2),  # 低值 > 数据 → m_q > m_d
    #     (("2012", "2012"), "2102", 0),  # 低值 < 数据 → m_q < m_d
    # ]

    # # 运行测试
    # for (C_q_l, C_q_h), C_d, expected in test_cases:
    #     result = trap_compare((C_q_l, C_q_h), C_d)
    #     print(f"陷阱门低值: {C_q_l}, 陷阱门高值: {C_q_h}, 数据密文: {C_d}, 预期: {expected}, 实际: {result} → {'通过' if result == expected else '失败'}")
