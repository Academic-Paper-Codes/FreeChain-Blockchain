# FreeChain-Blockchain

A blockchain-based secure database system implementing the FreORE (Frequency-hiding Order-Revealing Encryption) scheme on FISCO-BCOS blockchain platform.

## Introduction

FreeChain-Blockchain is a project that deploys the FreORE encryption scheme to the FISCO-BCOS blockchain platform. FreORE is a specialized order-revealing encryption scheme that allows for secure comparison operations on encrypted data while hiding the frequency information of the underlying plaintext values. This implementation enables secure and privacy-preserving range queries on blockchain-stored data.

## FreORE Encryption Scheme

The FreORE scheme is designed to provide secure order-revealing encryption with frequency-hiding properties. It allows for comparing encrypted values without revealing the actual data, making it suitable for privacy-preserving applications on blockchain.

### Key Features

- **Order-Revealing Encryption**: Enables comparison operations on encrypted data
- **Frequency-Hiding**: Protects against frequency analysis attacks
- **Efficient Comparison**: Optimized for blockchain environments
- **Trapdoor Mechanism**: Supports secure range queries

## Code Structure

The implementation consists of the following main components:

### FreORE Class

The core encryption class that implements the FreORE scheme.

#### Constructor Parameters

- `d`: Base for the scientific notation representation
- `alpha`: Coefficient for the first bit perturbation
- `beta`: Coefficient for the rest bits perturbation
- `gamma`: Range for random noise
- `pfk`: Pseudo-random function key
- `nx`: Bit length for the mantissa part
- `ny`: Bit length for the exponent part

#### Main Functions

| Function | Description |
|----------|-------------|
| `data_encrypt(m)` | Encrypts an integer value into a ciphertext string |
| `_compare(c1, c2)` | Internal function to compare two ciphertexts |
| `trap_encrypt(m)` | Creates a trapdoor encryption for range queries |
| `trap_compare(c_q, c_d)` | Compares a trapdoor ciphertext with a data ciphertext |
| `data_compare(c1, c2)` | Compares two data ciphertexts, returns -1, 0, or 1 |
| `sort_encrypted(ciphertexts)` | Sorts a list of ciphertexts based on their order |

### Helper Functions

| Function | Description |
|----------|-------------|
| `_split_data(m, d)` | Splits a number into mantissa and exponent parts |
| `_prf(index, prefix)` | Pseudo-random function implementation using HMAC-SHA256 |

## Deployment to FISCO-BCOS

### Prerequisites

- FISCO-BCOS blockchain platform (v2.0 or higher)
- Python 3.13.2
- Conda environment manager

### Installation Steps

1. Clone the FISCO-BCOS repository:
   ```bash
   git clone https://github.com/FISCO-BCOS/FISCO-BCOS.git
   ```

2. Set up the FISCO-BCOS environment following the [official documentation](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/installation.html)

3. Create and activate the conda environment:
   ```bash
   conda env create -f env.yml
   conda activate experiment
   ```

4. Deploy the FreORE scheme to FISCO-BCOS:
   ```bash
   # Example deployment command
   python deploy_freore.py
   ```

## Dependencies

The project requires the following main dependencies (see `env.yml` for complete list):

- Python 3.13.2
- pycryptodome 3.21.0 (for cryptographic operations)
- numpy 2.2.5
- matplotlib 3.10.0 (for visualization)
- pytest 8.3.4 (for testing)
- scikit-learn 1.6.1
- scipy 1.15.3

## FISCO-BCOS Resources

- [FISCO-BCOS GitHub Repository](https://github.com/FISCO-BCOS/FISCO-BCOS)
- [Official Documentation](https://fisco-bcos-documentation.readthedocs.io/)
- [Developer Documentation](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/developer/index.html)
- [Python SDK](https://github.com/FISCO-BCOS/python-sdk)
- [Console Manual](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/manual/console.html)

## Usage Example

```python
from Schemes.FreORE import FreORE

# Initialize the FreORE scheme
ore = FreORE(d=2, alpha=1000, beta=10, gamma=5, pfk=b"secret_key", nx=8, ny=8)

# Encrypt a value
plaintext = 42
ciphertext = ore.data_encrypt(plaintext)
print(f"Plaintext: {plaintext}")
print(f"Ciphertext: {ciphertext}")

# Create a trapdoor for range query
trapdoor = ore.trap_encrypt(plaintext)

# Compare two encrypted values
p1 = 40
p2 = 50
c1 = ore.data_encrypt(p1)
c2 = ore.data_encrypt(p2)
result = ore.data_compare(c1, c2)  # Returns -1 (c1 < c2)

# Sort encrypted data
plaintexts = [i for i in range(100)]
ciphertexts = [ore.data_encrypt(m) for m in plaintexts]
sorted_ciphertexts = ore.sort_encrypted(ciphertexts)
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
