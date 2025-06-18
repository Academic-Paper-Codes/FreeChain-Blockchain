# FreeChain-FISCO-BCOS

A FISCO-BCOS implementation of a secure blockchain database system that utilizes the FreORE (Frequency-resistant Order-Revealing Encryption) scheme for privacy-preserving data operations.

## Introduction

FreeChain-FISCO-BCOS is a blockchain-based secure database system that implements the FreORE encryption scheme on the FISCO-BCOS blockchain platform. FreeChain enables secure comparison operations on encrypted data while resisting frequency analysis attacks on the underlying plaintext values, making it ideal for privacy-preserving applications such as secure range queries, encrypted data sorting, and confidential data processing on blockchain.

## Package Overview

This package provides a complete implementation of the FreeChain system optimized for FISCO-BCOS blockchain. FreeChain utilizes the FreORE encryption scheme as a core component to enable secure and privacy-preserving data operations. The package includes Python interfaces and utility functions for deploying and interacting with the system on the blockchain.

### Key Features

- **Order-Revealing Encryption**: Enables comparison operations on encrypted data without revealing the actual values
- **Frequency-Resistant**: Protects against frequency analysis attacks by introducing controlled randomness
- **Efficient Blockchain Implementation**: Optimized for blockchain storage and performance
- **Trapdoor Mechanism**: Supports secure range queries on encrypted data

## Package Structure

```
FreeChain-FISCO-BCOS/
├── Schemes/                   # Encryption scheme implementations
│   ├── FreORE.py              # FreORE implementation
│   ├── BlockOPE.py            # BlockOPE implementation
│   ├── EncodeORE.py           # EncodeORE implementation
│   └── HybridORE.py           # HybridORE implementation
├── python/                    # Python implementation and interfaces
│   └── blockchain_interface.py # Interface to FISCO-BCOS
├── scripts/                   # Deployment and utility scripts
│   ├── deploy.py              # Deployment script
│   └── test.py                # Test script
└── examples/                  # Example applications
    └── secure_database.py     # Example of secure database application
```

### Core Components

#### Encryption Schemes

FreeChain incorporates multiple encryption schemes, with FreORE being the primary component:

##### FreORE Python Implementation (Schemes/FreORE.py)

The core encryption class that implements the Frequency-resistant Order-Revealing Encryption scheme.

##### Constructor Parameters

- `d`: Base for the scientific notation representation
- `alpha`: Coefficient for the first bit perturbation
- `beta`: Coefficient for the rest bits perturbation
- `gamma`: Range for random noise
- `pfk`: Pseudo-random function key
- `nx`: Bit length for the mantissa part
- `ny`: Bit length for the exponent part

##### Main Functions

| Function | Description |
|----------|-------------|
| `data_encrypt(m)` | Encrypts an integer value into a ciphertext string |
| `_compare(c1, c2)` | Internal function to compare two ciphertexts |
| `trap_encrypt(m)` | Creates a trapdoor encryption for range queries |
| `trap_compare(c_q, c_d)` | Compares a trapdoor ciphertext with a data ciphertext |
| `data_compare(c1, c2)` | Compares two data ciphertexts, returns -1, 0, or 1 |
| `sort_encrypted(ciphertexts)` | Sorts a list of ciphertexts based on their order |

## Deployment Guide

### Prerequisites

- FISCO-BCOS blockchain platform (v2.0 or higher)
- Python 3.13.2
- Conda environment manager
- FISCO-BCOS Python SDK

### Environment Setup

1. Install FISCO-BCOS following the [official documentation](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/installation.html)

2. Create and activate the conda environment:
   ```bash
   conda env create -f env.yml
   conda activate experiment
   ```

3. Install the FISCO-BCOS Python SDK:
   ```bash
   pip install client-sdk-python
   ```

### Deployment Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/FreeChain-FISCO-BCOS.git
   cd FreeChain-FISCO-BCOS
   ```

2. Configure your FISCO-BCOS connection:
   ```bash
   cp ./config-example.json ./config.json
   # Edit config.json with your FISCO-BCOS node information
   ```

3. Run the deployment script:
   ```bash
   python scripts/deploy.py
   ```

4. Verify the deployment:
   ```bash
   python scripts/test.py
   ```

### Integration with Existing Applications

To integrate the FreORE scheme with your existing FISCO-BCOS applications:

1. Import the FreORE Python module:
   ```python
   from Schemes.FreORE import FreORE
   ```

2. Initialize the blockchain interface:
   ```python
   from python.blockchain_interface import BlockchainInterface
   
   # Connect to your FISCO-BCOS node
   bc_interface = BlockchainInterface("config.json")
   ```

3. Use the FreORE scheme:
   ```python
   # Initialize FreORE
   ore = FreORE(d=2, alpha=1000, beta=10, gamma=5, pfk=b"secret_key", nx=8, ny=8)
   
   # Encrypt data
   plaintext = 42
   ciphertext = ore.data_encrypt(plaintext)
   
   # Store on blockchain
   tx_receipt = bc_interface.store_data(ciphertext)
   ```

## Technical Requirements

### Dependencies

The package requires the following main dependencies (see `env.yml` for complete list):

- Python 3.13.2
- pycryptodome 3.21.0 (for cryptographic operations)
- numpy 2.2.5
- client-sdk-python (FISCO-BCOS Python SDK)
- matplotlib 3.10.0 (for visualization)
- pytest 8.3.4 (for testing)
- scikit-learn 1.6.1
- scipy 1.15.3

### FISCO-BCOS Version Compatibility

- FISCO-BCOS v2.0+
- FISCO-BCOS v3.0+ (recommended)

## FISCO-BCOS Resources

- [FISCO-BCOS GitHub Repository](https://github.com/FISCO-BCOS/FISCO-BCOS)
- [Official Documentation](https://fisco-bcos-documentation.readthedocs.io/)
- [Developer Documentation](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/developer/index.html)
- [Python SDK](https://github.com/FISCO-BCOS/python-sdk)
- [Console Manual](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/manual/console.html)
- [Solidity Tutorial](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/developer/solidity.html)

## Usage Examples

### Python Client Example

```python
from Schemes.FreORE import FreORE
from python.blockchain_interface import BlockchainInterface

# Initialize the blockchain interface
bc = BlockchainInterface("config.json")

# Initialize the FreORE scheme
ore = FreORE(d=2, alpha=1000, beta=10, gamma=5, pfk=b"secret_key", nx=8, ny=8)

# Encrypt data
plaintext = 42
ciphertext = ore.data_encrypt(plaintext)
print(f"Plaintext: {plaintext}")
print(f"Ciphertext: {ciphertext}")

# Store encrypted data on the blockchain
tx_receipt = bc.store_data(ciphertext)
print(f"Data stored on blockchain. Transaction hash: {tx_receipt['transactionHash'].hex()}")

# Create a trapdoor for range query
trapdoor = ore.trap_encrypt(plaintext)

# Perform a range query
low_value = 30
high_value = 50
low_trapdoor, high_trapdoor = ore.trap_encrypt(low_value), ore.trap_encrypt(high_value)
query_results = bc.range_query(low_trapdoor, high_trapdoor)
print(f"Range query results: {query_results}")

# Compare two encrypted values
p1, p2 = 40, 50
c1, c2 = ore.data_encrypt(p1), ore.data_encrypt(p2)
comparison_result = ore.data_compare(c1, c2)  # Returns -1 (c1 < c2)
print(f"Comparison result: {comparison_result}")

# Sort encrypted data
plaintexts = [i for i in range(10)]
ciphertexts = [ore.data_encrypt(p) for p in plaintexts]
sorted_ciphertexts = ore.sort_encrypted(ciphertexts)
```

## Performance Considerations

When deploying the FreORE scheme on FISCO-BCOS, consider the following performance aspects:

- **Computational Complexity**: The comparison operations are computationally intensive
- **Storage Requirements**: Encrypted data requires more storage than plaintext
- **Node Configuration**: Ensure your FISCO-BCOS nodes have sufficient resources

## Security Considerations

- **Key Management**: Securely manage encryption keys off-chain
- **Access Control**: Implement proper access control mechanisms
- **Audit**: Regular security audits are recommended
- **Privacy Leakage**: While FreORE resists frequency analysis, it still reveals order relationships

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Contact and Support

For questions, issues, or contributions, please contact:
- GitHub Issues: [Create an issue](https://github.com/yourusername/FreeChain-FISCO-BCOS/issues)
- Email: your.email@example.com
