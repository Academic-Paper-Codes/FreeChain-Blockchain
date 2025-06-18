# FreORE-FISCO-BCOS

A FISCO-BCOS smart contract package implementing the FreORE (Frequency-hiding Order-Revealing Encryption) scheme for secure and privacy-preserving data operations on blockchain.

## Introduction

FreORE-FISCO-BCOS is a smart contract package that implements the FreORE encryption scheme for the FISCO-BCOS blockchain platform. This package enables secure comparison operations on encrypted data while hiding the frequency information of the underlying plaintext values, making it ideal for privacy-preserving applications such as secure range queries, encrypted data sorting, and confidential data processing on blockchain.

## Package Overview

This package provides a complete implementation of the FreORE encryption scheme optimized for FISCO-BCOS blockchain. It includes smart contracts, Python interfaces, and utility functions for deploying and interacting with the encryption scheme on the blockchain.

### Key Features

- **Order-Revealing Encryption**: Enables comparison operations on encrypted data without revealing the actual values
- **Frequency-Hiding**: Protects against frequency analysis attacks by introducing controlled randomness
- **Efficient Blockchain Implementation**: Optimized for gas efficiency and blockchain storage
- **Trapdoor Mechanism**: Supports secure range queries on encrypted data
- **Smart Contract Integration**: Ready-to-deploy contracts for FISCO-BCOS

## Package Structure

```
FreORE-FISCO-BCOS/
├── contracts/                 # Solidity smart contracts
│   ├── FreOREStorage.sol      # Contract for storing encrypted data
│   ├── FreORECompare.sol      # Contract for comparison operations
│   └── FreOREQuery.sol        # Contract for query operations
├── python/                    # Python implementation and interfaces
│   ├── freore.py              # FreORE implementation
│   └── blockchain_interface.py # Interface to FISCO-BCOS
├── scripts/                   # Deployment and utility scripts
│   ├── deploy.py              # Deployment script
│   └── test.py                # Test script
└── examples/                  # Example applications
    └── secure_database.py     # Example of secure database application
```

### Core Components

#### FreORE Python Implementation (Schemes/FreORE.py)

The core encryption class that implements the FreORE scheme.

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

#### Smart Contract Functions

| Contract | Function | Description |
|----------|----------|-------------|
| FreOREStorage | `storeData(bytes ciphertext)` | Stores encrypted data on the blockchain |
| FreOREStorage | `getData(uint256 id)` | Retrieves encrypted data by ID |
| FreORECompare | `compare(bytes c1, bytes c2)` | Compares two ciphertexts on-chain |
| FreOREQuery | `rangeQuery(bytes low, bytes high)` | Performs a range query using trapdoor encryption |

## Deployment Guide

### Prerequisites

- FISCO-BCOS blockchain platform (v2.0 or higher)
- Python 3.13.2
- Conda environment manager
- FISCO-BCOS Python SDK
- FISCO-BCOS Console

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
   git clone https://github.com/yourusername/FreORE-FISCO-BCOS.git
   cd FreORE-FISCO-BCOS
   ```

2. Configure your FISCO-BCOS connection:
   ```bash
   cp ./config-example.json ./config.json
   # Edit config.json with your FISCO-BCOS node information
   ```

3. Deploy the smart contracts:
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
   from python.freore import FreORE
   ```

2. Initialize the blockchain interface:
   ```python
   from python.blockchain_interface import BlockchainInterface
   
   # Connect to your FISCO-BCOS node
   bc_interface = BlockchainInterface("config.json")
   ```

3. Deploy and interact with the contracts:
   ```python
   # Deploy contracts
   contract_addresses = bc_interface.deploy_contracts()
   
   # Use the FreORE scheme
   ore = FreORE(d=2, alpha=1000, beta=10, gamma=5, pfk=b"secret_key", nx=8, ny=8)
   
   # Store encrypted data on the blockchain
   plaintext = 42
   ciphertext = ore.data_encrypt(plaintext)
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
- WeBASE 1.5.0+ (for web management interface)

## FISCO-BCOS Resources

- [FISCO-BCOS GitHub Repository](https://github.com/FISCO-BCOS/FISCO-BCOS)
- [Official Documentation](https://fisco-bcos-documentation.readthedocs.io/)
- [Developer Documentation](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/developer/index.html)
- [Python SDK](https://github.com/FISCO-BCOS/python-sdk)
- [Console Manual](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/manual/console.html)
- [Smart Contract Development](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/developer/smart_contract.html)
- [Solidity Tutorial](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/developer/solidity.html)

## Usage Examples

### Python Client Example

```python
from python.freore import FreORE
from python.blockchain_interface import BlockchainInterface

# Initialize the blockchain interface
bc = BlockchainInterface("config.json")

# Initialize the FreORE scheme
ore = FreORE(d=2, alpha=1000, beta=10, gamma=5, pfk=b"secret_key", nx=8, ny=8)

# Encrypt and store data on the blockchain
plaintext = 42
ciphertext = ore.data_encrypt(plaintext)
tx_receipt = bc.store_data(ciphertext)
print(f"Data stored on blockchain. Transaction hash: {tx_receipt['transactionHash'].hex()}")

# Create a trapdoor for range query
trapdoor = ore.trap_encrypt(plaintext)

# Perform a range query on the blockchain
low_value = 30
high_value = 50
low_trapdoor, high_trapdoor = ore.trap_encrypt(low_value), ore.trap_encrypt(high_value)
query_results = bc.range_query(low_trapdoor, high_trapdoor)
print(f"Range query results: {query_results}")

# Compare two encrypted values on the blockchain
p1, p2 = 40, 50
c1, c2 = ore.data_encrypt(p1), ore.data_encrypt(p2)
comparison_result = bc.compare(c1, c2)  # Returns -1 (c1 < c2)
print(f"Comparison result: {comparison_result}")
```

### Console Example

```bash
# Deploy the contracts
./console.sh deploy FreOREStorage.sol

# Get the contract address
./console.sh getDeployLog

# Call the store function
./console.sh call FreOREStorage 0x1234...5678 storeData "0x123456789abcdef"

# Call the compare function
./console.sh call FreORECompare 0x1234...5678 compare "0x123456789abcdef" "0x987654321fedcba"
```

### Web3j Example

```java
// Load the contract
FreOREStorage storage = FreOREStorage.load(
    "0x1234...5678",
    web3j,
    credentials,
    new StaticGasProvider(gasPrice, gasLimit)
);

// Store data
TransactionReceipt receipt = storage.storeData("0x123456789abcdef").send();

// Get data
Tuple2<Boolean, String> result = storage.getData(BigInteger.valueOf(1)).send();
String ciphertext = result.getValue2();
```

## Performance Considerations

When deploying the FreORE scheme on FISCO-BCOS, consider the following performance aspects:

- **Gas Consumption**: The comparison operations are computationally intensive and may consume significant gas
- **Storage Optimization**: Encrypted data requires more storage than plaintext
- **Batch Processing**: Use batch operations for better throughput
- **Node Configuration**: Ensure your FISCO-BCOS nodes have sufficient resources

## Security Considerations

- **Key Management**: Securely manage encryption keys off-chain
- **Access Control**: Implement proper access control for the smart contracts
- **Audit**: Regular security audits are recommended
- **Privacy Leakage**: While FreORE hides data values and frequency, it reveals order relationships

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Contact and Support

For questions, issues, or contributions, please contact:
- GitHub Issues: [Create an issue](https://github.com/yourusername/FreORE-FISCO-BCOS/issues)
- Email: your.email@example.com
