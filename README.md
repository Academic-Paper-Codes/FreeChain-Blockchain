# FreeChain: A Secure and Efficient Blockchain Database with Frequency-Resistant Order-Revealing Encryption

FreeChain is the first blockchain database that simultaneously ensures strong data security, query verifiability, and practical efficiency. It natively supports various order-oriented queries (e.g., range queries) over order-revealing ciphertexts, resists frequency inference attacks, and enables users to verify the soundness and completeness of query results.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Experimental Setup](#experimental-setup)
- [Installation](#installation)
- [Code Modules](#code-modules)
- [Libraries Used](#libraries-used)
- [Download and Running](#download-and-running)

## Overview

Blockchain databases have emerged as a promising decentralized storage paradigm. However, they face critical challenges in revealing order while ensuring data security, query verifiability, and practical efficiency. Existing secure order-revealing blockchain databases suffer from low throughput and high communication overhead due to interactive protocols. Although order-revealing encryption (ORE) offers non-interactive encryption and improved performance, it remains vulnerable to frequency inference attacks and lacks query verifiability.

FreeChain presents:
- **FreORE**: A frequency-resistant ORE scheme that generates order-revealing ciphertexts and resists frequency inference attacks
- **CVTree**: An authenticated data structure inspired by Merkle and prefix trees, constructed from FreORE's bitwise ciphertexts
- **BVTree**: A block-aligned authenticated data structure with a trade-off to CVTree between index storage and proof size

Experiments on the FISCO BCOS blockchain show that FreeChain outperforms the state-of-the-art by up to:
- 80× in encryption speed
- 27× in query speed
- 21× in throughput
- 4× in communication reduction

## Dependencies

### Core Components
- **FreORE**: Frequency-resistant Order-Revealing Encryption (version 1.0.0)
- **CVTree**: Cipher Verification Tree (version 1.0.0)
- **BVTree**: Block Verification Tree (version 1.0.0)

### Python Dependencies
```
python==3.12.9
hashlib
hmac
random
functools
dataclasses
typing
time
math
```

### Blockchain Platform
```
FISCO BCOS (version 2.9.0 or later)
Solidity==0.4.25
```

## Experimental Setup

Our experiments were performed on the following configuration:

- **Hardware**:
  - 3 virtual machines with Intel Core i7-12700F CPU @ 2.10GHz
  - 32GB RAM on host machines
  - Each VM using 2 host cores and 4GB RAM

- **Software**:
  - Operating System: Ubuntu 22.04
  - Python 3.12.9
  - Solidity 0.4.25
  - FISCO BCOS blockchain platform

## Installation

### Prerequisites
- Ubuntu 22.04 (or compatible Linux distribution)
- Python 3.12.9
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/freechain.git
cd freechain
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up FISCO BCOS
FISCO BCOS is an open-source enterprise-level financial blockchain platform. Follow these steps to set it up:

1. Download the FISCO BCOS build tool:
```bash
curl -LO https://github.com/FISCO-BCOS/FISCO-BCOS/releases/download/v2.9.0/build_chain.sh
chmod +x build_chain.sh
```

2. Build a 3-node blockchain network:
```bash
./build_chain.sh -l "127.0.0.1:4" -p 30300,20200,8545
```

3. Start the blockchain network:
```bash
cd nodes/127.0.0.1
./start_all.sh
```

4. Check if the nodes are running:
```bash
ps -ef | grep fisco-bcos
```

For more detailed instructions on FISCO BCOS setup, please refer to the [official documentation](https://fisco-bcos-documentation.readthedocs.io/en/latest/docs/installation.html).

### Step 4: Deploy Smart Contracts
```bash
python deploy_contracts.py
```

## Code Modules

### FreORE.py
The core implementation of the Frequency-resistant Order-Revealing Encryption scheme.

#### Key Functions:
- `__init__(d, alpha, beta, gamma, pfk, nx, ny)`: Initialize FreORE with parameters
- `data_encrypt(m)`: Encrypt integer m to ciphertext
- `trap_encrypt(m)`: Generate trapdoor for query
- `data_compare(c1, c2)`: Compare two ciphertexts
- `trap_compare(c_q, c_d)`: Compare trapdoor with ciphertext
- `sort_encrypted(ciphertexts)`: Sort encrypted values

### CVTree.py
Implementation of the Cipher Verification Tree, an authenticated data structure for efficient query verification.

#### Key Functions:
- `__init__(freore_instance)`: Initialize CVTree with FreORE instance
- `insert(plaintext, file_address)`: Insert data into the tree
- `range_query(low_plaintext, high_plaintext)`: Perform range query
- `generate_proof(plaintext)`: Generate verification proof
- `verify_proof(proof, root_hash)`: Verify a proof against root hash
- `compute_hashes()`: Compute all node hashes and return storage cost

### BVTree.py
Implementation of the Block Verification Tree, optimized for blockchain storage.

#### Key Functions:
- `__init__(freore_instance, block_size)`: Initialize BVTree with parameters
- `insert(plaintext, file_address)`: Insert data into blocks
- `range_query(low_plaintext, high_plaintext)`: Perform range query
- `generate_proof(plaintext)`: Generate verification proof
- `verify_proof(proof)`: Verify a proof
- `compute_merkle_roots()`: Compute Merkle roots for all blocks

## Libraries Used

### Cryptographic Functions
- **hashlib**: For SHA-256 hashing operations
- **hmac**: For HMAC-based message authentication
- **random**: For generating random values in FreORE

### Data Structures
- **dataclasses**: For defining structured data classes
- **typing**: For type annotations
- **functools**: For functional programming utilities (cmp_to_key)

### Performance Measurement
- **time**: For measuring execution times
- **math**: For mathematical operations

## Download and Running

### Download
```bash
git clone https://github.com/your-username/freechain.git
cd freechain
```

### Configuration
Edit the `config.py` file to set your parameters:
```python
# FreORE parameters
FREORE_PARAMS = {
    'd': 2,
    'alpha': 1000,
    'beta': 10,
    'gamma': 5,
    'pfk': b"your_secret_key",
    'nx': 8,
    'ny': 8
}

# BVTree parameters
BVTREE_BLOCK_SIZE = 1000

# FISCO BCOS connection
FISCO_BCOS_CONFIG = {
    'channel_host': '127.0.0.1',
    'channel_port': 20200,
    'contract_address': 'your_deployed_contract_address'
}
```

### Running Examples

#### Basic Usage Example
```python
from FreORE import FreORE
from cvtree import CVTree
from bvtree import BVTree

# Initialize FreORE
ore = FreORE(d=2, alpha=1000, beta=10, gamma=5, pfk=b"secret_key", nx=8, ny=8)

# Encrypt values
c1 = ore.data_encrypt(42)
c2 = ore.data_encrypt(50)

# Compare encrypted values
result = ore.data_compare(c1, c2)  # Returns -1 (42 < 50)

# Initialize CVTree
cvtree = CVTree(ore)

# Insert data
cvtree.insert(42, "file1.txt")
cvtree.insert(50, "file2.txt")

# Perform range query
results = cvtree.range_query(40, 60)  # Returns ["file1.txt", "file2.txt"]

# Initialize BVTree
bvtree = BVTree(ore, block_size=1000)

# Insert data
bvtree.insert(42, "file1.txt")
bvtree.insert(50, "file2.txt")

# Perform range query
results = bvtree.range_query(40, 60)  # Returns ["file1.txt", "file2.txt"]
```

#### Running Benchmarks
```bash
python benchmarks.py
```

#### Deploying to FISCO BCOS
```bash
python deploy_to_blockchain.py
```

For more detailed examples and use cases, please refer to the `examples/` directory in the repository.

---

## Citation
If you use FreeChain in your research, please cite our paper:
```
@inproceedings{freechain2023,
  title={FreeChain: A Secure and Efficient Blockchain Database with Frequency-Resistant Order-Revealing Encryption},
  author={[Author Names]},
  booktitle={[Conference Name]},
  year={2023}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
