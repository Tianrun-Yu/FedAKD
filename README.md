# FedAKD


1. **Set paths** in each script (`RAW_DATA_ROOT`, `RAW_DATA_PATH`, `TRAIN_PATH`, `RESULT_DIR`, etc.).
2. **Download raw data**:
   ```bash
   python3 raw.py
3.**Partition data into clients**: 
   ```bash
   python3 devide.py
4. **Train standalone**:
   ```bash
   python3 standalone.py
5. **Run FedACKD**:
   ```bash
   python3 fedackd.py
