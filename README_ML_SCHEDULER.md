# ML Scheduler - Comprehensive Feature Extraction

## Overview
ML-based warp scheduler that extracts **23 comprehensive features** from all available warp state information.

## Feature Set (23 Features)

### Basic Identification (4)
1. **age**: Dynamic warp ID (creation order)
2. **warp_id**: Static warp slot ID
3. **cta_id**: CTA (thread block) ID
4. **stream_id_hash**: CUDA stream identifier (hashed)

### Pipeline State (4)
5. **inst_in_pipeline**: Total instructions in warp's pipeline
6. **inst_in_buffer**: Instructions in instruction buffer
7. **issued_inst_in_pipeline**: Already-issued instructions
8. **stores_outstanding**: Pending store operations (binary)

### Thread State (3)
9. **completed_threads**: Number of threads that have exited
10. **active_thread_count**: Currently active threads
11. **completion_ratio**: Fraction of completed threads (0-1)

### Memory/Sync State (6)
12. **inst_miss**: I-cache miss pending (binary)
13. **membar_waiting**: Waiting at memory barrier (binary)
14. **n_atomic**: Number of atomic operations in flight
15. **stores_done**: All stores completed (binary)
16. **done_exit**: Warp has exited (binary)

### Timing (2)
17. **last_fetch_age**: Cycles since last instruction fetch
18. **waiting_status**: Warp is waiting on dependencies (binary)

### Dependencies (1)
19. **scoreboard_stall**: Scoreboard has pending writes (binary)

### Derived Features (4)
20. **pipeline_stall_ratio**: Stalled vs total pipeline instructions
21. **age_squared**: age² for nonlinear patterns
22. **age_pipeline_interaction**: age × inst_in_pipeline
23. **ibuffer_empty**: Instruction buffer is empty (binary)

## Usage

### Enable ML Scheduler
In `gpgpusim.config`:
```
-gpgpu_scheduler ml
```

### Weight File
Weights are loaded from `ml_scheduler_weights.txt` in the working directory.

Format:
```
feature_name weight_value
# Comments start with #
```

### Data Collection for Training
```bash
export ML_COLLECT_DATA=1
./bin/release/accel-sim.out ...
```

This generates `ml_training_data_core{X}_sched{Y}.csv` with all 23 features + IPC.

### CSV Output Format
```csv
cycle,core_id,sched_id,warp_id,age,warp_id_feat,cta_id,stream_id_hash,...,recent_ipc
```

23 features + metadata + target (IPC)

## Training Scripts

### Advanced Training (Python)
Use `/home/jeong/ML_project/train_advanced_ml_scheduler.py`:
- MLP with BatchNorm, Dropout, He initialization
- XGBoost for feature importance
- StandardScaler normalization
- Early stopping
- Handles all 23 features automatically

### Installation
```bash
cd /home/jeong/ML_project
pip install -r requirements_advanced.txt
```

### Train
```bash
python train_advanced_ml_scheduler.py \
  --data-pattern "ml_training_data_*.csv" \
  --model mlp \
  --epochs 100 \
  --output-dir trained_models/
```

Outputs:
- `ml_scheduler_weights_mlp.txt` - MLP weights
- `ml_scheduler_weights_xgb.txt` - XGBoost weights
- `feature_importance.png` - Which features matter most

### Deploy Weights
```bash
cp trained_models/ml_scheduler_weights_mlp.txt ml_scheduler_weights.txt
cmake --build ./gpu-simulator/build -j8
cmake --install ./gpu-simulator/build
```

## Implementation Details

### Dynamic Weight Loading
Weights are stored in `std::map<string, double>` for flexibility:
- Easy to add/remove features
- Missing weights use defaults
- Supports any subset of features

### Feature Extraction
All features extracted in `extract_all_features()`:
- Direct warp state queries
- Derived calculations (ratios, interactions)
- Cached in `FeatureVector` struct

### Scoring
Linear model: `score = Σ(weight_i × feature_i)`

Higher score = higher priority for scheduling

## Files Modified
- `ml_scheduler.h` - 23 feature extractors
- `ml_scheduler.cc` - Implementation
- `shader.h` - Added CONCRETE_SCHEDULER_ML enum
- `shader.cc` - Factory + parsing
- `CMakeLists.txt` - Added ml_scheduler.cc

## Performance Tips

1. **Start with defaults**: Built-in weights work reasonably
2. **Collect diverse data**: Mix of kernels (compute, memory, control)
3. **Feature importance**: Use XGBoost to identify key features
4. **Iterate**: Train, test, analyze, repeat

## Debugging

Check if ML scheduler is active:
```bash
./bin/release/accel-sim.out ... 2>&1 | grep "ML Scheduler"
```

Should see:
```
ML Scheduler: Loaded X weights from 'ml_scheduler_weights.txt'
```

With `ML_COLLECT_DATA=1`:
```
ML Scheduler: Data collection ENABLED - Writing to ml_training_data_core0_sched0.csv (23 features)
```

## Next Steps

1. **Collect baseline data** with GTO scheduler
2. **Collect ML data** with default ML weights
3. **Train models** on collected data
4. **Deploy learned weights**
5. **Compare performance**: GTO vs ML vs LRR
6. **Iterate**: Add features, tune hyperparameters, collect more data

## Contact
For issues or questions about the ML scheduler implementation, check:
- `README_ADVANCED_TRAINING.md` - Training script details
- Implementation plan artifact
- Walkthrough artifact
