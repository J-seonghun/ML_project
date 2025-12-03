#!/bin/bash
set -e  # Exit on error

echo "========================================="
echo "ML Scheduler Training Workflow"
echo "========================================="

TRACE_DIR="/home/HDD/jeong/accel_sim_traces/rodinia_2.0-ft/11.0/"

# Step 1: Collect training data with ML scheduler
echo ""
echo "Step 1: Collecting training data with ML scheduler..."
export ML_COLLECT_DATA=1

./util/job_launching/run_simulations.py \
  -B rodinia_2.0-ft \
  -C RTX3070-SASS \
  -T $TRACE_DIR \
  -N ml_data_collection

echo "Waiting for data collection to complete..."
./util/job_launching/monitor_func_test.py -v -N ml_data_collection

# Check if data files were created
echo ""
echo "Checking for collected data files..."
DATA_FILES=$(ls ml_training_data_*.csv 2>/dev/null | wc -l)
if [ $DATA_FILES -eq 0 ]; then
    echo "WARNING: No training data files found!"
    echo "Data collection may have failed. Check simulation logs."
    exit 1
fi

echo "Found $DATA_FILES training data files"
ls -lh ml_training_data_*.csv

# Step 2: Train weights from collected data
echo ""
echo "Step 2: Training weights from collected data..."
unset ML_COLLECT_DATA  # Disable for training

python3 train_from_data.py --visualize

# Check if learned weights were created
if [ ! -f ml_scheduler_weights_learned.txt ]; then
    echo "ERROR: Training failed - no learned weights file found"
    exit 1
fi

echo ""
echo "Learned weights:"
cat ml_scheduler_weights_learned.txt

# Step 3: Apply learned weights
echo ""
echo "Step 3: Applying learned weights..."
cp ml_scheduler_weights_learned.txt ml_scheduler_weights.txt

# Rebuild to ensure weights are picked up
source gpu-simulator/setup_environment.sh
make -j4 -C gpu-simulator

# Step 4: Run with optimized weights
echo ""
echo "Step 4: Running simulations with optimized weights..."
./util/job_launching/run_simulations.py \
  -B rodinia_2.0-ft \
  -C RTX3070-SASS \
  -T $TRACE_DIR \
  -N ml_optimized

echo "Waiting for optimized run to complete..."
./util/job_launching/monitor_func_test.py -v -N ml_optimized

# Step 5: Compare results
echo ""
echo "========================================="
echo "Step 5: Comparing Results"
echo "========================================="

echo ""
echo "Generating performance comparison..."

# Extract IPC values
echo "Benchmark,Baseline_IPC,Optimized_IPC,Improvement" > ml_performance_comparison.csv

for bench in backprop bfs hotspot heartwall lud nw nn pathfinder srad_v2 streamcluster; do
    BASELINE_IPC=$(./util/job_launching/get_stats.py -N ml_data_collection 2>/dev/null | grep "gpu_tot_ipc" | grep "$bench" | awk -F',' '{print $2}' | head -1)
    OPTIMIZED_IPC=$(./util/job_launching/get_stats.py -N ml_optimized 2>/dev/null | grep "gpu_tot_ipc" | grep "$bench" | awk -F',' '{print $2}' | head -1)
    
    if [ ! -z "$BASELINE_IPC" ] && [ ! -z "$OPTIMIZED_IPC" ]; then
        IMPROVEMENT=$(echo "scale=2; ($OPTIMIZED_IPC - $BASELINE_IPC) / $BASELINE_IPC * 100" | bc)
        echo "$bench,$BASELINE_IPC,$OPTIMIZED_IPC,$IMPROVEMENT%" >> ml_performance_comparison.csv
    fi
done

echo ""
cat ml_performance_comparison.csv

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - ml_training_data_*.csv (training data)"
echo "  - ml_scheduler_weights_learned.txt (learned weights)"
echo "  - ml_performance_comparison.csv (performance comparison)"
echo "  - weight_comparison.png (visualization)"
echo ""
echo "To use the optimized weights in future runs:"
echo "  1. Weights are already applied in ml_scheduler_weights.txt"
echo "  2. Just run simulations normally"
echo ""
