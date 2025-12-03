#!/bin/bash
# Quick Start: ML Scheduler with Feature Logging

echo "========================================="
echo "ML Scheduler Feature Logging Quick Start"
echo "========================================="

# Step 1: Collect training data
echo ""
echo "Step 1: Collecting training data..."
export ML_COLLECT_DATA=1

./util/job_launching/run_simulations.py \
  -B rodinia_2.0-ft \
  -C RTX3070-SASS \
  -T /home/HDD/jeong/accel_sim_traces/rodinia_2.0-ft/11.0/ \
  -N ml_data_collection

# Wait for completion
./util/job_launching/monitor_func_test.py -v -N ml_data_collection

echo ""
echo "Step 2: Training weights from collected data..."
./train_from_data.py --visualize

echo ""
echo "Step 3: Apply learned weights..."
if [ -f ml_scheduler_weights_learned.txt ]; then
    cp ml_scheduler_weights_learned.txt ml_scheduler_weights.txt
    echo "Weights applied!"
else
    echo "Warning: No learned weights file found"
fi

echo ""
echo "Step 4: Rebuild simulator..."
source gpu-simulator/setup_environment.sh
make -j -C gpu-simulator

echo ""
echo "Step 5: Run with optimized weights..."
unset ML_COLLECT_DATA  # Disable data collection for validation
./util/job_launching/run_simulations.py \
  -B rodinia_2.0-ft \
  -C RTX3070-SASS \
  -T /home/HDD/jeong/accel_sim_traces/rodinia_2.0-ft/11.0/ \
  -N ml_optimized

echo ""
echo "========================================="
echo "Done! Compare results with:"
echo "  ./util/job_launching/get_stats.py -N ml_data_collection"
echo "  ./util/job_launching/get_stats.py -N ml_optimized"
echo "========================================="
