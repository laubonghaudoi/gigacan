#!/bin/bash
echo "Applying 4000 Gbps (2000 MHz) Memory Overclock..."
nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=4000"

echo "Overclocked!"
