#!/bin/bash
#
# RT Analysis Script - Flexible Grid Resolution Processing
#
# Usage:
#   ./run_rt_analysis.sh 640x800                    # Single grid
#   ./run_rt_analysis.sh 640x800 960x1200           # Multiple grids
#   ./run_rt_analysis.sh 640x800 960x1200 1280x1600 # All remaining grids
#
# Available grids: 160x200, 320x400, 640x800, 960x1200, 1280x1600
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# Base paths
#BASE_DIR="/media/rod/ResearchII_III/svofRuns/Dalziel_1999"
BASE_DIR="/media/rod/ResearchII_III/svofRuns/Hardy_1992"
ANALYZER="/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI/integration/rt_analyzer_refactored.py"

# Analysis settings
ANALYSIS_TYPES="all"  # fractal, rt_classification, mixing, power_spectrum, velocity, multifractal
OUTPUT_DIR="rt_analysis_ai"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Usage function
usage() {
    echo ""
    echo "Usage: $0 <grid1> [grid2] [grid3] ..."
    echo ""
    echo "Examples:"
    echo "  $0 640x800                           # Process single grid"
    echo "  $0 640x800 960x1200                  # Process two grids"
    echo "  $0 640x800 960x1200 1280x1600        # Process three grids"
    echo ""
    echo "Available grids:"
    echo "  160x200, 320x400, 640x800, 960x1200, 1280x1600"
    echo ""
    exit 1
}

# Check if any arguments provided
if [ $# -eq 0 ]; then
    echo -e "${RED}ERROR: No grid resolutions specified${NC}"
    usage
fi

# Function to estimate analysis time
estimate_time() {
    local grid_size=$1
    local base_time=192  # seconds per file for 320x400

    case $grid_size in
        160x200)
            factor=0.25
            ;;
        320x400)
            factor=1.0
            ;;
        640x800)
            factor=4.0
            ;;
        960x1200)
            factor=9.0
            ;;
        1280x1600)
            factor=16.0
            ;;
        *)
            factor=1.0
            ;;
    esac

    echo $(echo "$base_time * $factor" | bc | cut -d. -f1)
}

# Function to run analysis for a grid
run_grid_analysis() {
    local grid_size=$1
    #local data_dir="${BASE_DIR}/${grid_size}/slimMaster"
    local data_dir="${BASE_DIR}/${grid_size}/slimMaster/vorig/vorig-0.1/IC-rand"

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Processing Grid: ${grid_size}${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}Data directory: ${data_dir}${NC}"
    echo -e "${BLUE}Output directory: ${data_dir}/${OUTPUT_DIR}${NC}"
    echo ""

    # Check if directory exists
    if [ ! -d "$data_dir" ]; then
        echo -e "${RED}ERROR: Directory not found: ${data_dir}${NC}"
        echo -e "${RED}Skipping ${grid_size}${NC}"
        return 1
    fi

    # Count VTK files
    vtk_count=$(find "$data_dir" -maxdepth 1 -name "*.vtk" ! -name "*Mesh*" | wc -l)

    if [ $vtk_count -eq 0 ]; then
        echo -e "${RED}ERROR: No VTK files found in ${data_dir}${NC}"
        echo -e "${RED}Skipping ${grid_size}${NC}"
        return 1
    fi

    echo -e "${BLUE}Found ${vtk_count} VTK files to process${NC}"

    # Estimate time
    est_time_per_file=$(estimate_time "$grid_size")
    est_total_seconds=$((est_time_per_file * vtk_count))
    est_hours=$((est_total_seconds / 3600))
    est_minutes=$(((est_total_seconds % 3600) / 60))

    echo -e "${YELLOW}Estimated time: ~${est_hours}h ${est_minutes}m (${est_time_per_file}s per file)${NC}"
    echo ""

    # Start time
    start_time=$(date +%s)
    echo -e "${BLUE}Starting analysis at $(date)${NC}"
    echo ""

    # Run analysis
    python3 "$ANALYZER" \
        "$data_dir" \
        --temporal \
        --analysis "$ANALYSIS_TYPES" \
        --save-csv \
        --plot \
        --output-dir "$OUTPUT_DIR"

    # End time
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))
    seconds=$((elapsed % 60))

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Grid ${grid_size} completed!${NC}"
    echo -e "${GREEN}Actual time: ${hours}h ${minutes}m ${seconds}s${NC}"
    echo -e "${GREEN}Completed at: $(date)${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    return 0
}

# Main execution
echo ""
echo -e "${YELLOW}╔════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  RT Analysis - Dalziel 1999                    ║${NC}"
echo -e "${YELLOW}║  Rayleigh-Taylor Instability                   ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Display grids to be processed
echo -e "${BLUE}Grids to process: $@${NC}"

# Track overall start time
overall_start=$(date +%s)

# Track success/failure
successful_grids=()
failed_grids=()

# Process each grid argument
for grid in "$@"; do
    if run_grid_analysis "$grid"; then
        successful_grids+=("$grid")
    else
        failed_grids+=("$grid")
    fi
done

# Overall completion
overall_end=$(date +%s)
total_elapsed=$((overall_end - overall_start))
total_hours=$((total_elapsed / 3600))
total_minutes=$(((total_elapsed % 3600) / 60))
total_seconds=$((total_elapsed % 60))

echo ""
echo -e "${YELLOW}╔════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  ANALYSIS COMPLETE                             ║${NC}"
echo -e "${YELLOW}║  Total time: ${total_hours}h ${total_minutes}m ${total_seconds}s${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Summary
if [ ${#successful_grids[@]} -gt 0 ]; then
    echo -e "${GREEN}Successfully processed (${#successful_grids[@]}):"
    for grid in "${successful_grids[@]}"; do
        echo -e "  ✓ ${grid} → ${BASE_DIR}/${grid}/slimMaster/${OUTPUT_DIR}/${NC}"
    done
    echo ""
fi

if [ ${#failed_grids[@]} -gt 0 ]; then
    echo -e "${RED}Failed to process (${#failed_grids[@]}):"
    for grid in "${failed_grids[@]}"; do
        echo -e "  ✗ ${grid}${NC}"
    done
    echo ""
fi

echo -e "${BLUE}Results are in each grid's ${OUTPUT_DIR}/ directory${NC}"
echo -e "${BLUE}  - CSV: temporal_evolution.csv${NC}"
echo -e "${BLUE}  - Plots: *_evolution.png${NC}"
echo ""
