#!/bin/bash

# Print environment information
echo "Running pipeline in $(python3 -c "from config import ENVIRONMENT; print(ENVIRONMENT)") environment..."

# Function to run a Python script with error handling
run_script() {
    local script=$1
    echo "🚀 Running $script..."
    if python3 "$script"; then
        echo "✅ $script completed successfully"
    else
        echo "❌ $script failed"
        exit 1
    fi
}

# Run each script in sequence
run_script "src/retrieval_results_builder.py"
run_script "src/evaluation_summary.py"
run_script "src/translate_error_analysis.py"

echo "✅ All scripts completed successfully"