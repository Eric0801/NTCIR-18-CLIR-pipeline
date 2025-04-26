#!/bin/bash
python3 src/retrieval_results_builder.py
python3 src/evaluation_summary.py
python3 src/translate_error_analysis.py