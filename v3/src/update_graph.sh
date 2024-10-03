#!/bin/bash
# conda activate quizzer_v3_env


# Check if a corpora directory is provided
if [ $# -eq 0 ]; then
    echo "Please provide the path to the corpora directory."
    echo "Usage: ./update_graph /path/to/corpora/directory"
    exit 1
fi

ROOT_DIR="/Users/chim/Working/Quizzer/v3/src"

CORPORA_DIR="$1"

# Check if the provided directory exists
if [ ! -d "$CORPORA_DIR" ]; then
    echo "The specified directory does not exist: $CORPORA_DIR"
    exit 1
fi

echo "Updating graph for corpora in: $CORPORA_DIR"

# Run get_metadata.py
echo "Step 1: Running get_metadata.py"
python3 "$ROOT_DIR/get_metadata.py" "$CORPORA_DIR"

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error occurred while running get_metadata.py"
    exit 1
fi

# Run get_chunks.py
echo "Step 2: Running get_chunks.py"
python3 "$ROOT_DIR/get_chunks.py" --corpora-dir "$CORPORA_DIR"

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error occurred while running get_chunks.py"
    exit 1
fi

# Run make_graph.py
echo "Step 3: Running make_graph.py"
python3 "$ROOT_DIR/make_graph.py" --corpora-dir "$CORPORA_DIR" --output "$CORPORA_DIR/corpora_graph.gpickle"
# python3 "$ROOT_DIR/make_graph.py" --corpora-dir "$CORPORA_DIR" --output "corpora_graph.gpickle"

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error occurred while running make_graph.py"
    exit 1
fi

echo "Graph update process completed successfully."