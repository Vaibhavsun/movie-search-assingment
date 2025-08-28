#!/bin/bash

# Check if user passed an argument
if [ -z "$1" ]; then
  echo "⚠️ Usage: ./run_query.sh <query>"
  exit 1
fi

# Run the Python script with the argument
python3 movie_search.py "$1"
