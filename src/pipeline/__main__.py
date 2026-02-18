"""CLI entry point for the pipeline.

Run with:
    python -m pipeline
"""

from pipeline.runner import run_pipeline
from pipeline.config import PipelineConfig

if __name__ == "__main__":
    # In future, we can add argparse here to override config
    config = PipelineConfig()
    run_pipeline(config)
