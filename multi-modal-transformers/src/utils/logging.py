import os
import logging
import json
from typing import Dict, Any, Optional
import time
import datetime


def setup_logging(output_dir: str, log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log files
        log_level: Logging level
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file path
    log_file = os.path.join(output_dir, "training.log")
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log basic information
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Output directory: {output_dir}")


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
    output_dir: Optional[str] = None
) -> None:
    """
    Log metrics to console and optionally to a file.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current step
        prefix: Prefix for metric names
        output_dir: Directory to save metrics file
    """
    logger = logging.getLogger(__name__)
    
    # Format metrics for logging
    metrics_str = " - ".join([f"{prefix + '_' if prefix else ''}{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Step {step}: {metrics_str}")
    
    # Save metrics to file if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, "metrics.jsonl")
        
        # Add step and timestamp to metrics
        metrics_with_meta = {
            "step": step,
            "timestamp": datetime.datetime.now().isoformat(),
            "prefix": prefix
        }
        metrics_with_meta.update(metrics)
        
        # Append to metrics file
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics_with_meta) + "\n")


class MetricsLogger:
    """
    Class for tracking and logging metrics during training.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize metrics logger.
        
        Args:
            output_dir: Directory to save metrics files
        """
        self.output_dir = output_dir
        self.metrics = {}
        self.start_time = time.time()
    
    def update(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """
        Update metrics.
        
        Args:
            metrics: Dictionary of metrics to update
            prefix: Prefix for metric names
        """
        for k, v in metrics.items():
            key = f"{prefix}_{k}" if prefix else k
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(v)
    
    def log(self, step: int, window_size: int = 100) -> Dict[str, float]:
        """
        Log current metrics.
        
        Args:
            step: Current step
            window_size: Number of recent values to average
            
        Returns:
            Dictionary of averaged metrics
        """
        # Compute average of recent values for each metric
        avg_metrics = {}
        for k, v in self.metrics.items():
            if len(v) > 0:
                avg_metrics[k] = sum(v[-window_size:]) / len(v[-window_size:])
        
        # Add elapsed time
        elapsed = time.time() - self.start_time
        avg_metrics["elapsed_time"] = elapsed
        avg_metrics["steps_per_second"] = step / elapsed if elapsed > 0 else 0
        
        # Log metrics
        log_metrics(avg_metrics, step, output_dir=self.output_dir)
        
        return avg_metrics
    
    def reset(self) -> None:
        """Reset metrics."""
        self.metrics = {}
        self.start_time = time.time()
    
    def save(self, filename: str) -> None:
        """
        Save metrics to a file.
        
        Args:
            filename: Name of the file to save metrics to
        """
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            metrics_file = os.path.join(self.output_dir, filename)
            
            with open(metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=2) 