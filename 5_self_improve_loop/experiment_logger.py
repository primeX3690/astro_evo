# 5_self_improve_loop/experiment_logger.py
"""Logs every trial (config + metrics) to JSON; retrieves the best trial found so far."""
import json
import os
import time


class ExperimentLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                json.dump([], f)

    def log_trial(self, config, metrics, tag=""):
        trial = {"timestamp": time.time(), "tag": tag, "config": config, "metrics": metrics}
        trials = self._load()
        trials.append(trial)
        with open(self.log_path, "w") as f:
            json.dump(trials, f, indent=2)
        return trial

    def _load(self):
        with open(self.log_path, "r") as f:
            return json.load(f)

    def all_trials(self):
        return self._load()

    def get_best(self, metric_key, maximize=True):
        trials = self._load()
        if not trials:
            return None
        key_fn = lambda t: t["metrics"].get(metric_key, float("-inf") if maximize else float("inf"))
        return max(trials, key=key_fn) if maximize else min(trials, key=key_fn)

    def summary(self, metric_key):
        trials = self._load()
        values = [t["metrics"].get(metric_key) for t in trials if metric_key in t["metrics"]]
        if not values:
            return {"count": 0}
        return {"count": len(values), "best": max(values), "worst": min(values), "mean": sum(values) / len(values)}