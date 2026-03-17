#!/usr/bin/env python3
"""
Scheduler base interface for cloud-agnostic scheduling.
Supports multiple deployment environments: systemd, Docker, Kubernetes, Serverless.
"""

from abc import ABC, abstractmethod
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class SchedulerBase(ABC):
    """Abstract base class for all schedulers."""
    
    @abstractmethod
    def schedule(self, task: Callable, schedule_spec: str) -> None:
        """
        Schedule a task to run periodically.
        
        Args:
            task: Callable function to execute
            schedule_spec: Schedule specification (format depends on scheduler)
        """
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the scheduler."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the scheduler."""
        pass


class SystemdScheduler(SchedulerBase):
    """
    Systemd-based scheduler for traditional Linux servers.
    Use with: systemctl enable/start track2college.timer
    """
    
    def schedule(self, task: Callable, schedule_spec: str) -> None:
        logger.info(f"Systemd scheduler: Use systemctl to manage scheduling")
        logger.info(f"Expected schedule_spec format: 'monthly', 'daily', etc.")
        logger.info(f"Run: sudo systemctl enable track2college.timer")
    
    def start(self) -> None:
        logger.info("Systemd scheduler: Use 'sudo systemctl start track2college.timer'")
    
    def stop(self) -> None:
        logger.info("Systemd scheduler: Use 'sudo systemctl stop track2college.timer'")


class APSchedulerScheduler(SchedulerBase):
    """
    APScheduler-based scheduler for Docker/Kubernetes/self-managed servers.
    Works in application context without systemd dependency.
    """
    
    def __init__(self):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            self.scheduler = BackgroundScheduler()
            self.is_running = False
        except ImportError:
            raise ImportError(
                "APScheduler not installed. Install with: pip install apscheduler"
            )
    
    def schedule(self, task: Callable, schedule_spec: str) -> None:
        """
        Schedule a task using APScheduler.
        
        Args:
            task: Callable function to execute
            schedule_spec: Cron-like spec (e.g., "0 0 1 * *" for monthly)
                          or "monthly", "weekly", "daily", etc.
        """
        # Convert simple specs to cron
        cron_specs = {
            "monthly": "0 0 1 * *",      # 1st day of month, midnight
            "weekly": "0 0 * * 0",        # Every Sunday, midnight
            "daily": "0 0 * * *",         # Every day, midnight
            "hourly": "0 * * * *",        # Every hour
        }
        
        cron_spec = cron_specs.get(schedule_spec.lower(), schedule_spec)
        
        try:
            self.scheduler.add_job(
                task,
                "cron",
                args=(),
                id="track2college_pipeline",
                replace_existing=True,
            )
            
            # Parse cron to readable format
            parts = cron_spec.split()
            if len(parts) == 5:
                minute, hour, day, month, dow = parts
                logger.info(f"Scheduled: minute={minute}, hour={hour}, day={day}, month={month}, dow={dow}")
            else:
                logger.info(f"Scheduled with spec: {schedule_spec}")
                
        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
            raise
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            logger.info("APScheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("APScheduler stopped")


class KubernetesScheduler(SchedulerBase):
    """
    Kubernetes CronJob scheduler.
    Note: Actual scheduling is handled by Kubernetes, not this class.
    This is a placeholder for documentation purposes.
    """
    
    def schedule(self, task: Callable, schedule_spec: str) -> None:
        logger.info(
            "Kubernetes scheduler: Define CronJob in YAML manifest "
            "(see kubernetes/cronjob.yaml)"
        )
    
    def start(self) -> None:
        logger.info("Kubernetes scheduler: Use 'kubectl apply -f kubernetes/cronjob.yaml'")
    
    def stop(self) -> None:
        logger.info("Kubernetes scheduler: Use 'kubectl delete cronjob track2college-pipeline'")


class ServerlessScheduler(SchedulerBase):
    """
    Serverless scheduler (AWS Lambda, GCP Cloud Functions, etc.)
    Note: Actual scheduling is handled by cloud provider, not this class.
    """
    
    def schedule(self, task: Callable, schedule_spec: str) -> None:
        logger.info(
            "Serverless scheduler: Configure in cloud provider "
            "(CloudWatch Events, Cloud Scheduler, etc.)"
        )
    
    def start(self) -> None:
        logger.info("Serverless scheduler: Configured in cloud provider console")
    
    def stop(self) -> None:
        logger.info("Serverless scheduler: Disable in cloud provider console")


def get_scheduler(scheduler_type: str = "systemd") -> SchedulerBase:
    """
    Factory function to get the appropriate scheduler.
    
    Args:
        scheduler_type: One of 'systemd', 'apscheduler', 'kubernetes', 'serverless'
    
    Returns:
        SchedulerBase instance
    """
    schedulers = {
        "systemd": SystemdScheduler,
        "apscheduler": APSchedulerScheduler,
        "kubernetes": KubernetesScheduler,
        "serverless": ServerlessScheduler,
    }
    
    scheduler_class = schedulers.get(scheduler_type.lower())
    if not scheduler_class:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Choose from: {', '.join(schedulers.keys())}"
        )
    
    logger.info(f"Using scheduler: {scheduler_type}")
    return scheduler_class()
