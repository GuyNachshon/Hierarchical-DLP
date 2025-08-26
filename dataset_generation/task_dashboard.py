#!/usr/bin/env python3
"""
Live Task Dashboard for Data Generation Progress Tracking.

Provides a clean, real-time view of all concurrent operations instead of
scrolling logs. Shows progress, active tasks, batch status, and thread activity.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich.align import Align


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a single task in the dashboard"""
    id: str
    name: str
    status: TaskStatus
    details: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    error_message: Optional[str] = None
    
    def duration(self) -> Optional[timedelta]:
        """Get task duration if completed or current duration if in progress"""
        if self.start_time is None:
            return None
        end_time = self.end_time if self.end_time else datetime.now()
        return end_time - self.start_time


@dataclass
class BatchInfo:
    """Batch status information"""
    batch_id: str
    provider: str
    model: str
    total_requests: int
    completed_requests: int = 0
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None


@dataclass  
class ThreadInfo:
    """Thread activity information"""
    name: str
    status: str
    current_task: str = ""
    last_update: datetime = field(default_factory=datetime.now)


class TaskManager:
    """Centralized task state management with thread safety"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.tasks: Dict[str, Task] = {}
        self.batches: Dict[str, BatchInfo] = {}
        self.threads: Dict[str, ThreadInfo] = {}
        self.logs: List[Tuple[datetime, str]] = []
        self.max_logs = 100
        self.global_progress = {
            'current': 0,
            'target': 0,
            'start_time': datetime.now()
        }
        self._shutdown = False
        
    def create_task(self, task_id: str, name: str, details: str = "") -> None:
        """Create a new task"""
        with self.lock:
            if task_id not in self.tasks:
                self.tasks[task_id] = Task(
                    id=task_id,
                    name=name,
                    status=TaskStatus.PENDING,
                    details=details
                )

    def update_task(self, task_id: str, status: TaskStatus, details: str = "",
                   progress: float = 0.0, error_message: str = None) -> None:
        """Update task status and details"""
        with self.lock:
            if task_id not in self.tasks:
                # Create task inline to avoid recursive lock
                self.tasks[task_id] = Task(
                    id=task_id,
                    name=task_id.replace("_", " ").title(),
                    status=TaskStatus.PENDING,
                    details=details
                )
                
            task = self.tasks[task_id]
            old_status = task.status
            task.status = status
            task.details = details
            task.progress = progress
            task.error_message = error_message
            
            # Track timing
            if old_status == TaskStatus.PENDING and status == TaskStatus.IN_PROGRESS:
                task.start_time = datetime.now()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.end_time = datetime.now()
                
    def complete_task(self, task_id: str, details: str = "") -> None:
        """Mark task as completed"""
        self.update_task(task_id, TaskStatus.COMPLETED, details)
        
    def fail_task(self, task_id: str, error_message: str, details: str = "") -> None:
        """Mark task as failed"""
        self.update_task(task_id, TaskStatus.FAILED, details, error_message=error_message)
        
    def update_batch(self, batch_id: str, provider: str, model: str, 
                    total_requests: int, completed_requests: int = 0,
                    status: str = "pending", estimated_completion: datetime = None) -> None:
        """Update batch information"""
        with self.lock:
            self.batches[batch_id] = BatchInfo(
                batch_id=batch_id,
                provider=provider,
                model=model,
                total_requests=total_requests,
                completed_requests=completed_requests,
                status=status,
                estimated_completion=estimated_completion
            )
            
    def update_thread(self, thread_name: str, status: str, current_task: str = "") -> None:
        """Update thread activity"""
        with self.lock:
            self.threads[thread_name] = ThreadInfo(
                name=thread_name,
                status=status,
                current_task=current_task,
                last_update=datetime.now()
            )
            
    def set_global_progress(self, current: int, target: int) -> None:
        """Update global progress counters"""
        with self.lock:
            self.global_progress['current'] = current
            self.global_progress['target'] = target
            
    def increment_progress(self, amount: int = 1) -> None:
        """Increment global progress counter"""
        with self.lock:
            self.global_progress['current'] += amount

    def add_log(self, message: str) -> None:
        """Add a log message to the dashboard"""
        with self.lock:
            self.logs.append((datetime.now(), message))
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)

    # Optional: integrate Python logging to live logs
    def attach_logging_handler(self, level: int = None):
        """Attach a logging handler that forwards records to the dashboard logs panel"""
        import logging

        class _DashboardLogHandler(logging.Handler):
            def emit(inner_self, record: logging.LogRecord) -> None:
                try:
                    msg = inner_self.format(record)
                except Exception:
                    msg = record.getMessage()
                # Avoid deadlocks by not acquiring the logger's lock inside add_log call context
                get_task_manager().add_log(msg)

        handler = _DashboardLogHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.addHandler(handler)
        if level is not None:
            root.setLevel(level)
        return handler
            
    def get_state(self) -> Dict[str, Any]:
        """Get current state snapshot (thread-safe)"""
        with self.lock:
            return {
                'tasks': dict(self.tasks),
                'batches': dict(self.batches),
                'threads': dict(self.threads),
                'progress': dict(self.global_progress),
                'logs': list(self.logs)
            }
            
    def shutdown(self):
        """Signal shutdown"""
        self._shutdown = True
        
    @property
    def is_shutdown(self) -> bool:
        return self._shutdown


class LiveDashboard:
    """Live updating terminal dashboard"""
    
    def __init__(self, task_manager: TaskManager, refresh_rate: float = 1.0):
        self.task_manager = task_manager
        self.refresh_rate = refresh_rate
        self.console = Console()
        self.layout = Layout()
        self._setup_layout()
        
    def _setup_layout(self):
        """Setup the dashboard layout structure"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        self.layout["left"].split_column(
            Layout(name="progress", size=8),
            Layout(name="tasks")
        )
        
        self.layout["right"].split_column(
            Layout(name="batches", ratio=1),
            Layout(name="threads", ratio=1),
            Layout(name="logs", ratio=2)
        )
        
    def _create_header(self, state: Dict[str, Any]) -> Panel:
        """Create header panel with overall status"""
        progress = state['progress']
        current = progress['current']
        target = progress['target']
        percentage = (current / target * 100) if target > 0 else 0
        
        # Calculate ETA
        elapsed = datetime.now() - progress['start_time']
        if current > 0:
            rate = current / elapsed.total_seconds()
            remaining = target - current
            eta_seconds = remaining / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = eta.strftime("%H:%M:%S")
        else:
            eta_str = "calculating..."
            
        header_text = f"🎯 Data Generation Progress: {current:,}/{target:,} ({percentage:.1f}%) | ETA: {eta_str}"
        return Panel(Align.center(Text(header_text, style="bold blue")),
                    title="Agentic Data Generator", border_style="blue")
        
    def _create_progress_panel(self, state: Dict[str, Any]) -> Panel:
        """Create progress visualization panel"""
        progress_info = state['progress']
        current = progress_info['current']
        target = progress_info['target']
        
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("({task.percentage:>3.0f}%)"),
            TimeRemainingColumn(),
        )
        
        task_id = progress.add_task("Examples Generated", completed=current, total=target)
        
        # Add batch progress
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        # Calculate rates
        elapsed = datetime.now() - progress_info['start_time']
        rate = current / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        
        table.add_row("Examples/hour", f"{rate * 3600:.1f}")
        table.add_row("Active batches", str(len([b for b in state['batches'].values() if b.status == 'in_progress'])))
        table.add_row("Completed batches", str(len([b for b in state['batches'].values() if b.status == 'completed'])))
        table.add_row("Total runtime", str(elapsed).split('.')[0])
        
        return Panel(progress, title="🔄 Progress", border_style="green")
        
    def _create_tasks_panel(self, state: Dict[str, Any]) -> Panel:
        """Create active tasks panel"""
        tasks = state['tasks']
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status", width=3)
        table.add_column("Task", style="cyan")
        table.add_column("Details", style="dim")
        table.add_column("Duration", style="yellow", width=8)
        
        # Sort tasks by status priority and creation time
        status_priority = {
            TaskStatus.IN_PROGRESS: 0,
            TaskStatus.PENDING: 1, 
            TaskStatus.FAILED: 2,
            TaskStatus.COMPLETED: 3
        }
        
        sorted_tasks = sorted(
            tasks.values(),
            key=lambda t: (status_priority.get(t.status, 4), t.start_time or datetime.min)
        )
        
        # Show only recent tasks (last 10)
        for task in sorted_tasks[-10:]:
            if task.status == TaskStatus.IN_PROGRESS:
                status_icon = "[bold green]►[/bold green]"
            elif task.status == TaskStatus.COMPLETED:
                status_icon = "[bold green]✓[/bold green]"
            elif task.status == TaskStatus.FAILED:
                status_icon = "[bold red]✗[/bold red]"
            else:
                status_icon = "[dim]○[/dim]"
                
            duration = task.duration()
            duration_str = str(duration).split('.')[0] if duration else ""
            
            details = task.details
            if task.error_message:
                details = f"[red]{task.error_message}[/red]"
                
            table.add_row(status_icon, task.name, details, duration_str)
            
        return Panel(table, title="📋 Tasks", border_style="yellow")
        
    def _create_batches_panel(self, state: Dict[str, Any]) -> Panel:
        """Create batch status panel"""
        batches = state['batches']
        
        if not batches:
            return Panel("[dim]No active batches[/dim]", title="📦 Batch Status", border_style="blue")
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Batch ID", width=12)
        table.add_column("Progress", width=15)
        table.add_column("Status", style="cyan")
        table.add_column("Model", style="dim")
        
        for batch in sorted(batches.values(), key=lambda b: b.created_at, reverse=True)[:8]:
            batch_id_short = batch.batch_id[:12] + "..." if len(batch.batch_id) > 12 else batch.batch_id
            
            # Create progress bar
            progress_pct = (batch.completed_requests / batch.total_requests * 100) if batch.total_requests > 0 else 0
            progress_bar = "█" * int(progress_pct / 10) + "░" * (10 - int(progress_pct / 10))
            progress_text = f"[{progress_bar}] {batch.completed_requests}/{batch.total_requests}"
            
            status_color = {
                'pending': 'yellow',
                'in_progress': 'blue', 
                'completed': 'green',
                'failed': 'red'
            }.get(batch.status, 'white')
            
            status_text = f"[{status_color}]{batch.status}[/{status_color}]"
            model_short = batch.model.split('/')[-1] if '/' in batch.model else batch.model
            
            table.add_row(batch_id_short, progress_text, status_text, model_short)
            
        return Panel(table, title="📦 Batch Status", border_style="blue")
        
    def _create_threads_panel(self, state: Dict[str, Any]) -> Panel:
        """Create thread activity panel"""
        threads = state['threads']
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Thread", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Current Task", style="dim")
        table.add_column("Last Update", style="yellow", width=8)
        
        for thread in sorted(threads.values(), key=lambda t: t.last_update, reverse=True):
            time_ago = datetime.now() - thread.last_update
            if time_ago.total_seconds() < 60:
                update_str = f"{int(time_ago.total_seconds())}s ago"
            else:
                update_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                
            table.add_row(thread.name, thread.status, thread.current_task, update_str)
            
        return Panel(table, title="🧵 Thread Activity", border_style="purple")
        
    def _create_footer(self, state: Dict[str, Any]) -> Panel:
        """Create footer with controls"""
        footer_text = "Press Ctrl+C to stop generation | Live dashboard updates every 1s"
        return Panel(Align.center(Text(footer_text, style="dim")),
                    border_style="dim")

    def _create_logs_panel(self, state: Dict[str, Any]) -> Panel:
        """Create live logs panel"""
        logs = state.get('logs', [])
        
        log_text = ""
        for timestamp, message in logs[-10:]:
            time_str = timestamp.strftime("%H:%M:%S")
            log_text += f"[dim]{time_str}[/dim] {message}\n"
            
        return Panel(Text(log_text, no_wrap=True), title="🔴 Live Logs", border_style="red")
        
    def update_display(self):
        """Update the dashboard display"""
        state = self.task_manager.get_state()
        
        self.layout["header"].update(self._create_header(state))
        self.layout["progress"].update(self._create_progress_panel(state))
        self.layout["tasks"].update(self._create_tasks_panel(state))
        self.layout["batches"].update(self._create_batches_panel(state))
        self.layout["threads"].update(self._create_threads_panel(state))
        self.layout["logs"].update(self._create_logs_panel(state))
        self.layout["footer"].update(self._create_footer(state))
        
        return self.layout
        
    def run(self):
        """Run the live dashboard with exclusive terminal control"""
        # Show startup message before taking control
        self.console.print("[bold blue]📊 Live Dashboard Starting...[/bold blue]")
        
        # Force clear screen and start dashboard with screen mode for full control
        self.console.clear()
        with Live(self.update_display(), console=self.console, refresh_per_second=self.refresh_rate, screen=True, vertical_overflow="visible") as live:
            while not self.task_manager.is_shutdown:
                try:
                    live.update(self.update_display())
                    time.sleep(1.0 / self.refresh_rate)
                except KeyboardInterrupt:
                    # Signal shutdown to task manager so main thread knows to stop
                    self.console.print("\n[yellow]⚠️  Ctrl+C detected - signaling shutdown...[/yellow]")
                    self.task_manager.shutdown()
                    break
                    
        # Show final summary
        self.console.print("\n[bold green]✅ Dashboard stopped[/bold green]")


# Singleton instance for global access
_task_manager: Optional[TaskManager] = None
_dashboard: Optional[LiveDashboard] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


def get_dashboard() -> LiveDashboard:
    """Get the global dashboard instance"""
    global _dashboard, _task_manager
    if _dashboard is None:
        if _task_manager is None:
            _task_manager = TaskManager()
        _dashboard = LiveDashboard(_task_manager)
    return _dashboard


def start_dashboard_thread() -> threading.Thread:
    """Start the dashboard in a separate thread"""
    dashboard = get_dashboard()
    thread = threading.Thread(target=dashboard.run, name="Dashboard", daemon=True)
    thread.start()
    return thread


# Convenience function for external modules
def attach_logging(level: int = None):
    """Attach root logging to the dashboard's live logs panel"""
    tm = get_task_manager()
    return tm.attach_logging_handler(level)
