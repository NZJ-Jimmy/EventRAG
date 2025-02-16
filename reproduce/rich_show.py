from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, MofNCompleteColumn, SpinnerColumn
from rich.live import Live
# from rich import print
console = Console()

progress = Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    MofNCompleteColumn(),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    speed_estimate_period=240.0,
    refresh_per_second=1,
    transient=False
) 

# 创建 Live 对象
live = Live(
            progress, 
            console=console, 
            refresh_per_second=10
        )
