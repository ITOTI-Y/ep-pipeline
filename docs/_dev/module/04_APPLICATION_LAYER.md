# 应用层实现指南

> Application Layer Implementation Guide
>
> 版本：1.0
> 更新日期：2025-10-27

---

## 目录

1. [概述](#概述)
2. [设计原则](#设计原则)
3. [目录结构](#目录结构)
4. [CLI接口设计](#cli接口设计)
5. [批处理编排](#批处理编排)
6. [进度监控](#进度监控)
7. [事件系统](#事件系统)
8. [配置管理](#配置管理)
9. [使用示例](#使用示例)
10. [测试策略](#测试策略)

---

## 概述

应用层（Application Layer）是系统的最外层，负责与用户交互、接收命令、编排业务流程并展示结果。

### 核心职责

1. **用户界面**：CLI命令行接口
2. **请求处理**：解析用户输入，调用服务层
3. **流程编排**：协调复杂的业务流程
4. **进度反馈**：实时进度监控和反馈
5. **错误处理**：友好的错误信息展示

### 设计特点

- ✅ 清晰的命令行界面（Click框架）
- ✅ 观察者模式实现进度监控
- ✅ 事件驱动架构
- ✅ 完整的错误处理和日志记录
- ✅ 丰富的输出格式支持

---

## 设计原则

### 薄应用层原则

```python
# ✅ 应用层只做编排，不做业务逻辑
class SimulationCLI:
    def run_baseline(self, building_id: str) -> None:
        """运行基准模拟（编排）"""
        # 1. 解析输入
        building = self._building_repo.find_by_id(building_id)

        # 2. 调用服务层
        result = self._orchestrator.execute_baseline(building)

        # 3. 展示结果
        self._display_result(result)

# ❌ 不要在应用层实现业务逻辑
class SimulationCLI:
    def run_baseline(self, building_id: str) -> None:
        # 错误：应用层不应该有复杂的业务逻辑
        idf = IDF(building.idf_file_path)
        idf.run(weather=weather_file, ...)
        ...
```

### 依赖方向

```
CLI (应用层)
  ↓ 依赖
Service Layer (服务层)
  ↓ 依赖
Domain Layer (领域层)
```

应用层依赖服务层，但服务层不知道应用层的存在。

---

## 目录结构

```
backend/application/
├── __init__.py
├── cli/                          # CLI命令
│   ├── __init__.py
│   ├── main.py                  # 主入口
│   ├── commands/                # 命令模块
│   │   ├── __init__.py
│   │   ├── simulate.py         # 模拟命令
│   │   ├── optimize.py         # 优化命令
│   │   ├── analyze.py          # 分析命令
│   │   └── config.py           # 配置命令
│   │
│   └── formatters/              # 输出格式化
│       ├── __init__.py
│       ├── table_formatter.py
│       ├── json_formatter.py
│       └── csv_formatter.py
│
├── observers/                    # 观察者
│   ├── __init__.py
│   ├── progress_observer.py    # 进度观察者
│   ├── logging_observer.py     # 日志观察者
│   └── metric_observer.py      # 指标观察者
│
├── events/                       # 事件系统
│   ├── __init__.py
│   ├── event_bus.py            # 事件总线
│   └── events.py               # 事件定义
│
└── workflows/                    # 工作流
    ├── __init__.py
    ├── batch_workflow.py       # 批处理工作流
    └── optimization_workflow.py # 优化工作流
```

---

## CLI接口设计

### 使用Click框架

```python
"""
CLI主入口

使用Click构建命令行接口。
"""

import click
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.utils.config import get_settings, setup_container
from backend.application.cli.commands import (
    simulate,
    optimize,
    analyze,
)


@click.group()
@click.version_option(version="1.0.0")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="配置文件路径",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="增加输出详细程度（-v, -vv, -vvv）",
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], verbose: int) -> None:
    """
    EP-WebUI 命令行工具

    建筑能耗模拟与优化平台

    \b
    示例：
        # 运行基准模拟
        ep-webui simulate baseline --building Chicago_OfficeLarge --weather TMY

        # 运行优化
        ep-webui optimize --building Chicago_OfficeLarge --iterations 100

        # 批量模拟
        ep-webui simulate batch --input jobs.yaml --output results/
    """
    # 设置日志级别
    log_levels = ["WARNING", "INFO", "DEBUG"]
    log_level = log_levels[min(verbose, len(log_levels) - 1)]

    # 初始化配置
    settings = get_settings()
    settings.log_level = log_level

    # 设置依赖注入容器
    container = setup_container(settings)

    # 保存到上下文
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings
    ctx.obj["container"] = container

    logger.info(f"EP-WebUI v1.0.0 initialized with log level: {log_level}")


# 注册命令组
cli.add_command(simulate.simulate)
cli.add_command(optimize.optimize)
cli.add_command(analyze.analyze)


if __name__ == "__main__":
    cli()
```

### 模拟命令

```python
"""
模拟命令实现
"""

import click
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from backend.domain.models import BuildingType, SimulationType
from backend.services.orchestration import SimulationOrchestrator
from backend.application.observers import ProgressObserver


console = Console()


@click.group()
def simulate() -> None:
    """模拟命令组"""
    pass


@simulate.command()
@click.option(
    "--building",
    "-b",
    required=True,
    help="建筑标识符（如 Chicago_OfficeLarge）",
)
@click.option(
    "--weather",
    "-w",
    required=True,
    help="天气文件场景（TMY/FTMY）",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("output/baseline"),
    help="输出目录",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    help="是否使用缓存",
)
@click.pass_context
def baseline(
    ctx: click.Context,
    building: str,
    weather: str,
    output: Path,
    cache: bool,
) -> None:
    """
    运行基准模拟

    \b
    示例：
        ep-webui simulate baseline -b Chicago_OfficeLarge -w TMY
    """
    container = ctx.obj["container"]

    console.print(f"\n[bold blue]运行基准模拟[/bold blue]")
    console.print(f"  建筑: {building}")
    console.print(f"  天气: {weather}")
    console.print(f"  输出: {output}")
    console.print(f"  缓存: {'启用' if cache else '禁用'}\n")

    try:
        # 解析建筑
        orchestrator: SimulationOrchestrator = container.resolve(SimulationOrchestrator)
        building_repo = container.resolve('IBuildingRepository')
        weather_repo = container.resolve('IWeatherRepository')

        # 查找建筑和天气
        buildings = building_repo.find_by_name(building)
        if not buildings:
            console.print(f"[red]错误: 找不到建筑 '{building}'[/red]")
            raise click.Abort()

        building_obj = buildings[0]

        weathers = weather_repo.find_by_scenario(weather)
        if not weathers:
            console.print(f"[red]错误: 找不到天气文件 '{weather}'[/red]")
            raise click.Abort()

        weather_obj = weathers[0]

        # 创建任务
        from backend.domain.models import SimulationJob
        job = SimulationJob(
            building=building_obj,
            weather_file=weather_obj,
            simulation_type=SimulationType.BASELINE,
            output_directory=output,
            output_prefix="baseline",
        )

        # 执行模拟（带进度条）
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]执行模拟...", total=1)

            def on_progress(completed: int, total: int):
                progress.update(task, completed=completed, total=total)

            results = orchestrator.execute_batch(
                jobs=[job],
                progress_callback=on_progress,
                use_cache=cache,
            )

        # 显示结果
        result = results[0]
        if result.success:
            _display_success_result(result)
        else:
            _display_failure_result(result)
            raise click.Abort()

    except Exception as e:
        logger.exception("模拟执行失败")
        console.print(f"\n[red]模拟失败: {e}[/red]")
        raise click.Abort()


@simulate.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="批处理任务文件（YAML/JSON）",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("output/batch"),
    help="输出目录",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=4,
    help="并行工作进程数",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    help="是否使用缓存",
)
@click.pass_context
def batch(
    ctx: click.Context,
    input_file: Path,
    output: Path,
    workers: int,
    cache: bool,
) -> None:
    """
    批量模拟

    从YAML或JSON文件读取任务列表并批量执行。

    \b
    输入文件格式示例（YAML）:
        jobs:
          - building: Chicago_OfficeLarge
            weather: TMY
            type: baseline
          - building: Chicago_OfficeLarge
            weather: FTMY
            type: baseline

    \b
    示例：
        ep-webui simulate batch -i jobs.yaml -o results/ -w 8
    """
    container = ctx.obj["container"]

    console.print(f"\n[bold blue]批量模拟[/bold blue]")
    console.print(f"  输入文件: {input_file}")
    console.print(f"  输出目录: {output}")
    console.print(f"  工作进程: {workers}")
    console.print(f"  缓存: {'启用' if cache else '禁用'}\n")

    try:
        # 加载任务
        import yaml
        with open(input_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        job_configs = data.get('jobs', [])
        console.print(f"加载了 {len(job_configs)} 个任务\n")

        # 创建任务列表
        orchestrator: SimulationOrchestrator = container.resolve(SimulationOrchestrator)
        jobs = _create_jobs_from_configs(job_configs, output, container)

        # 执行批处理（带进度条）
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]批量执行模拟...",
                total=len(jobs),
            )

            def on_progress(completed: int, total: int):
                progress.update(task, completed=completed, total=total)

            results = orchestrator.execute_batch(
                jobs=jobs,
                progress_callback=on_progress,
                use_cache=cache,
            )

        # 统计结果
        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count

        console.print(f"\n[bold green]批量执行完成[/bold green]")
        console.print(f"  成功: {success_count}")
        console.print(f"  失败: {failure_count}")

        # 显示汇总表格
        _display_batch_summary(results)

        # 保存结果
        _save_batch_results(results, output / "summary.json")

    except Exception as e:
        logger.exception("批量模拟失败")
        console.print(f"\n[red]批量模拟失败: {e}[/red]")
        raise click.Abort()


def _display_success_result(result) -> None:
    """显示成功结果"""
    table = Table(title="模拟结果", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    table.add_row("Source EUI", f"{result.source_eui:.2f} kWh/m²/yr")
    table.add_row("Site EUI", f"{result.site_eui:.2f} kWh/m²/yr")
    table.add_row("执行时间", f"{result.execution_time:.2f} 秒")
    table.add_row("输出目录", str(result.output_directory))

    console.print(table)


def _display_failure_result(result) -> None:
    """显示失败结果"""
    console.print("[red]模拟失败[/red]\n")
    console.print("[yellow]错误信息:[/yellow]")
    for error in result.error_messages:
        console.print(f"  • {error}")


def _display_batch_summary(results: list) -> None:
    """显示批量结果汇总"""
    table = Table(title="批量结果汇总", show_header=True, header_style="bold magenta")
    table.add_column("任务ID", style="cyan")
    table.add_column("状态", style="white")
    table.add_column("Source EUI", style="green")
    table.add_column("时间(s)", style="yellow")

    for result in results:
        status = "✓ 成功" if result.success else "✗ 失败"
        eui = f"{result.source_eui:.2f}" if result.source_eui else "N/A"
        time_str = f"{result.execution_time:.2f}" if result.execution_time else "N/A"

        table.add_row(
            str(result.job_id)[:8],
            status,
            eui,
            time_str,
        )

    console.print(table)


def _create_jobs_from_configs(configs: list, output_base: Path, container) -> list:
    """从配置创建任务列表"""
    from backend.domain.models import SimulationJob, SimulationType

    building_repo = container.resolve('IBuildingRepository')
    weather_repo = container.resolve('IWeatherRepository')

    jobs = []
    for i, config in enumerate(configs):
        # 查找建筑
        buildings = building_repo.find_by_name(config['building'])
        if not buildings:
            continue
        building = buildings[0]

        # 查找天气
        weathers = weather_repo.find_by_scenario(config['weather'])
        if not weathers:
            continue
        weather = weathers[0]

        # 创建任务
        job = SimulationJob(
            building=building,
            weather_file=weather,
            simulation_type=config.get('type', 'baseline'),
            output_directory=output_base / f"job_{i:03d}",
            output_prefix=f"sim_{i:03d}",
        )
        jobs.append(job)

    return jobs


def _save_batch_results(results: list, output_file: Path) -> None:
    """保存批量结果到JSON"""
    import json

    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_data = []
    for result in results:
        results_data.append({
            'job_id': str(result.job_id),
            'success': result.success,
            'source_eui': result.source_eui,
            'site_eui': result.site_eui,
            'execution_time': result.execution_time,
            'errors': result.error_messages,
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    console.print(f"\n结果已保存到: {output_file}")
```

---

## 批处理编排

### BatchOrchestrator

```python
"""
批处理编排器

负责大规模批量任务的调度和执行。
"""

from typing import List, Callable, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import time

from loguru import logger

from backend.domain.models import SimulationJob, SimulationResult
from backend.services.orchestration import SimulationOrchestrator
from backend.application.events import EventBus, SimulationStartedEvent, SimulationCompletedEvent


@dataclass
class BatchConfig:
    """批处理配置"""

    max_workers: int = 4
    use_cache: bool = True
    retry_failed: bool = True
    max_retries: int = 3
    checkpoint_interval: int = 10  # 每10个任务保存检查点
    checkpoint_file: Optional[Path] = None


class BatchOrchestrator:
    """
    批处理编排器

    支持：
    - 大规模并行执行
    - 进度检查点
    - 失败重试
    - 事件通知
    - 资源管理

    Example:
        >>> config = BatchConfig(max_workers=8, use_cache=True)
        >>> orchestrator = BatchOrchestrator(
        ...     simulation_orchestrator=sim_orchestrator,
        ...     event_bus=event_bus,
        ...     config=config,
        ... )
        >>>
        >>> results = orchestrator.execute_batch(jobs, progress_callback=print_progress)
    """

    def __init__(
        self,
        simulation_orchestrator: SimulationOrchestrator,
        event_bus: EventBus,
        config: BatchConfig,
    ):
        """
        初始化批处理编排器

        Args:
            simulation_orchestrator: 模拟编排器
            event_bus: 事件总线
            config: 批处理配置
        """
        self._orchestrator = simulation_orchestrator
        self._event_bus = event_bus
        self._config = config
        self._logger = logger

        # 状态管理
        self._completed_jobs: List[str] = []
        self._failed_jobs: List[str] = []

    def execute_batch(
        self,
        jobs: List[SimulationJob],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SimulationResult]:
        """
        执行批处理任务

        Args:
            jobs: 任务列表
            progress_callback: 进度回调函数

        Returns:
            结果列表
        """
        self._logger.info(f"Starting batch execution: {len(jobs)} jobs")

        start_time = time.time()
        total_jobs = len(jobs)
        completed = 0

        # 加载检查点（如果存在）
        checkpoint_data = self._load_checkpoint()
        if checkpoint_data:
            self._completed_jobs = checkpoint_data['completed']
            jobs = [job for job in jobs if str(job.id) not in self._completed_jobs]
            completed = len(self._completed_jobs)
            self._logger.info(f"Resumed from checkpoint: {completed} jobs already completed")

        results: List[SimulationResult] = []

        # 分批执行
        batch_size = self._config.max_workers * 2
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]

            # 执行当前批次
            batch_results = self._execute_batch_chunk(
                batch,
                lambda c, t: progress_callback(completed + c, total_jobs) if progress_callback else None,
            )

            results.extend(batch_results)
            completed += len(batch_results)

            # 保存检查点
            if self._config.checkpoint_file and completed % self._config.checkpoint_interval == 0:
                self._save_checkpoint(completed, total_jobs)

        elapsed = time.time() - start_time

        self._logger.info(
            f"Batch execution completed in {elapsed:.2f}s. "
            f"Success: {sum(1 for r in results if r.success)}/{total_jobs}"
        )

        # 发布完成事件
        self._event_bus.publish(BatchCompletedEvent(
            total_jobs=total_jobs,
            successful_jobs=sum(1 for r in results if r.success),
            failed_jobs=sum(1 for r in results if not r.success),
            execution_time=elapsed,
        ))

        return results

    def _execute_batch_chunk(
        self,
        jobs: List[SimulationJob],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[SimulationResult]:
        """执行批次中的一个块"""
        return self._orchestrator.execute_batch(
            jobs=jobs,
            progress_callback=progress_callback,
            use_cache=self._config.use_cache,
        )

    def _save_checkpoint(self, completed: int, total: int) -> None:
        """保存检查点"""
        if not self._config.checkpoint_file:
            return

        import json

        checkpoint_data = {
            'completed': self._completed_jobs,
            'failed': self._failed_jobs,
            'progress': {
                'completed': completed,
                'total': total,
                'percentage': (completed / total * 100) if total > 0 else 0,
            },
            'timestamp': time.time(),
        }

        with open(self._config.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        self._logger.info(f"Checkpoint saved: {completed}/{total} jobs")

    def _load_checkpoint(self) -> Optional[dict]:
        """加载检查点"""
        if not self._config.checkpoint_file:
            return None

        if not self._config.checkpoint_file.exists():
            return None

        import json

        try:
            with open(self._config.checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self._logger.warning(f"Failed to load checkpoint: {e}")
            return None


@dataclass
class BatchCompletedEvent:
    """批处理完成事件"""

    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    execution_time: float
```

---

## 进度监控

### 观察者模式

```python
"""
进度观察者

使用观察者模式实现进度监控。
"""

from abc import ABC, abstractmethod
from typing import List
from datetime import datetime

from loguru import logger


class IProgressObserver(ABC):
    """进度观察者接口"""

    @abstractmethod
    def on_started(self, total: int) -> None:
        """任务开始"""
        pass

    @abstractmethod
    def on_progress(self, completed: int, total: int) -> None:
        """进度更新"""
        pass

    @abstractmethod
    def on_completed(self) -> None:
        """任务完成"""
        pass

    @abstractmethod
    def on_error(self, error: str) -> None:
        """发生错误"""
        pass


class ConsoleProgressObserver(IProgressObserver):
    """控制台进度观察者"""

    def __init__(self):
        self._start_time: Optional[datetime] = None

    def on_started(self, total: int) -> None:
        """任务开始"""
        self._start_time = datetime.now()
        print(f"\n开始执行 {total} 个任务...")

    def on_progress(self, completed: int, total: int) -> None:
        """进度更新"""
        percentage = (completed / total * 100) if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * completed / total) if total > 0 else 0
        bar = '=' * filled + '-' * (bar_length - filled)

        print(f'\r进度: [{bar}] {completed}/{total} ({percentage:.1f}%)', end='', flush=True)

    def on_completed(self) -> None:
        """任务完成"""
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
            print(f'\n\n✓ 所有任务完成！耗时: {elapsed:.2f}秒')

    def on_error(self, error: str) -> None:
        """发生错误"""
        print(f'\n✗ 错误: {error}')


class LoggingProgressObserver(IProgressObserver):
    """日志进度观察者"""

    def __init__(self):
        self._logger = logger

    def on_started(self, total: int) -> None:
        """任务开始"""
        self._logger.info(f"Batch execution started: {total} jobs")

    def on_progress(self, completed: int, total: int) -> None:
        """进度更新"""
        percentage = (completed / total * 100) if total > 0 else 0
        self._logger.debug(f"Progress: {completed}/{total} ({percentage:.1f}%)")

    def on_completed(self) -> None:
        """任务完成"""
        self._logger.info("Batch execution completed successfully")

    def on_error(self, error: str) -> None:
        """发生错误"""
        self._logger.error(f"Batch execution error: {error}")


class MetricProgressObserver(IProgressObserver):
    """指标收集观察者"""

    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'progress_history': [],
        }

    def on_started(self, total: int) -> None:
        """任务开始"""
        self.metrics['start_time'] = datetime.now()
        self.metrics['total_jobs'] = total

    def on_progress(self, completed: int, total: int) -> None:
        """进度更新"""
        self.metrics['completed_jobs'] = completed
        self.metrics['progress_history'].append({
            'timestamp': datetime.now().isoformat(),
            'completed': completed,
            'total': total,
        })

    def on_completed(self) -> None:
        """任务完成"""
        self.metrics['end_time'] = datetime.now()

    def on_error(self, error: str) -> None:
        """发生错误"""
        self.metrics['failed_jobs'] += 1

    def get_metrics(self) -> dict:
        """获取指标"""
        return self.metrics.copy()


class ProgressSubject:
    """
    进度主题

    管理观察者并通知进度更新。
    """

    def __init__(self):
        self._observers: List[IProgressObserver] = []

    def attach(self, observer: IProgressObserver) -> None:
        """附加观察者"""
        self._observers.append(observer)

    def detach(self, observer: IProgressObserver) -> None:
        """移除观察者"""
        self._observers.remove(observer)

    def notify_started(self, total: int) -> None:
        """通知任务开始"""
        for observer in self._observers:
            observer.on_started(total)

    def notify_progress(self, completed: int, total: int) -> None:
        """通知进度更新"""
        for observer in self._observers:
            observer.on_progress(completed, total)

    def notify_completed(self) -> None:
        """通知任务完成"""
        for observer in self._observers:
            observer.on_completed()

    def notify_error(self, error: str) -> None:
        """通知错误"""
        for observer in self._observers:
            observer.on_error(error)
```

---

## 事件系统

### 事件总线

```python
"""
事件系统

实现事件驱动架构。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Any
from datetime import datetime
from uuid import UUID

from loguru import logger


@dataclass
class Event:
    """基础事件类"""

    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SimulationStartedEvent(Event):
    """模拟开始事件"""

    job_id: UUID
    building_name: str
    simulation_type: str


@dataclass
class SimulationCompletedEvent(Event):
    """模拟完成事件"""

    job_id: UUID
    success: bool
    source_eui: float
    execution_time: float


@dataclass
class SimulationFailedEvent(Event):
    """模拟失败事件"""

    job_id: UUID
    error_message: str


class EventBus:
    """
    事件总线

    负责事件的发布和订阅。

    Example:
        >>> event_bus = EventBus()
        >>>
        >>> # 订阅事件
        >>> def on_simulation_completed(event: SimulationCompletedEvent):
        ...     print(f"Simulation {event.job_id} completed: {event.source_eui} kWh/m²/yr")
        >>>
        >>> event_bus.subscribe(SimulationCompletedEvent, on_simulation_completed)
        >>>
        >>> # 发布事件
        >>> event_bus.publish(SimulationCompletedEvent(
        ...     job_id=job.id,
        ...     success=True,
        ...     source_eui=150.0,
        ...     execution_time=60.5,
        ... ))
    """

    def __init__(self):
        self._subscribers: Dict[type, List[Callable]] = {}
        self._logger = logger

    def subscribe(self, event_type: type, handler: Callable[[Event], None]) -> None:
        """
        订阅事件

        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(handler)
        self._logger.debug(f"Subscribed to {event_type.__name__}")

    def unsubscribe(self, event_type: type, handler: Callable[[Event], None]) -> None:
        """
        取消订阅

        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    def publish(self, event: Event) -> None:
        """
        发布事件

        Args:
            event: 事件对象
        """
        event_type = type(event)

        self._logger.debug(f"Publishing event: {event_type.__name__}")

        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    self._logger.error(f"Event handler failed: {e}")

    def clear(self) -> None:
        """清空所有订阅"""
        self._subscribers.clear()
```

---

## 配置管理

### 命令行配置

```python
"""
命令行配置管理
"""

import click
from pathlib import Path
from typing import Optional

from backend.utils.config import get_settings


class ConfigManager:
    """
    配置管理器

    处理命令行参数和配置文件的加载。
    """

    @staticmethod
    def load_from_file(config_path: Path) -> dict:
        """从文件加载配置"""
        import yaml

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def merge_configs(base: dict, override: dict) -> dict:
        """合并配置"""
        result = base.copy()
        result.update(override)
        return result

    @staticmethod
    def validate_config(config: dict) -> bool:
        """验证配置"""
        required_keys = ['paths', 'simulation']
        return all(key in config for key in required_keys)


def load_config_from_cli(
    ctx: click.Context,
    config_file: Optional[Path] = None,
) -> dict:
    """
    从CLI加载配置

    Args:
        ctx: Click上下文
        config_file: 配置文件路径

    Returns:
        配置字典
    """
    settings = get_settings()

    # 如果提供了配置文件，加载并合并
    if config_file:
        file_config = ConfigManager.load_from_file(config_file)
        # 这里可以实现配置合并逻辑

    return {
        'paths': settings.paths,
        'simulation': settings.simulation,
        'max_workers': settings.max_workers,
    }
```

---

## 使用示例

### 基本使用

```bash
# 1. 运行单个基准模拟
ep-webui simulate baseline -b Chicago_OfficeLarge -w TMY

# 2. 运行批量模拟
ep-webui simulate batch -i jobs.yaml -o results/ -w 8 --cache

# 3. 运行优化
ep-webui optimize -b Chicago_OfficeLarge -w TMY --iterations 100

# 4. 分析结果
ep-webui analyze compare baseline.csv ecm.csv

# 5. 查看配置
ep-webui config show

# 6. 详细日志
ep-webui -vvv simulate baseline -b Chicago_OfficeLarge -w TMY
```

### Python API使用

```python
"""
应用层Python API使用示例
"""

from pathlib import Path

from backend.utils.config import get_settings, setup_container
from backend.application.cli.commands.simulate import baseline
from backend.application.observers import (
    ConsoleProgressObserver,
    LoggingProgressObserver,
    MetricProgressObserver,
    ProgressSubject,
)
from backend.application.events import EventBus, SimulationCompletedEvent


def example_with_observers():
    """使用观察者模式的示例"""

    # 初始化
    settings = get_settings()
    container = setup_container(settings)

    # 创建进度主题
    progress_subject = ProgressSubject()

    # 附加多个观察者
    console_observer = ConsoleProgressObserver()
    logging_observer = LoggingProgressObserver()
    metric_observer = MetricProgressObserver()

    progress_subject.attach(console_observer)
    progress_subject.attach(logging_observer)
    progress_subject.attach(metric_observer)

    # 执行任务
    orchestrator = container.resolve('SimulationOrchestrator')

    # 创建任务...
    jobs = [...]

    # 执行
    progress_subject.notify_started(len(jobs))

    def on_progress(completed, total):
        progress_subject.notify_progress(completed, total)

    try:
        results = orchestrator.execute_batch(
            jobs=jobs,
            progress_callback=on_progress,
        )
        progress_subject.notify_completed()

    except Exception as e:
        progress_subject.notify_error(str(e))

    # 获取指标
    metrics = metric_observer.get_metrics()
    print(f"Total execution time: {metrics['end_time'] - metrics['start_time']}")


def example_with_events():
    """使用事件系统的示例"""

    # 创建事件总线
    event_bus = EventBus()

    # 订阅事件
    def on_simulation_completed(event: SimulationCompletedEvent):
        print(f"✓ 模拟完成: {event.job_id}")
        print(f"  Source EUI: {event.source_eui} kWh/m²/yr")
        print(f"  执行时间: {event.execution_time}秒")

    event_bus.subscribe(SimulationCompletedEvent, on_simulation_completed)

    # 执行模拟...
    # 在模拟完成时会自动触发事件处理函数


if __name__ == "__main__":
    example_with_observers()
    example_with_events()
```

---

## 测试策略

### CLI测试

```python
"""
CLI命令测试
"""

import pytest
from click.testing import CliRunner

from backend.application.cli.main import cli


class TestCLI:
    """CLI测试"""

    @pytest.fixture
    def runner(self):
        """创建CLI运行器"""
        return CliRunner()

    def test_cli_help(self, runner):
        """测试帮助信息"""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'EP-WebUI' in result.output

    def test_simulate_baseline(self, runner, tmp_path):
        """测试基准模拟命令"""
        # 这里需要准备测试数据
        result = runner.invoke(cli, [
            'simulate', 'baseline',
            '--building', 'TestBuilding',
            '--weather', 'TMY',
            '--output', str(tmp_path),
            '--no-cache',
        ])

        # 根据实际情况验证结果
        # assert result.exit_code == 0


class TestObservers:
    """观察者测试"""

    def test_console_observer(self):
        """测试控制台观察者"""
        observer = ConsoleProgressObserver()

        observer.on_started(10)
        observer.on_progress(5, 10)
        observer.on_completed()

        # 验证输出（可以使用mock或捕获stdout）

    def test_metric_observer(self):
        """测试指标观察者"""
        observer = MetricProgressObserver()

        observer.on_started(100)
        observer.on_progress(50, 100)
        observer.on_completed()

        metrics = observer.get_metrics()
        assert metrics['total_jobs'] == 100
        assert metrics['completed_jobs'] == 50


class TestEventBus:
    """事件总线测试"""

    def test_subscribe_and_publish(self):
        """测试订阅和发布"""
        event_bus = EventBus()

        received_events = []

        def handler(event: SimulationCompletedEvent):
            received_events.append(event)

        event_bus.subscribe(SimulationCompletedEvent, handler)

        event = SimulationCompletedEvent(
            job_id=uuid4(),
            success=True,
            source_eui=150.0,
            execution_time=60.0,
        )

        event_bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0] == event
```

---

## 总结

应用层实现了：

### 核心特性

1. **CLI接口**：基于Click的命令行工具
2. **批处理**：大规模并行执行支持
3. **进度监控**：观察者模式实现
4. **事件系统**：事件驱动架构
5. **丰富输出**：表格、JSON、CSV等格式

### 可扩展性

添加新命令：

```python
# 1. 创建新命令文件
# backend/application/cli/commands/my_command.py

@click.command()
def my_command():
    """新命令"""
    pass

# 2. 注册到主CLI
# backend/application/cli/main.py
from .commands import my_command
cli.add_command(my_command.my_command)
```

### 下一步

继续阅读：
- [05_REPOSITORY_LAYER.md](05_REPOSITORY_LAYER.md) - 仓储层实现
- [07_TESTING_STRATEGY.md](07_TESTING_STRATEGY.md) - 测试策略

---

**文档版本**: 1.0
**最后更新**: 2025-10-27
**下一篇**: [05_REPOSITORY_LAYER.md](05_REPOSITORY_LAYER.md)
