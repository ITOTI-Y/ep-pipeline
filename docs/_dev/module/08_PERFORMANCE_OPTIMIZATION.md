# 性能优化实施指南

> Performance Optimization Implementation Guide
>
> 版本：1.0
> 更新日期：2025-10-27

---

## 目录

1. [概述](#概述)
2. [优化目标](#优化目标)
3. [并行执行](#并行执行)
4. [智能缓存](#智能缓存)
5. [对象池化](#对象池化)
6. [内存优化](#内存优化)
7. [性能监控](#性能监控)
8. [性能测试](#性能测试)

---

## 概述

性能优化是重构的关键目标之一，通过并行执行、缓存、对象池等技术，预期实现**4-8倍**的性能提升。

### 优化原则

1. **测量优先**: 先测量，再优化
2. **瓶颈识别**: 优化最慢的部分
3. **权衡取舍**: 平衡性能与复杂度
4. **持续监控**: 持续跟踪性能指标

---

## 优化目标

### 性能指标

| 场景 | 当前时间 | 目标时间 | 提升 |
|------|---------|---------|------|
| 单个模拟 | 60s | 50s | 17% |
| 10个模拟（顺序） | 600s | 150s | 75% |
| 100个模拟（顺序） | 6000s | 800s | 87% |
| 缓存命中率 | 0% | >80% | +80% |

### 资源使用

- **CPU利用率**: 目标 80-90%
- **内存使用**: <2GB
- **磁盘I/O**: 最小化

---

## 并行执行

### 1. 进程池并行

```python
"""
使用joblib进行并行执行
"""

from joblib import Parallel, delayed, cpu_count
from typing import List, Callable, Optional
from loguru import logger


class ParallelSimulationExecutor:
    """
    并行模拟执行器

    使用多进程执行模拟，充分利用CPU核心。
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化执行器

        Args:
            max_workers: 最大工作进程数（None = CPU核心数）
        """
        if max_workers is None:
            max_workers = cpu_count()

        self._max_workers = max_workers
        self._logger = logger

    def execute_batch(
        self,
        jobs: List[SimulationJob],
        executor_func: Callable[[SimulationJob], SimulationResult],
    ) -> List[SimulationResult]:
        """
        并行执行批量任务

        Args:
            jobs: 任务列表
            executor_func: 执行函数

        Returns:
            结果列表
        """
        self._executor_func = executor_func
        try:
            results = Parallel(
                n_jobs=self._max_workers,
                backend='loky',
                verbose=0
            )(delayed(self._safe_execute)(job) for job in jobs)

            return results
        except Exception as e:
            self._logger.error(f"Parallel execution failed: {e}")
            raise

    def _safe_execute(self, job):
        try:
            return self._executor_func(job)
        except Exception as e:
            self._logger.error(f"Job failed: {e}")
            return SimulationResult(
                job_id = job.id,
                output_directory = job.output_directory,
                success = False
                error_message=str(e)
            )
```

### 2. 工作负载平衡

```python
"""
工作负载平衡策略
"""

from typing import List
import time


def estimate_job_duration(job: SimulationJob) -> float:
    """
    估算任务执行时间

    基于建筑规模、天气文件等因素估算。

    Args:
        job: 模拟任务

    Returns:
        估算时间（秒）
    """
    base_time = 60.0  # 基础时间

    # 根据建筑面积调整
    if job.building.floor_area:
        area_factor = job.building.floor_area / 5000.0
        base_time *= area_factor

    # 根据模拟类型调整
    type_factors = {
        "baseline": 1.0,
        "pv": 1.2,
        "optimization": 5.0,
    }
    type_factor = type_factors.get(job.simulation_type, 1.0)
    base_time *= type_factor

    return base_time


def balance_workload(
    jobs: List[SimulationJob],
    num_workers: int,
) -> List[List[SimulationJob]]:
    """
    平衡工作负载

    将任务分配到工作进程，使每个进程的总时间相近。

    Args:
        jobs: 任务列表
        num_workers: 工作进程数

    Returns:
        分配后的任务列表（每个工作进程一个列表）
    """
    # 估算每个任务的时间
    jobs_with_time = [
        (job, estimate_job_duration(job))
        for job in jobs
    ]

    # 按时间降序排序
    jobs_with_time.sort(key=lambda x: x[1], reverse=True)

    # 初始化工作进程负载
    worker_loads = [[] for _ in range(num_workers)]
    worker_times = [0.0] * num_workers

    # 贪心分配：每次分配给负载最轻的工作进程
    for job, duration in jobs_with_time:
        min_idx = worker_times.index(min(worker_times))
        worker_loads[min_idx].append(job)
        worker_times[min_idx] += duration

    return worker_loads
```

---

## 智能缓存

### 1. 多级缓存策略

```python
"""
多级缓存实现

L1: 内存缓存（热数据）
L2: 磁盘缓存（温数据）
L3: 远程缓存（可选，如Redis）
"""

from typing import Optional, Any, Dict, Tuple
from loguru import logger
import pickle
import hashlib
from pathlib import Path
import time


class MultiLevelCache:
    """
    多级缓存

    实现内存+磁盘的两级缓存。
    """

    def __init__(
        self,
        cache_dir: Path,
        max_memory_items: int = 100,
        max_disk_size_gb: float = 10.0,
    ):
        """
        初始化缓存

        Args:
            cache_dir: 磁盘缓存目录
            max_memory_items: 最大内存缓存项数
            max_disk_size_gb: 最大磁盘缓存大小（GB）
        """
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # L1: 内存缓存
        self._memory_cache: Dict[str, Tuple[Any, float, int]] = {}
        # 格式: {key: (value, expires_at, access_count)}

        self._max_memory_items = max_memory_items
        self._max_disk_size = max_disk_size_gb * 1024 * 1024 * 1024

        # 统计信息
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        先查L1（内存），再查L2（磁盘）。

        Args:
            key: 缓存键

        Returns:
            缓存值或None
        """
        # L1: 内存缓存
        if key in self._memory_cache:
            value, expires_at, access_count = self._memory_cache[key]

            if time.time() < expires_at:
                # 更新访问计数（LFU）
                self._memory_cache[key] = (value, expires_at, access_count + 1)
                self._hits += 1
                return value
            else:
                # 过期，删除
                del self._memory_cache[key]

        # L2: 磁盘缓存
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                value, expires_at = cached_data['value'], cached_data['expires_at']

                if time.time() < expires_at:
                    # 提升到L1
                    self._promote_to_memory(key, value, expires_at)
                    self._hits += 1
                    return value
                else:
                    # 过期，删除
                    cache_file.unlink()

            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        设置缓存值

        同时保存到L1和L2。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        expires_at = time.time() + ttl

        # L1: 内存缓存
        self._add_to_memory(key, value, expires_at)

        # L2: 磁盘缓存
        self._add_to_disk(key, value, expires_at)

    def _promote_to_memory(self, key: str, value: Any, expires_at: float) -> None:
        """提升到内存缓存"""
        self._add_to_memory(key, value, expires_at)

    def _add_to_memory(self, key: str, value: Any, expires_at: float) -> None:
        """添加到内存缓存"""
        # 如果满了，使用LFU策略淘汰
        if len(self._memory_cache) >= self._max_memory_items:
            # 找到访问次数最少的
            lfu_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k][2]
            )
            del self._memory_cache[lfu_key]

        self._memory_cache[key] = (value, expires_at, 1)

    def _add_to_disk(self, key: str, value: Any, expires_at: float) -> None:
        """添加到磁盘缓存"""
        # 检查磁盘空间
        self._cleanup_disk_if_needed()

        cache_file = self._get_cache_file(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'expires_at': expires_at,
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save disk cache: {e}")

    def _cleanup_disk_if_needed(self) -> None:
        """清理磁盘缓存（如果超过限制）"""
        total_size = sum(
            f.stat().st_size
            for f in self._cache_dir.glob("*.cache")
        )

        if total_size > self._max_disk_size:
            # 删除最旧的文件
            files = sorted(
                self._cache_dir.glob("*.cache"),
                key=lambda f: f.stat().st_mtime
            )

            for file in files[:len(files) // 4]:  # 删除25%
                file.unlink()

    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'memory_items': len(self._memory_cache),
        }
```

### 2. 缓存键策略

```python
def generate_cache_key(job: SimulationJob) -> str:
    """
    生成缓存键

    基于任务的关键属性生成唯一键。

    Args:
        job: 模拟任务

    Returns:
        缓存键
    """
    components = [
        job.building.get_identifier(),
        job.weather_file.get_identifier(),
        job.simulation_type,
    ]

    # 如果有ECM参数，包含其哈希
    if job.ecm_parameters:
        ecm_hash = hash(job.ecm_parameters)
        components.append(str(ecm_hash))

    return "_".join(components)
```

---

## 对象池化

### IDF对象池

```python
"""
IDF对象池

重用IDF对象，避免重复加载。
"""

from queue import Queue
from threading import Lock
from pathlib import Path

from eppy.modeleditor import IDF


class IDFObjectPool:
    """
    IDF对象池

    为每个IDF文件维护一个对象池。
    """

    def __init__(self, max_size: int = 5):
        """
        初始化对象池

        Args:
            max_size: 每个文件的最大池大小
        """
        self._max_size = max_size
        self._pools: Dict[Path, Queue] = {}
        self._locks: Dict[Path, Lock] = {}
        self._sizes: Dict[Path, int] = {}

    def acquire(self, idf_path: Path) -> IDF:
        """
        获取IDF对象

        Args:
            idf_path: IDF文件路径

        Returns:
            IDF对象
        """
        # 确保有池和锁
        if idf_path not in self._pools:
            self._pools[idf_path] = Queue(maxsize=self._max_size)
            self._locks[idf_path] = Lock()
            self._sizes[idf_path] = 0

        # 尝试从池中获取
        try:
            idf = self._pools[idf_path].get_nowait()
            return idf
        except:
            # 池为空，创建新对象
            with self._locks[idf_path]:
                if self._sizes[idf_path] < self._max_size:
                    idf = IDF(str(idf_path))
                    self._sizes[idf_path] += 1
                    return idf
                else:
                    # 达到最大值，等待
                    return self._pools[idf_path].get()

    def release(self, idf_path: Path, idf: IDF) -> None:
        """
        归还IDF对象

        Args:
            idf_path: IDF文件路径
            idf: IDF对象
        """
        try:
            self._pools[idf_path].put_nowait(idf)
        except:
            # 池已满，丢弃对象
            with self._locks[idf_path]:
                self._sizes[idf_path] -= 1


# 使用上下文管理器
from contextlib import contextmanager

@contextmanager
def borrow_idf(pool: IDFObjectPool, idf_path: Path):
    """
    借用IDF对象

    使用with语句自动归还。

    Example:
        >>> pool = IDFObjectPool()
        >>> with borrow_idf(pool, Path("building.idf")) as idf:
        ...     # 使用idf
        ...     idf.run(...)
    """
    idf = pool.acquire(idf_path)
    try:
        yield idf
    finally:
        pool.release(idf_path, idf)
```

---

## 内存优化

### 1. 生成器使用

```python
"""
使用生成器减少内存占用
"""

def process_large_results(result_directory: Path):
    """
    使用生成器处理大量结果

    而不是一次性加载所有结果。
    """
    for result_file in result_directory.glob("*Table.csv"):
        # 逐个处理，而不是全部加载
        result = parse_result(result_file)
        yield result


# 使用
for result in process_large_results(Path("output")):
    process_single_result(result)
    # 处理完后，result会被垃圾回收
```

### 2. 数据压缩

```python
"""
压缩大型数据结构
"""

import gzip
import pickle


def save_compressed(data: Any, file_path: Path) -> None:
    """保存压缩数据"""
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_compressed(file_path: Path) -> Any:
    """加载压缩数据"""
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)
```

---

## 性能监控

### 性能分析器

```python
"""
性能分析装饰器
"""

import time
import functools
from typing import Callable


def profile_performance(func: Callable) -> Callable:
    """
    性能分析装饰器

    记录函数执行时间。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        logger.info(
            f"Performance: {func.__name__} took {elapsed:.2f}s"
        )

        return result

    return wrapper


# 使用
@profile_performance
def expensive_operation():
    # 耗时操作
    ...
```

### 性能指标收集

```python
"""
性能指标收集
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PerformanceMetrics:
    """性能指标"""

    function_name: str
    execution_times: List[float] = field(default_factory=list)

    def add_time(self, time: float) -> None:
        """添加执行时间"""
        self.execution_times.append(time)

    def get_average(self) -> float:
        """平均时间"""
        return sum(self.execution_times) / len(self.execution_times)

    def get_min(self) -> float:
        """最小时间"""
        return min(self.execution_times)

    def get_max(self) -> float:
        """最大时间"""
        return max(self.execution_times)

    def summary(self) -> Dict[str, float]:
        """统计摘要"""
        return {
            'count': len(self.execution_times),
            'avg': self.get_average(),
            'min': self.get_min(),
            'max': self.get_max(),
        }
```

---

## 性能测试

### 基准测试

```python
"""
性能基准测试
"""

import pytest
import time


@pytest.mark.performance
def test_parallel_speedup():
    """测试并行加速比"""
    # 创建测试任务
    jobs = create_benchmark_jobs(count=10)

    # 顺序执行
    start = time.time()
    sequential_results = execute_sequential(jobs)
    sequential_time = time.time() - start

    # 并行执行
    start = time.time()
    parallel_results = execute_parallel(jobs, workers=4)
    parallel_time = time.time() - start

    # 计算加速比
    speedup = sequential_time / parallel_time

    print(f"\nPerformance Results:")
    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Parallel:   {parallel_time:.2f}s")
    print(f"  Speedup:    {speedup:.2f}x")

    # 验证至少3倍加速
    assert speedup >= 3.0


@pytest.mark.performance
def test_cache_performance():
    """测试缓存性能"""
    cache = MultiLevelCache(Path(".cache"))
    jobs = create_test_jobs(count=20)

    # 第一次运行（填充缓存）
    start = time.time()
    results1 = execute_with_cache(jobs, cache, use_cache=False)
    time1 = time.time() - start

    # 第二次运行（使用缓存）
    start = time.time()
    results2 = execute_with_cache(jobs, cache, use_cache=True)
    time2 = time.time() - start

    # 获取缓存统计
    stats = cache.get_stats()

    print(f"\nCache Performance:")
    print(f"  Without cache: {time1:.2f}s")
    print(f"  With cache:    {time2:.2f}s")
    print(f"  Speedup:       {time1/time2:.2f}x")
    print(f"  Hit rate:      {stats['hit_rate']:.1f}%")

    # 验证缓存命中率
    assert stats['hit_rate'] > 80.0
```

---

## 优化检查清单

### 实施前

- [ ] 建立性能基准
- [ ] 识别性能瓶颈
- [ ] 设定优化目标

### 实施中

- [ ] 并行执行实现
- [ ] 缓存系统实现
- [ ] 对象池实现
- [ ] 内存优化

### 实施后

- [ ] 性能测试验证
- [ ] 监控系统部署
- [ ] 文档更新
- [ ] 持续监控

---

## 总结

性能优化实现：

1. **并行执行**: 4-8倍加速
2. **智能缓存**: >80%命中率
3. **对象池化**: 减少对象创建开销
4. **内存优化**: 控制内存使用
5. **性能监控**: 持续跟踪指标

预期达到：**4-8倍性能提升**

---

**文档版本**: 1.0
**最后更新**: 2025-10-27
**系列完结**
