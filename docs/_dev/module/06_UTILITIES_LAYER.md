# 工具层实现指南

> Utilities Layer Implementation Guide
>
> 版本：1.0
> 更新日期：2025-10-27

---

## 目录

1. [概述](#概述)
2. [配置管理](#配置管理)
3. [依赖注入](#依赖注入)
4. [验证器](#验证器)
5. [异常体系](#异常体系)
6. [并行执行器](#并行执行器)
7. [辅助工具](#辅助工具)

---

## 概述

工具层提供跨层使用的通用工具和基础设施，包括配置管理、依赖注入、验证、异常处理等。

### 核心组件

- **配置管理**：类型安全的配置系统
- **依赖注入**：管理组件依赖关系
- **验证器**：数据验证工具
- **异常体系**：自定义异常层次结构
- **并行执行器**：并行处理支持

---

## 配置管理

### 使用Pydantic Settings

```python
"""
配置管理实现

使用Pydantic Settings实现类型安全的配置。
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    """路径配置"""

    model_config = SettingsConfigDict(
        env_prefix="EP_",
        case_sensitive=False,
    )

    # IDF和天气文件目录
    prototype_idf: Path = Field(default=Path("backend/data/prototypes"))
    tmy_dir: Path = Field(default=Path("backend/data/tmys"))
    ftmy_dir: Path = Field(default=Path("backend/data/ftmys"))

    # 输出目录
    output_dir: Path = Field(default=Path("backend/output"))
    baseline_dir: Path = Field(default=Path("backend/output/baseline"))
    pv_dir: Path = Field(default=Path("backend/output/pv"))
    optimization_dir: Path = Field(default=Path("backend/output/optimization"))

    # EnergyPlus配置
    eplus_executable: Optional[Path] = None
    idd_file: Path = Field(default=Path("backend/data/Energy+.idd"))

    @field_validator('*', mode='before')
    @classmethod
    def expand_path(cls, v):
        """展开路径"""
        if isinstance(v, (str, Path)):
            return Path(v).expanduser().resolve()
        return v


class SimulationConfig(BaseSettings):
    """模拟配置"""

    model_config = SettingsConfigDict(
        env_prefix="EP_SIM_",
    )

    start_year: int = Field(default=2040)
    end_year: int = Field(default=2040)
    output_suffix: str = Field(default="L")
    cleanup_extensions: list[str] = Field(
        default=['.audit', '.bnd', '.eio', '.end', '.mdd', '.mtd', '.rdd', '.shd']
    )


class Settings(BaseSettings):
    """
    主配置类

    集成所有子配置。

    Example:
        >>> settings = Settings()
        >>> print(settings.paths.prototype_idf)
        PosixPath('/path/to/backend/data/prototypes')
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 子配置
    paths: PathsConfig = Field(default_factory=PathsConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    # 其他配置
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    max_workers: int = Field(default=4, ge=1, le=16)

    def create_directories(self) -> None:
        """创建必要的目录"""
        directories = [
            self.paths.output_dir,
            self.paths.baseline_dir,
            self.paths.pv_dir,
            self.paths.optimization_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    获取全局配置实例

    单例模式，确保只创建一次。

    Returns:
        Settings: 配置对象

    Example:
        >>> from backend.utils.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.paths.prototype_idf)
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.create_directories()
    return _settings
```

---

## 依赖注入

### DependencyContainer

```python
"""
依赖注入容器

管理服务的生命周期和依赖关系。
"""

from typing import Any, Callable, Type, TypeVar, Dict, Optional

from loguru import logger


T = TypeVar('T')


class DependencyContainer:
    """
    依赖注入容器

    支持单例（Singleton）和瞬态（Transient）两种生命周期。

    Example:
        >>> container = DependencyContainer()
        >>>
        >>> # 注册单例
        >>> container.register_singleton(ILogger, LoguruLogger())
        >>>
        >>> # 注册工厂
        >>> container.register_transient(
        ...     IService,
        ...     lambda: ServiceImpl(container.resolve(ILogger))
        ... )
        >>>
        >>> # 解析依赖
        >>> service = container.resolve(IService)
    """

    def __init__(self):
        self._singletons: Dict[Type, Any] = {}
        self._transients: Dict[Type, Callable[[], Any]] = {}
        self._logger = logger

    def register_singleton(self, interface: Type[T], instance: T) -> None:
        """
        注册单例

        单例在整个应用生命周期中只有一个实例。

        Args:
            interface: 接口类型
            instance: 实例对象

        Example:
            >>> config = Settings()
            >>> container.register_singleton(Settings, config)
        """
        self._singletons[interface] = instance
        self._logger.debug(f"Registered singleton: {interface.__name__}")

    def register_transient(
        self,
        interface: Type[T],
        factory: Callable[[], T],
    ) -> None:
        """
        注册瞬态（每次创建新实例）

        Args:
            interface: 接口类型
            factory: 工厂函数

        Example:
            >>> container.register_transient(
            ...     IService,
            ...     lambda: ServiceImpl()
            ... )
        """
        self._transients[interface] = factory
        self._logger.debug(f"Registered transient: {interface.__name__}")

    def resolve(self, interface: Type[T]) -> T:
        """
        解析依赖

        Args:
            interface: 接口类型

        Returns:
            实例对象

        Raises:
            ValueError: 未注册的接口

        Example:
            >>> service = container.resolve(IService)
        """
        # 首先检查单例
        if interface in self._singletons:
            return self._singletons[interface]

        # 然后检查瞬态
        if interface in self._transients:
            return self._transients[interface]()

        raise ValueError(
            f"No registration found for {interface}. "
            f"Use register_singleton() or register_transient() first."
        )

    def resolve_all(self, *interfaces: Type) -> tuple:
        """
        解析多个依赖

        Args:
            interfaces: 接口类型列表

        Returns:
            实例对象元组

        Example:
            >>> logger, config = container.resolve_all(ILogger, Settings)
        """
        return tuple(self.resolve(interface) for interface in interfaces)


def setup_container(settings: Settings) -> DependencyContainer:
    """
    设置依赖注入容器

    注册所有服务和依赖。

    Args:
        settings: 配置对象

    Returns:
        DependencyContainer: 配置好的容器

    Example:
        >>> settings = get_settings()
        >>> container = setup_container(settings)
        >>> orchestrator = container.resolve(SimulationOrchestrator)
    """
    container = DependencyContainer()

    # 注册配置（单例）
    container.register_singleton(Settings, settings)

    # 注册日志器（单例）
    from backend.infrastructure.logging import LoguruLogger
    logger_instance = LoguruLogger(
        log_dir=Path("logs"),
        level=settings.log_level,
    )
    container.register_singleton('ILogger', logger_instance)

    # 注册缓存服务（单例）
    from backend.infrastructure.cache import SmartCache
    cache = SmartCache(
        cache_dir=Path(".cache"),
        max_memory_items=100,
    )
    container.register_singleton('ICacheService', cache)

    # 注册EnergyPlus执行器（单例）
    from backend.infrastructure.energyplus import EnergyPlusExecutor
    executor = EnergyPlusExecutor(
        idd_path=settings.paths.idd_file,
    )
    container.register_singleton('IEnergyPlusExecutor', executor)

    # 注册结果解析器（单例）
    from backend.infrastructure.energyplus import ResultParser
    parser = ResultParser()
    container.register_singleton('IResultParser', parser)

    # 注册工厂（单例）
    from backend.factories import ServiceFactory, BuildingFactory
    building_factory = BuildingFactory()
    container.register_singleton(BuildingFactory, building_factory)

    service_factory = ServiceFactory(container)
    container.register_singleton(ServiceFactory, service_factory)

    # 注册仓储（单例）
    from backend.infrastructure.repositories import FileSystemBuildingRepository
    building_repo = FileSystemBuildingRepository(
        base_directory=settings.paths.prototype_idf,
        building_factory=building_factory,
    )
    container.register_singleton('IBuildingRepository', building_repo)

    # 注册编排器（单例）
    from backend.services.orchestration import SimulationOrchestrator
    orchestrator = SimulationOrchestrator(
        service_factory=service_factory,
        cache_service=cache,
        max_workers=settings.max_workers,
    )
    container.register_singleton(SimulationOrchestrator, orchestrator)

    return container
```

---

## 验证器

### 数据验证

```python
"""
数据验证工具
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, field_validator, ValidationError as PydanticValidationError

from backend.utils.exceptions import ValidationError


class FilePathValidator(BaseModel):
    """文件路径验证器"""

    file_path: Path

    @field_validator('file_path')
    @classmethod
    def validate_exists(cls, v: Path) -> Path:
        """验证文件存在"""
        if not v.exists():
            raise ValueError(f"File does not exist: {v}")
        return v

    @field_validator('file_path')
    @classmethod
    def validate_is_file(cls, v: Path) -> Path:
        """验证是文件而非目录"""
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return v


class IDFValidator(FilePathValidator):
    """IDF文件验证器"""

    @field_validator('file_path')
    @classmethod
    def validate_idf_extension(cls, v: Path) -> Path:
        """验证IDF扩展名"""
        if v.suffix.lower() != '.idf':
            raise ValueError(f"Not an IDF file: {v}")
        return v


class EPWValidator(FilePathValidator):
    """EPW文件验证器"""

    @field_validator('file_path')
    @classmethod
    def validate_epw_extension(cls, v: Path) -> Path:
        """验证EPW扩展名"""
        if v.suffix.lower() != '.epw':
            raise ValueError(f"Not an EPW file: {v}")
        return v


def validate_file_exists(file_path: Path, file_type: str = "File") -> None:
    """
    验证文件存在

    Args:
        file_path: 文件路径
        file_type: 文件类型描述

    Raises:
        ValidationError: 文件不存在
    """
    if not file_path.exists():
        raise ValidationError(f"{file_type} does not exist: {file_path}")


def validate_directory_exists(directory: Path) -> None:
    """
    验证目录存在

    Args:
        directory: 目录路径

    Raises:
        ValidationError: 目录不存在
    """
    if not directory.exists():
        raise ValidationError(f"Directory does not exist: {directory}")

    if not directory.is_dir():
        raise ValidationError(f"Path is not a directory: {directory}")
```

---

## 异常体系

### 自定义异常

```python
"""
自定义异常体系
"""


class EPWebUIException(Exception):
    """基础异常类"""
    pass


class ConfigurationError(EPWebUIException):
    """配置错误"""
    pass


class ValidationError(EPWebUIException):
    """验证错误"""
    pass


class SimulationError(EPWebUIException):
    """模拟执行错误"""
    pass


class ExecutionError(SimulationError):
    """EnergyPlus执行错误"""
    pass


class ParsingError(SimulationError):
    """结果解析错误"""
    pass


class EPFileNotFoundError(EPWebUIException):
    """文件未找到错误"""
    pass


class CacheError(EPWebUIException):
    """缓存错误"""
    pass


class RepositoryError(EPWebUIException):
    """仓储错误"""
    pass


class OptimizationError(EPWebUIException):
    """优化错误"""
    pass
```

---

## 并行执行器

### ParallelExecutor

```python
"""
并行执行器

支持多线程和多进程并行执行。
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, TypeVar
import multiprocessing

from loguru import logger


T = TypeVar('T')
R = TypeVar('R')


class ParallelExecutor:
    """
    并行执行器

    支持多线程（I/O密集）和多进程（CPU密集）。

    Attributes:
        _max_workers: 最大工作数
        _use_processes: 是否使用多进程

    Example:
        >>> executor = ParallelExecutor(max_workers=4, use_processes=True)
        >>>
        >>> def process_item(item):
        ...     return item * 2
        >>>
        >>> items = [1, 2, 3, 4, 5]
        >>> results = executor.map(process_item, items)
        >>> print(results)
        [2, 4, 6, 8, 10]
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
    ):
        """
        初始化并行执行器

        Args:
            max_workers: 最大工作数（None表示CPU核心数）
            use_processes: 是否使用多进程（True）还是多线程（False）
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        self._max_workers = max_workers
        self._use_processes = use_processes
        self._logger = logger

    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[R]:
        """
        并行映射函数到列表

        Args:
            func: 要执行的函数
            items: 输入列表
            progress_callback: 进度回调 (completed, total)

        Returns:
            结果列表

        Example:
            >>> def square(x):
            ...     return x ** 2
            >>>
            >>> results = executor.map(square, [1, 2, 3, 4])
        """
        executor_class = ProcessPoolExecutor if self._use_processes else ThreadPoolExecutor

        total = len(items)
        results: List[Optional[R]] = [None] * total
        completed = 0

        self._logger.info(f"Starting parallel execution: {total} items, {self._max_workers} workers")

        with executor_class(max_workers=self._max_workers) as executor:
            future_to_idx = {executor.submit(func, item): idx for idx, item in enumerate(items)}

            for future in as_completed(future_to_idx):
                try:
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                except Exception as e:
                    self._logger.error(f"Task failed: {e}")
                    results[idx] = None

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        self._logger.info(f"Parallel execution completed: {completed}/{total} tasks")
        return results
```

---

## 辅助工具

### 文件辅助函数

```python
"""
文件辅助函数
"""

from pathlib import Path
from typing import List


def find_files(
    directory: Path,
    pattern: str = "*",
    recursive: bool = True,
) -> List[Path]:
    """
    查找文件

    Args:
        directory: 目录
        pattern: 文件模式
        recursive: 是否递归搜索

    Returns:
        文件路径列表
    """
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def clean_directory(
    directory: Path,
    extensions: List[str],
    keep_files: bool = False,
) -> int:
    """
    清理目录

    Args:
        directory: 目录
        extensions: 要删除的扩展名列表
        keep_files: 如果True，只删除指定扩展名外的文件

    Returns:
        删除的文件数
    """
    count = 0
    for file in directory.glob("*"):
        if file.is_file():
            if keep_files:  
                should_delete = (file.suffix not in extensions)  
            else:  
                should_delete = (file.suffix in extensions)

            if should_delete:
                try:
                    file.unlink()
                    count += 1
                except Exception:
                    pass

    return count


def ensure_directory(directory: Path) -> Path:
    """
    确保目录存在

    Args:
        directory: 目录路径

    Returns:
        目录路径
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory
```

---

## 使用示例

```python
"""
工具层使用示例
"""

from backend.utils.config import get_settings, setup_container
from backend.utils.exceptions import ValidationError
from backend.utils.validators import validate_file_exists
from backend.utils.parallel import ParallelExecutor


def example_usage():
    # 1. 配置管理
    settings = get_settings()
    print(f"IDF directory: {settings.paths.prototype_idf}")
    print(f"Max workers: {settings.max_workers}")

    # 2. 依赖注入
    container = setup_container(settings)
    orchestrator = container.resolve(SimulationOrchestrator)

    # 3. 验证
    try:
        validate_file_exists(Path("test.idf"), "IDF file")
    except ValidationError as e:
        print(f"Validation failed: {e}")

    # 4. 并行执行
    executor = ParallelExecutor(max_workers=4)

    def process(x):
        return x * 2

    results = executor.map(process, list(range(10)))
    print(f"Results: {results}")


if __name__ == "__main__":
    example_usage()
```

---

## 总结

工具层提供了：

1. **类型安全的配置**：使用Pydantic Settings
2. **依赖注入**：管理组件依赖
3. **验证工具**：数据验证
4. **异常体系**：清晰的错误层次
5. **并行执行**：性能优化

这些工具是整个系统的基石。

---

**文档版本**: 1.0
**最后更新**: 2025-10-27
