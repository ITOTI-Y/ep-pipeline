# 服务层实现指南

> Service Layer Implementation Guide
>
> 版本：1.0
> 更新日期：2025-10-27

---

## 目录

1. [概述](#概述)
2. [设计原则](#设计原则)
3. [目录结构](#目录结构)
4. [服务接口设计](#服务接口设计)
5. [基础服务实现](#基础服务实现)
6. [模拟服务](#模拟服务)
7. [优化服务](#优化服务)
8. [分析服务](#分析服务)
9. [服务编排](#服务编排)
10. [使用示例](#使用示例)
11. [测试策略](#测试策略)

---

## 概述

服务层（Service Layer）是系统的业务逻辑核心，负责协调领域模型、调用基础设施、编排复杂的业务流程。

### 核心职责

1. **业务逻辑编排**：协调多个领域对象完成复杂的业务操作
2. **事务管理**：确保操作的原子性和一致性
3. **依赖协调**：与基础设施层（文件系统、EnergyPlus）交互
4. **错误处理**：统一的异常处理和错误恢复

### 设计特点

- ✅ 接口驱动（Interface-driven）
- ✅ 依赖注入（Dependency Injection）
- ✅ 策略模式（Strategy Pattern）支持算法切换
- ✅ 模板方法（Template Method）统一流程
- ✅ 完整的类型提示和文档
- ✅ 可测试性（Testability）

### 与其他层的关系

```
Application Layer (应用层)
    ↓ 使用
Service Layer (服务层) ← [本文档]
    ↓ 使用
Domain Layer (领域层)
    ↑ 实现接口
Infrastructure Layer (基础设施层)
```

---

## 设计原则

### SOLID 原则应用

#### 1. 单一职责原则 (SRP)

每个服务只负责一个业务领域：

```python
# ✅ 正确：职责单一
class BaselineSimulationService:
    """只负责基准建筑模拟"""
    def run(self, context: SimulationContext) -> SimulationResult:
        ...

class PVSimulationService:
    """只负责光伏系统模拟"""
    def run(self, context: PVContext) -> SimulationResult:
        ...
```

#### 2. 开闭原则 (OCP)

通过接口和策略模式对扩展开放：

```python
# ✅ 新增服务无需修改现有代码
class ECMSimulationService(ISimulationService):
    """能效措施模拟服务 - 新增功能"""
    def run(self, context: SimulationContext) -> SimulationResult:
        ...
```

#### 3. 里氏替换原则 (LSP)

所有服务实现可以替换接口：

```python
def execute_simulation(service: ISimulationService, context: SimulationContext):
    """接受任何 ISimulationService 的实现"""
    return service.run(context)
```

#### 4. 接口隔离原则 (ISP)

客户端不依赖不需要的方法：

```python
# ✅ 分离接口
class ISimulationRunner(Protocol):
    """可运行接口"""
    def run(self, context: SimulationContext) -> SimulationResult: ...

class IResultValidator(Protocol):
    """可验证接口"""
    def validate(self, result: SimulationResult) -> bool: ...

# 服务只实现需要的接口
class SimpleService(ISimulationRunner):
    # 只实现 run，不实现 validate
    ...
```

#### 5. 依赖倒置原则 (DIP)

依赖抽象而非具体实现：

```python
# ✅ 依赖注入抽象
class BaselineSimulationService:
    def __init__(
        self,
        executor: IEnergyPlusExecutor,  # 依赖抽象
        parser: IResultParser,          # 依赖抽象
        logger: ILogger,                # 依赖抽象
    ):
        self._executor = executor
        self._parser = parser
        self._logger = logger
```

---

## 目录结构

```
backend/services/
├── __init__.py
├── interfaces/                    # 服务接口
│   ├── __init__.py
│   ├── i_simulation_service.py   # 模拟服务接口
│   ├── i_file_loader.py          # 文件加载接口
│   ├── i_result_processor.py     # 结果处理接口
│   └── i_energyplus_executor.py  # EnergyPlus执行接口
│
├── simulation/                    # 模拟服务
│   ├── __init__.py
│   ├── base_simulation_service.py      # 基础服务抽象类
│   ├── baseline_service.py             # 基准模拟
│   ├── pv_service.py                   # 光伏模拟
│   ├── ecm_service.py                  # ECM模拟
│   └── simulation_context.py           # 模拟上下文
│
├── optimization/                  # 优化服务
│   ├── __init__.py
│   ├── optimization_service.py   # 优化服务
│   └── strategies/               # 优化策略
│       ├── __init__.py
│       ├── i_optimization_strategy.py
│       ├── genetic_algorithm_strategy.py
│       ├── bayesian_optimization_strategy.py
│       └── pso_strategy.py
│
├── analysis/                      # 分析服务
│   ├── __init__.py
│   ├── sensitivity_service.py    # 敏感性分析
│   └── data_analysis_service.py  # 数据分析
│
└── orchestration/                 # 编排服务
    ├── __init__.py
    ├── simulation_orchestrator.py     # 模拟编排器
    └── batch_processor.py             # 批处理器
```

---

## 服务接口设计

### 1. 核心服务接口

#### ISimulationService

```python
"""
模拟服务接口

定义所有模拟服务的通用接口。
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from backend.domain.models import SimulationJob, SimulationResult

# 类型变量
TContext = TypeVar('TContext')


class ISimulationService(ABC, Generic[TContext]):
    """
    模拟服务接口

    所有模拟服务（Baseline, PV, ECM等）的统一接口。
    使用泛型支持不同的上下文类型。

    Type Parameters:
        TContext: 模拟上下文类型

    Example:
        >>> class BaselineService(ISimulationService[BaselineContext]):
        ...     def prepare(self, context: BaselineContext) -> None:
        ...         # 准备工作
        ...     def execute(self, context: BaselineContext) -> SimulationResult:
        ...         # 执行模拟
    """

    @abstractmethod
    def prepare(self, context: TContext) -> None:
        """
        准备模拟环境

        执行模拟前的准备工作，包括：
        - 创建输出目录
        - 验证文件存在性
        - 配置输出变量
        - 应用预处理逻辑

        Args:
            context: 模拟上下文，包含所有必要的配置和数据

        Raises:
            ValidationError: 配置验证失败
            FileNotFoundError: 必需文件不存在
            PreparationError: 准备过程失败
        """
        pass

    @abstractmethod
    def execute(self, context: TContext) -> SimulationResult:
        """
        执行模拟

        核心的模拟执行逻辑：
        - 调用 EnergyPlus 执行器
        - 解析模拟输出
        - 构建结果对象

        Args:
            context: 模拟上下文

        Returns:
            SimulationResult: 模拟结果对象

        Raises:
            SimulationError: 模拟执行失败
            ParsingError: 结果解析失败
        """
        pass

    @abstractmethod
    def cleanup(self, context: TContext) -> None:
        """
        清理临时文件和资源

        执行后的清理工作：
        - 删除临时文件
        - 释放资源
        - 记录日志

        Args:
            context: 模拟上下文

        Note:
            此方法应该是幂等的，即使多次调用也不会出错
        """
        pass

    def run(self, context: TContext) -> SimulationResult:
        """
        完整的模拟流程

        按顺序执行：prepare -> execute -> cleanup

        这是一个模板方法，定义了标准的执行流程。
        子类通常不需要重写此方法。

        Args:
            context: 模拟上下文

        Returns:
            SimulationResult: 模拟结果

        Example:
            >>> service = BaselineSimulationService(...)
            >>> context = BaselineContext(...)
            >>> result = service.run(context)
            >>> assert result.success
        """
        try:
            self.prepare(context)
            result = self.execute(context)
            return result
        finally:
            # 无论成功失败，都要清理
            self.cleanup(context)
```

#### IEnergyPlusExecutor

```python
"""
EnergyPlus 执行器接口
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from eppy.modeleditor import IDF


class ExecutionResult:
    """EnergyPlus 执行结果"""

    def __init__(
        self,
        success: bool,
        return_code: int,
        stdout: str,
        stderr: str,
        output_directory: Path,
    ):
        self.success = success
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.output_directory = output_directory
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, message: str) -> None:
        """添加错误信息"""
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """添加警告信息"""
        self.warnings.append(message)


class IEnergyPlusExecutor(ABC):
    """
    EnergyPlus 执行器接口

    负责调用 EnergyPlus 进行模拟。
    """

    @abstractmethod
    def run(
        self,
        idf: IDF,
        weather_file: Path,
        output_directory: Path,
        output_prefix: str,
        read_variables: bool = True,
    ) -> ExecutionResult:
        """
        运行 EnergyPlus 模拟

        Args:
            idf: IDF 对象
            weather_file: 天气文件路径
            output_directory: 输出目录
            output_prefix: 输出文件前缀
            read_variables: 是否读取输出变量

        Returns:
            ExecutionResult: 执行结果

        Raises:
            ExecutionError: 执行失败
        """
        pass

    @abstractmethod
    def validate_installation(self) -> bool:
        """
        验证 EnergyPlus 安装

        Returns:
            如果 EnergyPlus 正确安装则返回 True
        """
        pass
```

#### IResultParser

```python
"""
结果解析器接口
"""

from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID

from backend.domain.models import SimulationResult


class IResultParser(ABC):
    """
    结果解析器接口

    负责解析 EnergyPlus 输出文件（CSV、SQL等）。
    """

    @abstractmethod
    def parse(
        self,
        job_id: UUID,
        output_directory: Path,
        output_prefix: str,
    ) -> SimulationResult:
        """
        解析模拟结果

        Args:
            job_id: 任务 ID
            output_directory: 输出目录
            output_prefix: 输出文件前缀

        Returns:
            SimulationResult: 解析后的结果对象

        Raises:
            ParsingError: 解析失败
            FileNotFoundError: 输出文件不存在
        """
        pass

    @abstractmethod
    def parse_eui(self, table_csv_path: Path) -> dict[str, float]:
        """
        从 Table CSV 解析 EUI

        Args:
            table_csv_path: Table CSV 文件路径

        Returns:
            包含 EUI 指标的字典

        Example:
            >>> eui = parser.parse_eui(csv_path)
            >>> print(eui['source_eui'])
            150.5
        """
        pass
```

---

## 基础服务实现

### 模拟上下文（SimulationContext）

#### 为什么使用 Pydantic

使用 Pydantic 而非 dataclass 的关键优势：

1. **自动验证**: 运行时类型检查和数据验证
2. **架构一致性**: 与领域层保持一致（领域模型已使用 Pydantic）
3. **更好的错误信息**: 详细的验证错误提示
4. **JSON 序列化**: 内置序列化支持，便于缓存和日志
5. **字段验证器**: 使用 `@field_validator` 替代手写验证逻辑

```python
"""
模拟上下文（使用 Pydantic）

包含执行模拟所需的所有信息，提供自动验证和类型安全。
"""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from eppy.modeleditor import IDF

from backend.domain.models import SimulationJob


class SimulationContext(BaseModel):
    """
    模拟上下文基类

    使用 Pydantic 提供自动验证和类型安全。
    所有具体上下文的基类。

    Attributes:
        job: 模拟任务对象
        idf: IDF 对象（EnergyPlus 模型）
        working_directory: 工作目录路径
        metadata: 额外的元数据字典

    Example:
        >>> from pathlib import Path
        >>> context = SimulationContext(
        ...     job=simulation_job,
        ...     idf=idf_object,
        ...     working_directory=Path("output/baseline"),
        ... )
        >>> # 自动创建工作目录
        >>> assert context.working_directory.exists()
    """

    model_config = ConfigDict(
        validate_assignment=True,  # 赋值时也验证
        arbitrary_types_allowed=True,  # 允许 IDF 等非标准类型
        frozen=False,  # 允许修改（模拟过程中可能需要更新）
    )

    job: SimulationJob = Field(
        ...,
        description="模拟任务对象，包含建筑、天气文件等信息"
    )
    idf: Any = Field(  # 使用 Any 因为 IDF 不是 Pydantic 模型
        ...,
        description="EnergyPlus IDF 对象"
    )
    working_directory: Path = Field(
        ...,
        description="模拟工作目录，存放临时文件和输出"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外的元数据，用于扩展"
    )

    @field_validator("working_directory")
    def ensure_working_directory_exists(cls, v: Path) -> Path:
        """
        确保工作目录存在

        如果目录不存在，自动创建。

        注意：
            Pydantic V2 中，@field_validator 会自动将方法转换为 classmethod，
            因此不需要显式添加 @classmethod 装饰器。

        Args:
            cls: 类本身（自动传入）
            v: working_directory 字段的值

        Returns:
            Path: 验证后的路径
        """
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("idf")
    def validate_idf(cls, v: Any) -> Any:
        """
        验证 IDF 对象

        确保传入的是有效的 IDF 对象。

        Args:
            cls: 类本身（自动传入）
            v: IDF 对象

        Returns:
            Any: 验证后的 IDF 对象

        Raises:
            ValueError: 如果不是有效的 IDF 对象
        """
        # 检查是否是 IDF 实例（简单检查）
        if not hasattr(v, 'idfobjects'):
            raise ValueError(
                f"Invalid IDF object: must have 'idfobjects' attribute. "
                f"Got type: {type(v)}"
            )
        return v

    def get_output_path(self, filename: str) -> Path:
        """
        获取输出文件的完整路径

        Args:
            filename: 文件名

        Returns:
            Path: 完整的文件路径
        """
        return self.working_directory / filename

    def get_idf_path(self) -> Path:
        """
        获取 IDF 文件的保存路径

        Returns:
            Path: IDF 文件路径
        """
        return self.get_output_path(f"{self.job.output_prefix}.idf")


class BaselineContext(SimulationContext):
    """
    基准模拟上下文

    基准模拟的特定上下文。目前与基类相同，但预留扩展空间。

    Example:
        >>> context = BaselineContext(
        ...     job=simulation_job,
        ...     idf=idf_object,
        ...     working_directory=Path("output/baseline"),
        ... )
        >>> result = baseline_service.run(context)
    """
    pass


class PVContext(SimulationContext):
    """
    光伏模拟上下文

    包含光伏系统特定的配置参数。

    Attributes:
        pv_capacity: 光伏系统容量（kW）
        panel_efficiency: 光伏板效率（0-1之间）
        inverter_efficiency: 逆变器效率（0-1之间）
        tilt_angle: 倾斜角度（度，0-90）
        azimuth: 方位角（度，0-360，0=北，90=东，180=南）
        min_irradiance_threshold: 最小辐照度阈值（W/m²）

    Example:
        >>> context = PVContext(
        ...     job=simulation_job,
        ...     idf=idf_object,
        ...     working_directory=Path("output/pv"),
        ...     pv_capacity=100.0,
        ...     panel_efficiency=0.20,
        ...     inverter_efficiency=0.96,
        ...     tilt_angle=30.0,
        ...     azimuth=180.0,  # 南向
        ... )
        >>> result = pv_service.run(context)
    """

    pv_capacity: float = Field(
        ...,
        gt=0.0,
        description="光伏系统容量（kW），必须为正数"
    )
    panel_efficiency: float = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description="光伏板效率（0-1之间），默认20%"
    )
    inverter_efficiency: float = Field(
        default=0.96,
        gt=0.0,
        le=1.0,
        description="逆变器效率（0-1之间），默认96%"
    )
    tilt_angle: float = Field(
        default=30.0,
        ge=0.0,
        le=90.0,
        description="倾斜角度（度，0-90），默认30度"
    )
    azimuth: float = Field(
        default=180.0,
        ge=0.0,
        lt=360.0,
        description="方位角（度，0-360），0=北，90=东，180=南，默认180（南向）"
    )
    min_irradiance_threshold: float = Field(
        default=800.0,
        ge=0.0,
        description="最小辐照度阈值（W/m²），用于筛选合适的安装表面"
    )

    @model_validator(mode='after')
    def validate_pv_configuration(self) -> 'PVContext':
        """
        验证光伏配置的整体合理性

        可以添加跨字段的验证逻辑。

        注意：
            mode='after' 的 model_validator 使用 self（实例），
            不需要 @classmethod 装饰器。

        Returns:
            PVContext: 验证后的上下文
        """
        # 示例：可以添加更复杂的验证逻辑
        # 例如：根据建筑位置验证方位角是否合理
        return self


class ECMContext(SimulationContext):
    """
    能效措施（ECM）模拟上下文

    包含 ECM 参数的模拟上下文。ECM 参数从 job.ecm_parameters 获取。

    Example:
        >>> from backend.domain.value_objects import ECMParameters
        >>> ecm_params = ECMParameters(
        ...     window_u_value=1.5,
        ...     window_shgc=0.4,
        ...     cooling_cop=4.0,
        ... )
        >>> job.ecm_parameters = ecm_params
        >>>
        >>> context = ECMContext(
        ...     job=simulation_job,
        ...     idf=idf_object,
        ...     working_directory=Path("output/ecm"),
        ... )
        >>> result = ecm_service.run(context)
    """

    @model_validator(mode='after')
    def validate_ecm_parameters_exist(self) -> 'ECMContext':
        """
        验证 ECM 参数存在

        ECM 模拟必须提供 ECM 参数。

        注意：
            mode='after' 的 model_validator 使用 self（实例），
            可以访问所有已验证的字段。

        Returns:
            ECMContext: 验证后的上下文

        Raises:
            ValueError: 如果 job 中没有 ECM 参数
        """
        if self.job.ecm_parameters is None:
            raise ValueError(
                "ECM simulation requires ecm_parameters in the job. "
                "Please set job.ecm_parameters before creating ECMContext."
            )
        return self
```

---

### 🎓 **Pydantic V2 验证器装饰器总结**

#### ✅ **正确用法**

```python
from pydantic import BaseModel, field_validator, model_validator

class MyModel(BaseModel):
    value: int

    # ✅ field_validator - 不需要 @classmethod
    @field_validator('value')
    def check_value(cls, v: int) -> int:
        """Pydantic 会自动将其转换为 classmethod"""
        return v

    # ✅ model_validator (mode='before') - 不需要 @classmethod
    @model_validator(mode='before')
    def validate_before(cls, values: dict) -> dict:
        """Pydantic 会自动将其转换为 classmethod"""
        return values

    # ✅ model_validator (mode='after') - 使用 self
    @model_validator(mode='after')
    def validate_after(self) -> 'MyModel':
        """mode='after' 时使用实例方法"""
        return self
```

#### ❌ **错误用法**

```python
# ❌ 不要显式添加 @classmethod（虽然也能工作，但是冗余）
@field_validator('value')
@classmethod  # 不需要！
def check_value(cls, v: int) -> int:
    return v

# ❌ 装饰器顺序错误
@classmethod
@field_validator('value')  # 这样会导致验证器不工作！
def check_value(cls, v: int) -> int:
    return v
```

#### 📋 **快速参考表**

| 验证器类型 | 装饰器 | 第一个参数 | 需要 @classmethod? |
|-----------|--------|-----------|-------------------|
| `@field_validator` | 单字段验证 | `cls` | ❌ 不需要（自动） |
| `@model_validator(mode='before')` | 模型验证（前） | `cls` | ❌ 不需要（自动） |
| `@model_validator(mode='after')` | 模型验证（后） | `self` | ❌ 不需要 |

感谢你的纠正！这是 Pydantic V2 的一个重要改进，让代码更简洁。🎉
```

### BaseSimulationService

```python
"""
基础模拟服务

所有模拟服务的抽象基类。
"""

import time
from abc import abstractmethod
from typing import Generic

from loguru import logger

from backend.services.interfaces import (
    ISimulationService,
    IEnergyPlusExecutor,
    IResultParser,
)
from backend.services.simulation.simulation_context import SimulationContext
from backend.domain.models import SimulationResult
from backend.utils.exceptions import SimulationError


TContext = TypeVar('TContext', bound=SimulationContext)


class BaseSimulationService(ISimulationService[TContext], Generic[TContext]):
    """
    基础模拟服务

    提供模拟服务的通用逻辑和模板方法。

    Type Parameters:
        TContext: 具体的上下文类型

    Attributes:
        _executor: EnergyPlus 执行器
        _parser: 结果解析器
        _logger: 日志记录器
    """

    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        parser: IResultParser,
    ):
        """
        初始化基础服务

        Args:
            executor: EnergyPlus 执行器
            parser: 结果解析器
        """
        self._executor = executor
        self._parser = parser
        self._logger = logger

    def prepare(self, context: TContext) -> None:
        """
        准备模拟环境（通用逻辑）

        Args:
            context: 模拟上下文
        """
        self._logger.info(f"Preparing simulation for job {context.job.id}")

        # 创建输出目录
        context.job.output_directory.mkdir(parents=True, exist_ok=True)

        # 验证文件
        self._validate_files(context)

        # 添加输出变量
        self._add_output_variables(context.idf)

        # 子类特定的准备工作
        self._prepare_specific(context)

        self._logger.info("Preparation completed")

    @abstractmethod
    def _prepare_specific(self, context: TContext) -> None:
        """
        子类特定的准备逻辑

        子类覆盖此方法以实现特定的准备工作。

        Args:
            context: 模拟上下文
        """
        pass

    def execute(self, context: TContext) -> SimulationResult:
        """
        执行模拟（通用逻辑）

        Args:
            context: 模拟上下文

        Returns:
            SimulationResult: 模拟结果
        """
        self._logger.info(f"Executing simulation for job {context.job.id}")

        start_time = time.time()

        try:
            # 应用子类特定的 IDF 修改
            self._modify_idf(context)

            # 保存 IDF 到工作目录
            idf_path = context.working_directory / f"{context.job.output_prefix}.idf"
            context.idf.saveas(str(idf_path))

            # 执行 EnergyPlus
            execution_result = self._executor.run(
                idf=context.idf,
                weather_file=context.job.weather_file.file_path,
                output_directory=context.job.output_directory,
                output_prefix=context.job.output_prefix,
                read_variables=context.job.read_variables,
            )

            # 解析结果
            result = self._parser.parse(
                job_id=context.job.id,
                output_directory=context.job.output_directory,
                output_prefix=context.job.output_prefix,
            )

            # 设置执行信息
            result.execution_time = time.time() - start_time
            result.success = execution_result.success

            if not execution_result.success:
                for error in execution_result.errors:
                    result.add_error(error)

            self._logger.info(
                f"Simulation completed in {result.execution_time:.2f}s. "
                f"Source EUI: {result.source_eui} kWh/m²/yr"
            )

            return result

        except Exception as e:
            self._logger.error(f"Simulation failed: {e}")
            result = SimulationResult(
                job_id=context.job.id,
                output_directory=context.job.output_directory,
            )
            result.add_error(str(e))
            result.execution_time = time.time() - start_time
            return result

    @abstractmethod
    def _modify_idf(self, context: TContext) -> None:
        """
        修改 IDF 文件

        子类覆盖此方法以实现特定的 IDF 修改逻辑。

        Args:
            context: 模拟上下文
        """
        pass

    def cleanup(self, context: TContext) -> None:
        """
        清理临时文件（通用逻辑）

        Args:
            context: 模拟上下文
        """
        self._logger.info("Cleaning up temporary files")

        # 清理临时文件
        extensions_to_keep = ['.csv', '.sql', '.idf', '.epw']
        for file in context.job.output_directory.glob("*"):
            if file.suffix not in extensions_to_keep:
                try:
                    file.unlink()
                except Exception as e:
                    self._logger.warning(f"Failed to delete {file}: {e}")

    def _validate_files(self, context: TContext) -> None:
        """验证必需文件存在"""
        if not context.job.building.idf_file_path.exists():
            raise FileNotFoundError(
                f"IDF file not found: {context.job.building.idf_file_path}"
            )

        if not context.job.weather_file.file_path.exists():
            raise FileNotFoundError(
                f"Weather file not found: {context.job.weather_file.file_path}"
            )

    def _add_output_variables(self, idf: IDF) -> None:
        """添加必要的输出变量"""
        required_variables = [
            "Site Outdoor Air Drybulb Temperature",
            "Zone Mean Air Temperature",
            "Facility Total Electric Demand Power",
            "Facility Total Natural Gas Demand Rate",
        ]

        for var_name in required_variables:
            # 检查是否已存在
            try:
                ov_list = idf.idfobjects["OUTPUT:VARIABLE"]
            except Exception:
                ov_list = []
            exists = any(ov.Variable_Name == var_name for ov in ov_list)

            if not exists:
                idf.newidfobject(
                    "OUTPUT:VARIABLE",
                    Key_Value="*",
                    Variable_Name=var_name,
                    Reporting_Frequency="Hourly",
                )
```

---

## 模拟服务

### BaselineSimulationService

```python
"""
基准建筑模拟服务

执行不带任何能效措施的基准建筑模拟。
"""

from backend.services.simulation.base_simulation_service import BaseSimulationService
from backend.services.simulation.simulation_context import BaselineContext


class BaselineSimulationService(BaseSimulationService[BaselineContext]):
    """
    基准模拟服务

    执行建筑的基准能耗模拟，不应用任何 ECM 措施。

    Example:
        >>> executor = EnergyPlusExecutor(...)
        >>> parser = ResultParser(...)
        >>> service = BaselineSimulationService(executor, parser)
        >>>
        >>> context = BaselineContext(
        ...     job=simulation_job,
        ...     idf=idf_object,
        ...     working_directory=Path("output/baseline"),
        ... )
        >>>
        >>> result = service.run(context)
        >>> print(f"Source EUI: {result.source_eui} kWh/m²/yr")
    """

    def _prepare_specific(self, context: BaselineContext) -> None:
        """
        基准模拟的特定准备工作

        基准模拟通常不需要特殊准备，此方法为空。

        Args:
            context: 基准模拟上下文
        """
        # 基准模拟不需要特殊准备
        pass

    def _modify_idf(self, context: BaselineContext) -> None:
        """
        修改 IDF 文件

        基准模拟不修改 IDF，保持原样。

        Args:
            context: 基准模拟上下文
        """
        # 基准模拟不修改 IDF
        pass
```

### PVSimulationService

```python
"""
光伏系统模拟服务
"""

from backend.services.simulation.base_simulation_service import BaseSimulationService
from backend.services.simulation.simulation_context import PVContext
from backend.domain.services import IPVSystemDesigner


class PVSimulationService(BaseSimulationService[PVContext]):
    """
    光伏系统模拟服务

    在建筑上添加光伏系统并执行模拟。

    Attributes:
        _pv_designer: 光伏系统设计器（领域服务）
    """

    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        parser: IResultParser,
        pv_designer: IPVSystemDesigner,
    ):
        """
        初始化 PV 服务

        Args:
            executor: EnergyPlus 执行器
            parser: 结果解析器
            pv_designer: 光伏系统设计器
        """
        super().__init__(executor, parser)
        self._pv_designer = pv_designer

    def _prepare_specific(self, context: PVContext) -> None:
        """
        PV 模拟的准备工作

        验证 PV 参数的合理性。

        Args:
            context: PV 模拟上下文
        """
        self._logger.info(
            f"Preparing PV simulation: capacity={context.pv_capacity}kW, "
            f"efficiency={context.pv_efficiency}"
        )

        # 可以在这里添加更多验证逻辑
        # 例如：检查屋顶面积是否足够安装指定容量的光伏板

    def _modify_idf(self, context: PVContext) -> None:
        """
        添加光伏系统到 IDF

        使用领域服务 PVSystemDesigner 生成光伏对象。

        Args:
            context: PV 模拟上下文
        """
        self._logger.info("Adding PV system to IDF")

        # 使用领域服务设计光伏系统
        pv_objects = self._pv_designer.design_pv_system(
            idf=context.idf,
            capacity_kw=context.pv_capacity,
            efficiency=context.pv_efficiency,
            inverter_efficiency=context.inverter_efficiency,
            tilt=context.tilt_angle,
            azimuth=context.azimuth,
        )

        # 光伏对象已经添加到 IDF 中
        self._logger.info(f"Added {len(pv_objects)} PV-related objects to IDF")
```

### ECMSimulationService

```python
"""
能效措施（ECM）模拟服务
"""

from backend.services.simulation.base_simulation_service import BaseSimulationService
from backend.services.simulation.simulation_context import SimulationContext
from backend.domain.value_objects import ECMParameters
from backend.domain.services import IECMApplicator


class ECMSimulationService(BaseSimulationService[SimulationContext]):
    """
    能效措施模拟服务

    应用 ECM 参数到建筑并执行模拟。

    Attributes:
        _ecm_applicator: ECM 应用器（领域服务）
    """

    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        parser: IResultParser,
        ecm_applicator: IECMApplicator,
    ):
        """
        初始化 ECM 服务

        Args:
            executor: EnergyPlus 执行器
            parser: 结果解析器
            ecm_applicator: ECM 应用器
        """
        super().__init__(executor, parser)
        self._ecm_applicator = ecm_applicator

    def _prepare_specific(self, context: SimulationContext) -> None:
        """
        ECM 模拟的准备工作

        验证 ECM 参数存在。

        Args:
            context: 模拟上下文
        """
        if context.job.ecm_parameters is None:
            raise ValueError("ECM parameters are required for ECM simulation")

        self._logger.info(
            f"Preparing ECM simulation with parameters: "
            f"{context.job.ecm_parameters.to_dict()}"
        )

    def _modify_idf(self, context: SimulationContext) -> None:
        """
        应用 ECM 参数到 IDF

        使用领域服务 ECMApplicator 修改 IDF。

        Args:
            context: 模拟上下文
        """
        self._logger.info("Applying ECM parameters to IDF")

        if context.job.ecm_parameters is None:
            return  # 已在 prepare 中验证，这里作为保护

        # 使用领域服务应用 ECM
        self._ecm_applicator.apply(
            idf=context.idf,
            parameters=context.job.ecm_parameters,
        )

        self._logger.info("ECM parameters applied successfully")
```

---

## 优化服务

### 优化策略接口

```python
"""
优化策略接口
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Tuple, Optional, List

from backend.domain.value_objects import ECMParameters


class IOptimizationStrategy(ABC):
    """
    优化策略接口

    定义优化算法的统一接口。
    """

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameter_bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        constraints: Optional[List[Dict]] = None,
    ) -> ECMParameters:
        """
        执行优化

        Args:
            objective_function: 目标函数，接收参数字典，返回目标值
            parameter_bounds: 参数边界 {参数名: (最小值, 最大值)}
            max_iterations: 最大迭代次数
            constraints: 约束条件列表（可选）

        Returns:
            ECMParameters: 最优参数

        Example:
            >>> def objective(params: dict) -> float:
            ...     # 计算 EUI
            ...     return eui
            >>>
            >>> bounds = {
            ...     'window_u_value': (0.5, 5.0),
            ...     'cooling_cop': (2.0, 6.0),
            ... }
            >>>
            >>> optimal_params = strategy.optimize(objective, bounds, max_iterations=50)
        """
        pass
```

### 遗传算法策略

```python
"""
遗传算法优化策略

使用 DEAP 库实现遗传算法。
"""

from typing import Callable, Dict, List, Optional, Tuple

from deap import base, creator, tools, algorithms
import numpy as np

from backend.services.optimization.strategies import IOptimizationStrategy
from backend.domain.value_objects import ECMParameters


class GeneticAlgorithmStrategy(IOptimizationStrategy):
    """
    遗传算法优化策略

    使用 DEAP 库实现的遗传算法（GA）。

    Attributes:
        population_size: 种群大小
        crossover_prob: 交叉概率
        mutation_prob: 变异概率
        tournament_size: 锦标赛选择的参与者数量
    """

    def __init__(
        self,
        population_size: int = 50,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
    ):
        """
        初始化遗传算法策略

        Args:
            population_size: 种群大小
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            tournament_size: 锦标赛大小
        """
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size

    def optimize(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameter_bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        constraints: Optional[List[Dict]] = None,
    ) -> ECMParameters:
        """
        使用遗传算法执行优化

        Args:
            objective_function: 目标函数（最小化）
            parameter_bounds: 参数边界
            max_iterations: 最大代数
            constraints: 约束条件（暂不支持）

        Returns:
            ECMParameters: 最优参数
        """
        # 提取参数名和边界
        param_names = list(parameter_bounds.keys())
        bounds_list = [parameter_bounds[name] for name in param_names]

        # 创建 DEAP 类型
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # 注册生成器
        for i, (low, high) in enumerate(bounds_list):
            toolbox.register(f"attr_{i}", np.random.uniform, low, high)

        # 创建个体和种群
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            [getattr(toolbox, f"attr_{i}") for i in range(len(bounds_list))],
            n=1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # 评估函数
        def evaluate(individual):
            params_dict = {name: value for name, value in zip(param_names, individual)}
            try:
                fitness = objective_function(params_dict)
                return (fitness,)
            except Exception:
                return (float('inf'),)  # 惩罚无效解

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=0.1,
            indpb=0.1,
        )
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        # 创建初始种群
        population = toolbox.population(n=self.population_size)

        # 运行遗传算法
        algorithms.eaSimple(
            population,
            toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=max_iterations,
            verbose=False,
        )

        # 获取最优个体
        best_individual = tools.selBest(population, k=1)[0]
        best_params = {name: value for name, value in zip(param_names, best_individual)}

        return ECMParameters(**best_params)
```

### OptimizationService

```python
"""
优化服务
"""

from typing import Dict, Tuple
from uuid import uuid4
from pathlib import Path

from loguru import logger

from backend.services.optimization.strategies import IOptimizationStrategy
from backend.services.orchestration import SimulationOrchestrator
from backend.domain.models import Building, WeatherFile, SimulationJob
from backend.domain.value_objects import ECMParameters


class OptimizationService:
    """
    优化服务

    使用优化算法找到最优的 ECM 参数组合。

    Attributes:
        _strategy: 优化策略
        _orchestrator: 模拟编排器
    """

    def __init__(
        self,
        strategy: IOptimizationStrategy,
        orchestrator: SimulationOrchestrator,
    ):
        """
        初始化优化服务

        Args:
            strategy: 优化策略（可以动态切换）
            orchestrator: 模拟编排器
        """
        self._strategy = strategy
        self._orchestrator = orchestrator
        self._logger = logger

    def set_strategy(self, strategy: IOptimizationStrategy) -> None:
        """
        动态切换优化策略

        Args:
            strategy: 新的优化策略
        """
        self._strategy = strategy
        self._logger.info(f"Switched optimization strategy to {type(strategy).__name__}")

    def find_optimal_ecm(
        self,
        building: Building,
        weather_file: WeatherFile,
        parameter_bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
    ) -> Tuple[ECMParameters, float]:
        """
        找到最优 ECM 参数

        Args:
            building: 建筑对象
            weather_file: 天气文件对象
            parameter_bounds: 参数边界
            max_iterations: 最大迭代次数

        Returns:
            (最优参数, 最优 EUI)

        Example:
            >>> optimization_service = OptimizationService(
            ...     strategy=GeneticAlgorithmStrategy(),
            ...     orchestrator=orchestrator,
            ... )
            >>>
            >>> bounds = {
            ...     'window_u_value': (1.0, 3.0),
            ...     'cooling_cop': (3.0, 5.0),
            ...     'lighting_reduction_factor': (0.1, 0.5),
            ... }
            >>>
            >>> optimal_params, optimal_eui = optimization_service.find_optimal_ecm(
            ...     building=building,
            ...     weather_file=weather,
            ...     parameter_bounds=bounds,
            ...     max_iterations=50,
            ... )
            >>> print(f"Optimal EUI: {optimal_eui} kWh/m²/yr")
        """
        self._logger.info(
            f"Starting optimization for {building.name} with "
            f"{type(self._strategy).__name__}"
        )

        # 定义目标函数
        def objective_function(params: Dict[str, float]) -> float:
            """
            目标函数：最小化 EUI

            Args:
                params: ECM 参数字典

            Returns:
                Source EUI (kWh/m²/yr)
            """
            # 创建 ECM 参数对象
            ecm_params = ECMParameters(**params)

            # 创建模拟任务
            job = SimulationJob(
                building=building,
                weather_file=weather_file,
                simulation_type="ecm",
                output_directory=Path("temp/optimization") / str(uuid4()),
                output_prefix="opt",
                ecm_parameters=ecm_params,
            )

            # 执行模拟
            results = self._orchestrator.execute_batch([job], use_cache=True)
            result = results[0]

            if not result.success or result.source_eui is None:
                return float('inf')  # 惩罚失败的模拟

            return result.source_eui

        # 执行优化
        optimal_params = self._strategy.optimize(
            objective_function=objective_function,
            parameter_bounds=parameter_bounds,
            max_iterations=max_iterations,
        )

        # 计算最优 EUI（运行一次最优参数的模拟）
        optimal_eui = objective_function(optimal_params.to_dict())

        self._logger.info(
            f"Optimization completed. Optimal EUI: {optimal_eui} kWh/m²/yr"
        )

        return optimal_params, optimal_eui
```

---

## 分析服务

### 敏感性分析服务

敏感性分析服务用于评估不同参数对建筑能耗的影响程度。

```python
"""
敏感性分析服务

评估ECM参数对能耗的敏感性。
"""

from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

from backend.domain.models import Building, WeatherFile, SimulationJob
from backend.domain.value_objects import ECMParameters
from backend.services.orchestration import SimulationOrchestrator


class SensitivityAnalysisService:
    """
    敏感性分析服务

    使用单因素分析法(One-at-a-time, OAT)评估参数敏感性。

    Attributes:
        _orchestrator: 模拟编排器
        _logger: 日志记录器
    """

    def __init__(self, orchestrator: SimulationOrchestrator):
        """
        初始化敏感性分析服务

        Args:
            orchestrator: 模拟编排器
        """
        self._orchestrator = orchestrator
        self._logger = logger

    def analyze_parameter_sensitivity(
        self,
        building: Building,
        weather_file: WeatherFile,
        parameter_ranges: Dict[str, Tuple[float, float, int]],
        baseline_params: ECMParameters,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        分析参数敏感性

        对每个参数在指定范围内进行采样,保持其他参数不变,
        评估该参数对能耗的影响。

        Args:
            building: 建筑对象
            weather_file: 天气文件
            parameter_ranges: 参数范围 {参数名: (最小值, 最大值, 采样点数)}
            baseline_params: 基准参数

        Returns:
            Dict[str, List[Tuple[float, float]]]:
                {参数名: [(参数值, EUI值), ...]}

        Example:
            >>> sensitivity_service = SensitivityAnalysisService(orchestrator)
            >>>
            >>> baseline = ECMParameters(
            ...     window_u_value=2.0,
            ...     cooling_cop=3.5,
            ...     lighting_reduction_factor=0.2,
            ... )
            >>>
            >>> ranges = {
            ...     'window_u_value': (1.0, 3.0, 5),  # 5个采样点
            ...     'cooling_cop': (3.0, 5.0, 5),
            ...     'lighting_reduction_factor': (0.1, 0.4, 4),
            ... }
            >>>
            >>> results = sensitivity_service.analyze_parameter_sensitivity(
            ...     building=building,
            ...     weather_file=weather,
            ...     parameter_ranges=ranges,
            ...     baseline_params=baseline,
            ... )
            >>>
            >>> # 绘制敏感性曲线
            >>> for param_name, data_points in results.items():
            ...     param_values = [p[0] for p in data_points]
            ...     eui_values = [p[1] for p in data_points]
            ...     plt.plot(param_values, eui_values, label=param_name)
        """
        self._logger.info(
            f"Starting sensitivity analysis for {len(parameter_ranges)} parameters"
        )

        results: Dict[str, List[Tuple[float, float]]] = {}

        for param_name, (min_val, max_val, num_samples) in parameter_ranges.items():
            self._logger.info(
                f"Analyzing parameter: {param_name} "
                f"[{min_val}, {max_val}] with {num_samples} samples"
            )

            # 生成采样点
            param_values = np.linspace(min_val, max_val, num_samples)

            # 创建模拟任务
            jobs = []
            for value in param_values:
                # 复制基准参数
                params_dict = baseline_params.to_dict()
                # 修改当前参数
                params_dict[param_name] = value
                params = ECMParameters(**params_dict)

                job = SimulationJob(
                    building=building,
                    weather_file=weather_file,
                    simulation_type="ecm",
                    output_directory=Path(f"temp/sensitivity/{param_name}_{value}"),
                    output_prefix=f"sens_{param_name}",
                    ecm_parameters=params,
                )
                jobs.append(job)

            # 批量执行模拟
            simulation_results = self._orchestrator.execute_batch(
                jobs, use_cache=True
            )

            # 提取结果
            data_points = []
            for value, result in zip(param_values, simulation_results):
                if result.success and result.source_eui is not None:
                    data_points.append((float(value), result.source_eui))
                else:
                    self._logger.warning(
                        f"Simulation failed for {param_name}={value}"
                    )

            results[param_name] = data_points

            self._logger.info(
                f"Completed analysis for {param_name}: "
                f"{len(data_points)}/{num_samples} successful"
            )

        return results

    def calculate_sensitivity_indices(
        self,
        sensitivity_results: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, float]:
        """
        计算敏感性指数

        使用标准化的敏感性指数(Normalized Sensitivity Index):
        SI = (ΔY / Y_baseline) / (ΔX / X_baseline)

        Args:
            sensitivity_results: 敏感性分析结果

        Returns:
            Dict[str, float]: {参数名: 敏感性指数}

        Example:
            >>> indices = sensitivity_service.calculate_sensitivity_indices(results)
            >>> # 按敏感性排序
            >>> sorted_params = sorted(
            ...     indices.items(),
            ...     key=lambda x: abs(x[1]),
            ...     reverse=True
            ... )
            >>> print("Most sensitive parameters:")
            >>> for param, index in sorted_params[:3]:
            ...     print(f"  {param}: {index:.3f}")
        """
        indices = {}

        for param_name, data_points in sensitivity_results.items():
            if len(data_points) < 2:
                self._logger.warning(
                    f"Insufficient data points for {param_name}, skipping"
                )
                continue

            # 提取参数值和EUI值
            param_values = np.array([p[0] for p in data_points])
            eui_values = np.array([p[1] for p in data_points])

            # 计算基准值(中间点)
            mid_idx = len(data_points) // 2
            x_baseline = param_values[mid_idx]
            y_baseline = eui_values[mid_idx]

            # 计算变化量
            delta_x = param_values[-1] - param_values[0]
            delta_y = eui_values[-1] - eui_values[0]

            # 计算标准化敏感性指数
            if x_baseline != 0 and y_baseline != 0 and delta_x != 0:
                si = (delta_y / y_baseline) / (delta_x / x_baseline)
                indices[param_name] = si
            else:
                self._logger.warning(
                    f"Cannot calculate sensitivity index for {param_name}"
                )

        return indices
```

### 数据分析服务

```python
"""
数据分析服务

提供模拟结果的统计分析和可视化支持。
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from backend.domain.models import SimulationResult


class DataAnalysisService:
    """
    数据分析服务

    提供模拟结果的统计分析、比较和导出功能。

    Attributes:
        _logger: 日志记录器
    """

    def __init__(self):
        """初始化数据分析服务"""
        self._logger = logger

    def compare_results(
        self,
        baseline_result: SimulationResult,
        ecm_results: List[SimulationResult],
    ) -> pd.DataFrame:
        """
        比较基准和ECM结果

        Args:
            baseline_result: 基准模拟结果
            ecm_results: ECM模拟结果列表

        Returns:
            pd.DataFrame: 比较结果表格

        Example:
            >>> analysis_service = DataAnalysisService()
            >>>
            >>> comparison = analysis_service.compare_results(
            ...     baseline_result=baseline,
            ...     ecm_results=[ecm1, ecm2, ecm3],
            ... )
            >>>
            >>> print(comparison)
            >>> # 输出:
            >>> #   Scenario    Source_EUI  Site_EUI  Savings(%)  Cost_Savings($)
            >>> #   Baseline    150.0       140.0     0.0         0.0
            >>> #   ECM_1       120.0       112.0     20.0        5000.0
            >>> #   ECM_2       110.0       103.0     26.7        6500.0
        """
        data = []

        # 添加基准数据
        data.append({
            'Scenario': 'Baseline',
            'Source_EUI': baseline_result.source_eui,
            'Site_EUI': baseline_result.site_eui,
            'Savings_Percent': 0.0,
            'Absolute_Savings': 0.0,
        })

        # 添加ECM数据
        for i, ecm_result in enumerate(ecm_results, 1):
            if ecm_result.success and ecm_result.source_eui is not None:
                savings = baseline_result.source_eui - ecm_result.source_eui
                savings_pct = (savings / baseline_result.source_eui) * 100

                data.append({
                    'Scenario': f'ECM_{i}',
                    'Source_EUI': ecm_result.source_eui,
                    'Site_EUI': ecm_result.site_eui,
                    'Savings_Percent': savings_pct,
                    'Absolute_Savings': savings,
                })

        df = pd.DataFrame(data)
        return df

    def calculate_statistics(
        self,
        results: List[SimulationResult],
    ) -> Dict[str, float]:
        """
        计算结果统计信息

        Args:
            results: 模拟结果列表

        Returns:
            Dict[str, float]: 统计信息

        Example:
            >>> stats = analysis_service.calculate_statistics(results)
            >>> print(f"Mean EUI: {stats['mean_eui']:.2f}")
            >>> print(f"Std Dev: {stats['std_eui']:.2f}")
        """
        eui_values = [
            r.source_eui for r in results
            if r.success and r.source_eui is not None
        ]

        if not eui_values:
            return {}

        return {
            'mean_eui': np.mean(eui_values),
            'median_eui': np.median(eui_values),
            'std_eui': np.std(eui_values),
            'min_eui': np.min(eui_values),
            'max_eui': np.max(eui_values),
            'count': len(eui_values),
        }

    def export_to_csv(
        self,
        results: List[SimulationResult],
        output_path: Path,
    ) -> None:
        """
        导出结果到CSV

        Args:
            results: 模拟结果列表
            output_path: 输出文件路径

        Example:
            >>> analysis_service.export_to_csv(
            ...     results=all_results,
            ...     output_path=Path("output/results.csv"),
            ... )
        """
        data = []
        for result in results:
            if result.success:
                data.append({
                    'Job_ID': str(result.job_id),
                    'Source_EUI': result.source_eui,
                    'Site_EUI': result.site_eui,
                    'Execution_Time': result.execution_time,
                    'Success': result.success,
                })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        self._logger.info(f"Exported {len(data)} results to {output_path}")
```

---

## 服务编排

### SimulationOrchestrator

```python
"""
模拟编排器

负责批量模拟任务的调度、执行和监控。
"""

from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from backend.domain.models import SimulationJob, SimulationResult
from backend.services.interfaces import ICacheService
from backend.factories import ServiceFactory


class SimulationOrchestrator:
    """
    模拟编排器

    负责：
    - 批量模拟任务的调度
    - 并行执行管理
    - 缓存集成
    - 进度跟踪
    - 错误处理

    Attributes:
        _service_factory: 服务工厂
        _cache: 缓存服务
        _max_workers: 最大并行工作数
    """

    def __init__(
        self,
        service_factory: ServiceFactory,
        cache_service: ICacheService,
        max_workers: int = 4,
    ):
        """
        初始化编排器

        Args:
            service_factory: 服务工厂
            cache_service: 缓存服务
            max_workers: 最大并行工作数
        """
        self._service_factory = service_factory
        self._cache = cache_service
        self._max_workers = max_workers
        self._logger = logger

    def execute_batch(
        self,
        jobs: List[SimulationJob],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_cache: bool = True,
    ) -> List[SimulationResult]:
        """
        批量执行模拟任务

        Args:
            jobs: 模拟任务列表
            progress_callback: 进度回调函数 (completed, total)
            use_cache: 是否使用缓存

        Returns:
            模拟结果列表

        Example:
            >>> orchestrator = SimulationOrchestrator(...)
            >>> jobs = [job1, job2, job3]
            >>>
            >>> def on_progress(completed, total):
            ...     print(f"Progress: {completed}/{total}")
            >>>
            >>> results = orchestrator.execute_batch(
            ...     jobs,
            ...     progress_callback=on_progress,
            ...     use_cache=True,
            ... )
        """
        self._logger.info(f"Starting batch execution of {len(jobs)} jobs")

        results: List[SimulationResult] = []
        completed = 0
        total = len(jobs)

        # 检查缓存
        jobs_to_run = []
        for job in jobs:
            if use_cache:
                cache_key = job.get_cache_key()
                cached_result = self._cache.get(cache_key)
                if cached_result:
                    self._logger.info(f"Using cached result for job {job.id}")
                    results.append(cached_result)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                    continue

            jobs_to_run.append(job)

        # 并行执行剩余任务
        if jobs_to_run:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_job = {
                    executor.submit(self._execute_single_job, job): job
                    for job in jobs_to_run
                }

                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # 缓存成功的结果
                        if use_cache and result.success:
                            cache_key = job.get_cache_key()
                            self._cache.set(cache_key, result)

                    except Exception as e:
                        self._logger.error(f"Job {job.id} failed: {e}")
                        # 创建失败结果
                        result = SimulationResult(
                            job_id=job.id,
                            output_directory=job.output_directory,
                        )
                        result.add_error(str(e))
                        results.append(result)

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)

        success_count = sum(1 for r in results if r.success)
        self._logger.info(
            f"Batch execution completed. Success: {success_count}/{total}"
        )
        return results

    def _execute_single_job(self, job: SimulationJob) -> SimulationResult:
        """
        执行单个模拟任务

        Args:
            job: 模拟任务

        Returns:
            SimulationResult: 模拟结果
        """
        self._logger.info(f"Executing job {job.id}: {job.simulation_type}")

        # 根据任务类型创建服务
        service = self._service_factory.create_service(job.simulation_type)

        # 创建上下文
        context = self._create_context(job)

        # 执行模拟
        job.start()
        try:
            result = service.run(context)
            job.complete(result)
            return result
        except Exception as e:
            job.fail(str(e))
            raise

    def _create_context(self, job: SimulationJob):
        """
        创建模拟上下文

        Args:
            job: 模拟任务

        Returns:
            适当的上下文对象
        """
        from eppy.modeleditor import IDF
        from backend.services.simulation.simulation_context import (
            BaselineContext,
            PVContext,
        )

        # 加载 IDF
        idf = IDF(str(job.building.idf_file_path))

        # 根据模拟类型创建上下文
        if job.simulation_type == "pv":
            # PV 模拟需要 PV 参数（从 job.metadata 中获取）
            pv_capacity = job.building.get_metadata("pv_capacity", 100.0)
            return PVContext(
                job=job,
                idf=idf,
                working_directory=job.output_directory,
                pv_capacity=pv_capacity,
            )
        else:
            # 基准或 ECM 模拟使用基础上下文
            return BaselineContext(
                job=job,
                idf=idf,
                working_directory=job.output_directory,
            )
```

---

## 使用示例

### 完整的模拟流程

```python
"""
完整的模拟流程示例
"""

from pathlib import Path

from backend.utils.config import ConfigManager, setup_container
from backend.domain.models import Building, BuildingType, WeatherFile
from backend.domain.models import SimulationJob, SimulationType
from backend.domain.value_objects import ECMParameters
from backend.factories import ServiceFactory, BuildingFactory
from backend.services.orchestration import SimulationOrchestrator


def main():
    """主流程"""

    # 1. 初始化配置和依赖容器
    config = ConfigManager()
    container = setup_container(config)

    # 2. 创建建筑对象
    building_factory = container.resolve(BuildingFactory)
    building = building_factory.create_from_idf(
        idf_path=Path("data/prototypes/Chicago_OfficeLarge.idf"),
        building_type=BuildingType.OFFICE_LARGE,
        location="Chicago",
    )

    # 3. 创建天气文件对象
    weather = WeatherFile(
        file_path=Path("data/tmys/Chicago_TMY.epw"),
        location="Chicago",
        scenario="TMY",
    )

    # 4. 场景 1：基准模拟
    baseline_job = SimulationJob(
        building=building,
        weather_file=weather,
        simulation_type=SimulationType.BASELINE,
        output_directory=Path("output/baseline"),
        output_prefix="baseline_chicago",
    )

    # 5. 场景 2：ECM 模拟
    ecm_params = ECMParameters(
        window_u_value=1.5,
        window_shgc=0.4,
        cooling_cop=4.0,
        lighting_reduction_factor=0.2,
    )

    ecm_job = SimulationJob(
        building=building,
        weather_file=weather,
        simulation_type=SimulationType.ECM,
        output_directory=Path("output/ecm"),
        output_prefix="ecm_chicago",
        ecm_parameters=ecm_params,
    )

    # 6. 批量执行
    orchestrator = container.resolve(SimulationOrchestrator)

    def on_progress(completed: int, total: int):
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    results = orchestrator.execute_batch(
        jobs=[baseline_job, ecm_job],
        progress_callback=on_progress,
        use_cache=True,
    )

    # 7. 分析结果
    for result in results:
        if result.success:
            print(f"\nJob {result.job_id}:")
            print(f"  Source EUI: {result.source_eui:.2f} kWh/m²/yr")
            print(f"  Site EUI: {result.site_eui:.2f} kWh/m²/yr")
            print(f"  Execution time: {result.execution_time:.2f}s")
        else:
            print(f"\nJob {result.job_id} failed:")
            for error in result.error_messages:
                print(f"  Error: {error}")

    # 8. 计算节能量
    if len(results) == 2 and all(r.success for r in results):
        baseline_eui = results[0].source_eui
        ecm_eui = results[1].source_eui
        savings = baseline_eui - ecm_eui
        savings_pct = (savings / baseline_eui) * 100

        print(f"\nEnergy Savings:")
        print(f"  Absolute: {savings:.2f} kWh/m²/yr")
        print(f"  Percentage: {savings_pct:.1f}%")


if __name__ == "__main__":
    main()
```

### 优化流程示例

```python
"""
优化流程示例
"""

from backend.services.optimization import OptimizationService, GeneticAlgorithmStrategy


def run_optimization():
    """运行优化"""

    # 初始化
    config = ConfigManager()
    container = setup_container(config)

    # 创建建筑和天气
    building = ...  # 同上
    weather = ...   # 同上

    # 创建优化服务
    ga_strategy = GeneticAlgorithmStrategy(population_size=50)

    orchestrator = container.resolve(SimulationOrchestrator)
    optimization_service = OptimizationService(
        strategy=ga_strategy,
        orchestrator=orchestrator,
    )

    # 定义参数边界
    parameter_bounds = {
        'window_u_value': (1.0, 3.0),
        'window_shgc': (0.3, 0.6),
        'cooling_cop': (3.0, 5.0),
        'heating_efficiency': (0.8, 0.95),
        'lighting_reduction_factor': (0.1, 0.4),
    }

    # 执行优化
    optimal_params, optimal_eui = optimization_service.find_optimal_ecm(
        building=building,
        weather_file=weather,
        parameter_bounds=parameter_bounds,
        max_iterations=100,
    )

    print(f"Optimal Parameters:")
    for key, value in optimal_params.to_dict().items():
        print(f"  {key}: {value:.3f}")
    print(f"\nOptimal EUI: {optimal_eui:.2f} kWh/m²/yr")


if __name__ == "__main__":
    run_optimization()
```

---

## 测试策略

### 单元测试

```python
"""
服务层单元测试
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from backend.services.simulation import BaselineSimulationService
from backend.services.simulation.simulation_context import BaselineContext
from backend.domain.models import Building, BuildingType, WeatherFile, SimulationJob


class TestBaselineSimulationService:
    """基准模拟服务测试"""

    @pytest.fixture
    def mock_executor(self):
        """模拟 EnergyPlus 执行器"""
        executor = Mock()
        executor.run.return_value = Mock(
            success=True,
            return_code=0,
            errors=[],
        )
        return executor

    @pytest.fixture
    def mock_parser(self):
        """模拟结果解析器"""
        parser = Mock()
        parser.parse.return_value = Mock(
            success=True,
            source_eui=150.0,
            site_eui=140.0,
        )
        return parser

    @pytest.fixture
    def baseline_service(self, mock_executor, mock_parser):
        """创建服务实例"""
        return BaselineSimulationService(
            executor=mock_executor,
            parser=mock_parser,
        )

    @pytest.fixture
    def simulation_context(self, tmp_path):
        """创建模拟上下文"""
        # 创建临时文件
        idf_file = tmp_path / "test.idf"
        idf_file.touch()
        epw_file = tmp_path / "test.epw"
        epw_file.touch()

        building = Building(
            name="Test Building",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=idf_file,
        )

        weather = WeatherFile(
            file_path=epw_file,
            location="Chicago",
            scenario="TMY",
        )

        job = SimulationJob(
            building=building,
            weather_file=weather,
            simulation_type="baseline",
            output_directory=tmp_path / "output",
            output_prefix="test",
        )

        from eppy.modeleditor import IDF
        idf = Mock(spec=IDF)

        return BaselineContext(
            job=job,
            idf=idf,
            working_directory=tmp_path,
        )

    def test_run_success(self, baseline_service, simulation_context):
        """测试成功运行"""
        # Act
        result = baseline_service.run(simulation_context)

        # Assert
        assert result is not None
        assert result.success
        assert result.source_eui == 150.0

    def test_prepare_creates_output_directory(
        self,
        baseline_service,
        simulation_context,
    ):
        """测试准备阶段创建输出目录"""
        # Arrange
        output_dir = simulation_context.job.output_directory

        # Act
        baseline_service.prepare(simulation_context)

        # Assert
        assert output_dir.exists()

    def test_prepare_validates_files(self, baseline_service, simulation_context):
        """测试文件验证"""
        # Arrange - 删除 IDF 文件
        simulation_context.job.building.idf_file_path.unlink()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            baseline_service.prepare(simulation_context)
```

### 集成测试

```python
"""
服务层集成测试
"""

import pytest
from pathlib import Path

from backend.utils.config import ConfigManager, setup_container
from backend.factories import ServiceFactory


class TestBaselineSimulationIntegration:
    """基准模拟集成测试"""

    @pytest.fixture
    def integration_setup(self):
        """集成测试设置"""
        config = ConfigManager()
        container = setup_container(config)
        service_factory = container.resolve(ServiceFactory)
        return service_factory

    def test_full_baseline_simulation(
        self,
        integration_setup,
        real_building,
        real_weather,
        tmp_path,
    ):
        """测试完整的基准模拟流程"""
        # Arrange
        service_factory = integration_setup
        baseline_service = service_factory.create_service("baseline")

        job = SimulationJob(
            building=real_building,
            weather_file=real_weather,
            simulation_type="baseline",
            output_directory=tmp_path / "output",
            output_prefix="integration_test",
        )

        from eppy.modeleditor import IDF
        idf = IDF(str(real_building.idf_file_path))

        context = BaselineContext(
            job=job,
            idf=idf,
            working_directory=tmp_path,
        )

        # Act
        result = baseline_service.run(context)

        # Assert
        assert result.success
        assert result.source_eui > 0
        assert result.site_eui > 0
        assert result.execution_time > 0

        # 验证输出文件存在
        assert (job.output_directory / f"{job.output_prefix}Table.csv").exists()
```

---

## 总结

服务层是系统业务逻辑的核心，实现了：

### 核心特性

1. **统一接口**：`ISimulationService` 定义标准流程
2. **模板方法**：`BaseSimulationService` 提供通用逻辑
3. **策略模式**：支持多种优化算法
4. **依赖注入**：解耦服务依赖
5. **完整类型提示**：100% 类型安全

### 扩展性

添加新服务非常简单：

```python
# 1. 定义上下文（如果需要）
@dataclass
class NewContext(SimulationContext):
    # 特定字段
    ...

# 2. 实现服务
class NewSimulationService(BaseSimulationService[NewContext]):
    def _prepare_specific(self, context: NewContext) -> None:
        # 准备逻辑
        ...

    def _modify_idf(self, context: NewContext) -> None:
        # IDF 修改逻辑
        ...

# 3. 注册到工厂
service_factory.register("new_type", NewSimulationService)

# 完成！可以使用了
```

### 下一步

继续阅读：
- [03_INFRASTRUCTURE_LAYER.md](03_INFRASTRUCTURE_LAYER.md) - 基础设施层实现
- [04_APPLICATION_LAYER.md](04_APPLICATION_LAYER.md) - 应用层实现
- [07_TESTING_STRATEGY.md](07_TESTING_STRATEGY.md) - 测试策略

---

**文档版本**: 1.0
**最后更新**: 2025-10-27
**下一篇**: [03_INFRASTRUCTURE_LAYER.md](03_INFRASTRUCTURE_LAYER.md)
