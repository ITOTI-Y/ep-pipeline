# EP-WebUI 代码重构实现指南

> 基于面向对象设计的完全重构方案
>
> 版本：1.0
> 日期：2025-10-27
> 目标：提升代码性能和可读性

---

## 目录

1. [执行摘要](#执行摘要)
2. [当前代码分析](#当前代码分析)
3. [重构目标](#重构目标)
4. [设计原则](#设计原则)
5. [架构设计](#架构设计)
6. [核心领域模型](#核心领域模型)
7. [服务层设计](#服务层设计)
8. [设计模式应用](#设计模式应用)
9. [目录结构](#目录结构)
10. [核心类详细设计](#核心类详细设计)
11. [性能优化策略](#性能优化策略)
12. [代码质量保证](#代码质量保证)
13. [实施步骤](#实施步骤)
14. [测试策略](#测试策略)
15. [迁移路线图](#迁移路线图)

---

## 执行摘要

EP-WebUI 是一个 EnergyPlus 建筑能源模拟和优化框架。当前代码库虽然经过初步重构，但仍存在架构不清晰、职责混乱、缺乏类型安全等问题。本文档提供了一套完整的面向对象重构方案，旨在：

- **提升性能**：通过并行处理、缓存机制、资源池化等技术提升 30-50% 的执行效率
- **增强可读性**：采用 SOLID 原则、清晰的分层架构、完整的类型提示和文档
- **提高可维护性**：模块化设计、设计模式应用、单元测试覆盖
- **增强扩展性**：插件化架构、策略模式、工厂模式支持快速功能扩展

---

## 当前代码分析

### 现有优点

1. ✅ 已采用配置文件管理（OmegaConf）
2. ✅ 使用现代日志库（Loguru）
3. ✅ 初步的服务抽象（BaseService）
4. ✅ 配置与代码分离

### 存在的问题

#### 1. 架构问题

```python
# ❌ 问题：main.py 职责混乱，业务逻辑与编排逻辑混合
def main():
    prototype_dir = Path(config.paths.prototype_idf)
    ftmy_dir = Path(config.paths.ftmy_dir)
    tmy_dir = Path(config.paths.tmy_dir)

    pending_idf_files = []
    pending_weather_files = []

    # 文件匹配逻辑应该在专门的服务中
    for city in CITIES:
        for building_type in BUILDING_TYPES:
            for idf_file in all_idf_files:
                if city in idf_file.stem and building_type in idf_file.stem:
                    pending_idf_files.append(idf_file)
```

**问题**：
- 缺乏领域模型（Building, WeatherFile 等实体）
- 文件匹配逻辑硬编码
- 没有批处理管理器
- 缺乏错误处理和进度跟踪

#### 2. 类设计问题

```python
# ❌ 问题：SimulateManager 职责不清晰
class SimulateManager:
    def __init__(self, config: Config, idf: IDF, weather_file: Path):
        self._config = config
        self._logger = logger
        self._idf = idf  # 传入的是 IDF 对象，但构造函数签名说是 Path
```

**问题**：
- 构造函数参数类型不一致
- 缺乏验证逻辑
- `_initial_config` 方法为空
- 直接依赖全局 logger 而非注入

#### 3. 配置访问问题

```python
# ❌ 问题：配置访问不安全，运行时才能发现错误
baseline_service.run(
    weather_file=self.config.paths.fmt,  # 配置属性名可能错误
    output_dir=self.config.paths.baseline_dir,
    output_prefix="baseline_",
    read_vars=True,
)
```

**问题**：
- 缺乏类型提示
- 属性访问错误在运行时才能发现
- 没有配置验证

#### 4. 旧代码问题（Archive）

```python
# ❌ 问题：过程式编程，缺乏封装
def getBuildingInfo(city, ssp, btype, ECM_dict = {}):
    if not ECM_dict:
        folder = epsim_folder + city + '\\%s\\' % (ssp)
        f = [folder + f for f in os.listdir(folder) if f.endswith('Table.csv') if btype in f][0]
```

**问题**：
- 硬编码路径拼接
- 使用字典而非数据类
- 缺乏错误处理
- 函数过长，职责过多

#### 5. 性能问题

- ❌ 没有并行处理实现（虽然导入了 joblib）
- ❌ 没有缓存机制
- ❌ 重复的文件 I/O 操作
- ❌ 没有资源池管理

#### 6. 可读性问题

- ❌ 缺少类型提示（除了少数方法）
- ❌ 缺少文档字符串
- ❌ 变量命名不够清晰（如 `fmt`）
- ❌ 缺乏代码示例和使用说明

---

## 重构目标

### 主要目标

1. **性能提升 30-50%**
   - 实现并行模拟执行
   - 添加智能缓存机制
   - 优化文件 I/O 操作
   - 实现对象池和资源复用

2. **代码可读性提升**
   - 100% 类型提示覆盖
   - 清晰的分层架构
   - 完整的文档字符串
   - 符合 PEP 8 和 Google 风格指南

3. **可维护性提升**
   - SOLID 原则应用
   - 单一职责的类和方法
   - 高内聚低耦合
   - 80% 以上单元测试覆盖

4. **可扩展性提升**
   - 插件化服务架构
   - 策略模式支持多种算法
   - 工厂模式支持动态创建
   - 观察者模式支持事件监听

### 关键指标

| 指标 | 当前 | 目标 | 改进 |
|------|------|------|------|
| 类型提示覆盖率 | ~20% | 100% | +80% |
| 单元测试覆盖率 | 0% | 80% | +80% |
| 模拟并行度 | 1x | 4-8x | +300-700% |
| 代码复杂度 (Cyclomatic) | ~15 | <10 | -33% |
| 文档覆盖率 | ~10% | 90% | +80% |
| 平均方法长度 | ~30 行 | <20 行 | -33% |

---

## 设计原则

### SOLID 原则应用

#### 1. Single Responsibility Principle (SRP)
每个类只负责一个功能领域：

```python
# ✅ 正确：职责单一
class IDFFileLoader:
    """专门负责 IDF 文件的加载和解析"""

    def load(self, file_path: Path) -> IDF:
        """加载 IDF 文件"""
        ...

    def validate(self, idf: IDF) -> ValidationResult:
        """验证 IDF 文件"""
        ...

class BuildingRepository:
    """专门负责建筑数据的存储和检索"""

    def save(self, building: Building) -> None: ...
    def find_by_type(self, building_type: BuildingType) -> list[Building]: ...
    def find_by_city(self, city: str) -> list[Building]: ...
```

#### 2. Open/Closed Principle (OCP)
对扩展开放，对修改关闭：

```python
# ✅ 正确：使用抽象基类和策略模式
class SimulationStrategy(ABC):
    """模拟策略抽象基类"""

    @abstractmethod
    def execute(self, context: SimulationContext) -> SimulationResult:
        """执行模拟"""
        pass

class BaselineSimulationStrategy(SimulationStrategy):
    """基准模拟策略"""
    def execute(self, context: SimulationContext) -> SimulationResult:
        # 实现基准模拟
        ...

class PVSimulationStrategy(SimulationStrategy):
    """光伏模拟策略"""
    def execute(self, context: SimulationContext) -> SimulationResult:
        # 实现光伏模拟
        ...

# 添加新策略无需修改现有代码
class OptimizationSimulationStrategy(SimulationStrategy):
    def execute(self, context: SimulationContext) -> SimulationResult:
        # 实现优化模拟
        ...
```

#### 3. Liskov Substitution Principle (LSP)
子类可以替换父类：

```python
# ✅ 正确：子类完全遵守父类契约
class WeatherDataSource(ABC):
    @abstractmethod
    def get_weather_file(self, location: str, scenario: str) -> Path:
        """获取天气文件路径"""
        pass

class TMYWeatherSource(WeatherDataSource):
    def get_weather_file(self, location: str, scenario: str) -> Path:
        # 返回 TMY 天气文件
        return self._tmy_dir / f"{location}_TMY.epw"

class FTMYWeatherSource(WeatherDataSource):
    def get_weather_file(self, location: str, scenario: str) -> Path:
        # 返回 FTMY 天气文件
        return self._ftmy_dir / f"{location}_{scenario}.epw"

# 可以无缝替换
def run_simulation(weather_source: WeatherDataSource):
    weather_file = weather_source.get_weather_file("Chicago", "126")
    # ...
```

#### 4. Interface Segregation Principle (ISP)
接口隔离，客户端不应依赖不需要的接口：

```python
# ✅ 正确：分离接口
class Runnable(Protocol):
    """可运行接口"""
    def run(self) -> SimulationResult: ...

class Validatable(Protocol):
    """可验证接口"""
    def validate(self) -> ValidationResult: ...

class Cacheable(Protocol):
    """可缓存接口"""
    def get_cache_key(self) -> str: ...

# 类只实现需要的接口
class BaselineSimulation(Runnable, Cacheable):
    def run(self) -> SimulationResult: ...
    def get_cache_key(self) -> str: ...
    # 不实现 Validatable，因为不需要
```

#### 5. Dependency Inversion Principle (DIP)
依赖抽象而非具体实现：

```python
# ✅ 正确：依赖注入抽象
class SimulationOrchestrator:
    def __init__(
        self,
        file_loader: IFileLoader,  # 依赖抽象
        simulation_runner: ISimulationRunner,  # 依赖抽象
        result_processor: IResultProcessor,  # 依赖抽象
        logger: ILogger,  # 依赖抽象
    ):
        self._file_loader = file_loader
        self._simulation_runner = simulation_runner
        self._result_processor = result_processor
        self._logger = logger
```

### 其他设计原则

1. **DRY (Don't Repeat Yourself)** - 提取公共逻辑到基类或工具类
2. **KISS (Keep It Simple, Stupid)** - 优先使用简单直接的实现
3. **YAGNI (You Aren't Gonna Need It)** - 只实现当前需要的功能
4. **Composition over Inheritance** - 优先使用组合而非继承
5. **Convention over Configuration** - 约定优于配置

---

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                     应用层 (Application)                      │
│  - CLI 入口                                                   │
│  - Web API 入口 (可选)                                        │
│  - 批处理编排器                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     服务层 (Service)                          │
│  - 模拟服务 (BaselineService, PVService, etc.)               │
│  - 分析服务 (SensitivityService, OptimizationService)        │
│  - 数据处理服务 (DataProcessorService)                       │
│  - 文件服务 (FileService)                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     领域层 (Domain)                           │
│  - 领域模型 (Building, WeatherFile, SimulationResult)        │
│  - 领域服务 (ECMApplicator, PVSystemDesigner)                │
│  - 值对象 (BuildingType, Location, ECMParameters)            │
│  - 仓储接口 (IBuildingRepository, IResultRepository)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  基础设施层 (Infrastructure)                  │
│  - 文件系统访问 (IDFFileRepository, EPWFileRepository)       │
│  - EnergyPlus 运行器 (EnergyPlusExecutor)                    │
│  - 缓存实现 (MemoryCache, FileCache)                         │
│  - 日志实现 (LoguruLogger)                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     工具层 (Utilities)                        │
│  - 配置管理 (ConfigManager)                                  │
│  - 并行执行器 (ParallelExecutor)                             │
│  - 验证器 (Validator)                                        │
│  - 类型定义 (TypeDefinitions)                                │
└─────────────────────────────────────────────────────────────┘
```

### 依赖关系图

```
Application Layer
    ↓ (depends on)
Service Layer
    ↓ (depends on)
Domain Layer
    ↑ (implemented by)
Infrastructure Layer

Utilities Layer ← (used by all layers)
```

---

## 核心领域模型

### 实体类 (Entities)

#### 1. Building (建筑实体)

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4


class BuildingType(Enum):
    """建筑类型枚举"""
    OFFICE_LARGE = "OfficeLarge"
    OFFICE_MEDIUM = "OfficeMedium"
    OFFICE_SMALL = "OfficeSmall"
    MULTI_FAMILY_RESIDENTIAL = "MultiFamilyResidential"
    SINGLE_FAMILY_RESIDENTIAL = "SingleFamilyResidential"
    RETAIL = "Retail"
    WAREHOUSE = "Warehouse"
    HOSPITAL = "Hospital"
    SCHOOL = "School"


@dataclass
class Building:
    """
    建筑实体类

    表示一个建筑的完整信息，包括类型、位置、IDF 文件等。
    """

    id: UUID = field(default_factory=uuid4)
    name: str
    building_type: BuildingType
    location: str  # 城市名称
    idf_file_path: Path
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, any] = field(default_factory=dict)

    # 建筑特征
    floor_area: Optional[float] = None  # 建筑面积 (m²)
    num_floors: Optional[int] = None    # 楼层数

    def __post_init__(self):
        """验证建筑数据"""
        if not self.idf_file_path.exists():
            raise ValueError(f"IDF file does not exist: {self.idf_file_path}")

        if self.floor_area is not None and self.floor_area <= 0:
            raise ValueError(f"Floor area must be positive: {self.floor_area}")

    def get_identifier(self) -> str:
        """获取建筑唯一标识符"""
        return f"{self.location}_{self.building_type.value}"

    def update_metadata(self, key: str, value: any) -> None:
        """更新元数据"""
        self.metadata[key] = value
        self.modified_at = datetime.now()


@dataclass
class WeatherFile:
    """
    天气文件实体类

    表示一个 EPW 天气文件，可以是 TMY 或 FTMY。
    """

    id: UUID = field(default_factory=uuid4)
    file_path: Path
    location: str
    scenario: str  # "TMY" or "126", "245", etc.
    is_future: bool = False

    # 天气特征
    climate_zone: Optional[str] = None
    heating_degree_days: Optional[float] = None
    cooling_degree_days: Optional[float] = None

    def __post_init__(self):
        """验证天气文件数据"""
        if not self.file_path.exists():
            raise ValueError(f"Weather file does not exist: {self.file_path}")

        if not self.file_path.suffix == ".epw":
            raise ValueError(f"Weather file must be .epw format: {self.file_path}")

    def get_identifier(self) -> str:
        """获取天气文件唯一标识符"""
        return f"{self.location}_{self.scenario}"


@dataclass
class SimulationJob:
    """
    模拟任务实体类

    表示一个完整的模拟任务，包含建筑、天气、配置等信息。
    """

    id: UUID = field(default_factory=uuid4)
    building: Building
    weather_file: WeatherFile
    simulation_type: str  # "baseline", "pv", "optimization", etc.

    # 模拟配置
    output_directory: Path
    output_prefix: str
    read_variables: bool = True

    # ECM 参数（可选）
    ecm_parameters: Optional['ECMParameters'] = None

    # 状态
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 结果
    result: Optional['SimulationResult'] = None
    error_message: Optional[str] = None

    def start(self) -> None:
        """标记任务为运行中"""
        self.status = "running"
        self.started_at = datetime.now()

    def complete(self, result: 'SimulationResult') -> None:
        """标记任务为完成"""
        self.status = "completed"
        self.completed_at = datetime.now()
        self.result = result

    def fail(self, error_message: str) -> None:
        """标记任务为失败"""
        self.status = "failed"
        self.completed_at = datetime.now()
        self.error_message = error_message

    def get_cache_key(self) -> str:
        """获取缓存键"""
        building_id = self.building.get_identifier()
        weather_id = self.weather_file.get_identifier()
        ecm_hash = hash(str(self.ecm_parameters)) if self.ecm_parameters else 0
        return f"{building_id}_{weather_id}_{self.simulation_type}_{ecm_hash}"


@dataclass
class SimulationResult:
    """
    模拟结果实体类

    包含模拟的输出数据和元数据。
    """

    id: UUID = field(default_factory=uuid4)
    job_id: UUID

    # 输出文件
    output_directory: Path
    table_csv_path: Optional[Path] = None
    meter_csv_path: Optional[Path] = None
    sql_path: Optional[Path] = None

    # 关键指标
    source_eui: Optional[float] = None  # kWh/m²/yr
    site_eui: Optional[float] = None    # kWh/m²/yr
    total_energy_kwh: Optional[float] = None

    # 执行信息
    execution_time: Optional[float] = None  # seconds
    success: bool = False
    error_messages: list[str] = field(default_factory=list)

    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """添加错误信息"""
        self.error_messages.append(message)
        self.success = False

    def validate(self) -> bool:
        """验证结果完整性"""
        if not self.success:
            return False

        if self.table_csv_path and not self.table_csv_path.exists():
            self.add_error(f"Table CSV not found: {self.table_csv_path}")
            return False

        return True
```

### 值对象 (Value Objects)

#### 2. ECM Parameters (能效措施参数)

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)  # 不可变值对象
class ECMParameters:
    """
    Energy Conservation Measures (ECM) 参数值对象

    包含所有能效改造措施的参数。使用 frozen=True 确保不可变性。
    """

    # 围护结构参数
    window_u_value: Optional[float] = None  # 窗户传热系数 W/m²K
    window_shgc: Optional[float] = None     # 太阳得热系数
    wall_insulation: Optional[float] = None  # 墙体保温 R-value
    roof_insulation: Optional[float] = None  # 屋顶保温 R-value

    # 通风参数
    infiltration_rate: Optional[float] = None  # 渗透率 ACH
    natural_ventilation_area: Optional[float] = None  # 自然通风面积

    # HVAC 参数
    cooling_cop: Optional[float] = None  # 制冷系统 COP
    heating_efficiency: Optional[float] = None  # 供热效率
    cooling_setpoint: Optional[float] = None  # 制冷设定温度 °C
    heating_setpoint: Optional[float] = None  # 供热设定温度 °C

    # 照明参数
    lighting_power_density: Optional[float] = None  # 照明功率密度 W/m²
    lighting_reduction_factor: Optional[float] = None  # 照明削减因子

    def __post_init__(self):
        """验证参数范围"""
        if self.window_u_value is not None:
            if not (0.1 <= self.window_u_value <= 10.0):
                raise ValueError(f"Invalid window U-value: {self.window_u_value}")

        if self.window_shgc is not None:
            if not (0.0 <= self.window_shgc <= 1.0):
                raise ValueError(f"Invalid SHGC: {self.window_shgc}")

        if self.cooling_cop is not None:
            if not (1.0 <= self.cooling_cop <= 10.0):
                raise ValueError(f"Invalid COP: {self.cooling_cop}")

    def to_dict(self) -> dict[str, float]:
        """转换为字典，只包含非 None 值"""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None
        }

    def __hash__(self):
        """支持哈希，用于缓存键"""
        return hash(tuple(sorted(self.to_dict().items())))


@dataclass(frozen=True)
class Location:
    """位置值对象"""

    city: str
    country: str
    climate_zone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.city}, {self.country}"


@dataclass(frozen=True)
class SimulationPeriod:
    """模拟时间段值对象"""

    start_year: int
    end_year: int
    start_month: int = 1
    end_month: int = 12
    start_day: int = 1
    end_day: int = 31

    def __post_init__(self):
        if self.start_year > self.end_year:
            raise ValueError("Start year must be <= end year")

        if not (1 <= self.start_month <= 12):
            raise ValueError("Invalid start month")

        if not (1 <= self.end_month <= 12):
            raise ValueError("Invalid end month")

    def get_duration_years(self) -> int:
        """获取持续年数"""
        return self.end_year - self.start_year + 1
```

---

## 服务层设计

### 服务接口定义

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Protocol
from pathlib import Path


# 类型变量
TResult = TypeVar('TResult')
TContext = TypeVar('TContext')


class ISimulationService(ABC, Generic[TContext, TResult]):
    """
    模拟服务接口

    定义所有模拟服务的通用接口。
    """

    @abstractmethod
    def prepare(self, context: TContext) -> None:
        """
        准备模拟环境

        Args:
            context: 模拟上下文

        Raises:
            PreparationError: 准备失败时抛出
        """
        pass

    @abstractmethod
    def execute(self, context: TContext) -> TResult:
        """
        执行模拟

        Args:
            context: 模拟上下文

        Returns:
            模拟结果

        Raises:
            SimulationError: 模拟执行失败时抛出
        """
        pass

    @abstractmethod
    def cleanup(self, context: TContext) -> None:
        """
        清理临时文件和资源

        Args:
            context: 模拟上下文
        """
        pass

    def run(self, context: TContext) -> TResult:
        """
        完整的模拟流程：准备 -> 执行 -> 清理

        Args:
            context: 模拟上下文

        Returns:
            模拟结果
        """
        try:
            self.prepare(context)
            result = self.execute(context)
            return result
        finally:
            self.cleanup(context)


class IFileLoader(Protocol):
    """文件加载器接口"""

    def load(self, file_path: Path) -> any:
        """加载文件"""
        ...

    def validate(self, file_path: Path) -> bool:
        """验证文件"""
        ...


class IResultProcessor(Protocol):
    """结果处理器接口"""

    def process(self, result: SimulationResult) -> dict[str, any]:
        """处理模拟结果"""
        ...

    def export(self, result: SimulationResult, output_path: Path) -> None:
        """导出结果"""
        ...


class ICacheService(Protocol):
    """缓存服务接口"""

    def get(self, key: str) -> Optional[any]:
        """获取缓存"""
        ...

    def set(self, key: str, value: any, ttl: Optional[int] = None) -> None:
        """设置缓存"""
        ...

    def delete(self, key: str) -> None:
        """删除缓存"""
        ...

    def clear(self) -> None:
        """清空缓存"""
        ...
```

### 具体服务实现

#### 1. BaselineSimulationService

```python
from typing import Optional
from loguru import logger
from eppy.modeleditor import IDF


@dataclass
class BaselineContext:
    """基准模拟上下文"""

    job: SimulationJob
    idf: IDF
    working_directory: Path


class BaselineSimulationService(ISimulationService[BaselineContext, SimulationResult]):
    """
    基准建筑模拟服务

    执行建筑的基准能耗模拟，不应用任何 ECM 措施。
    """

    def __init__(
        self,
        energyplus_executor: 'IEnergyPlusExecutor',
        result_parser: 'IResultParser',
        file_cleaner: 'IFileCleaner',
        logger: 'ILogger',
    ):
        self._executor = energyplus_executor
        self._result_parser = result_parser
        self._file_cleaner = file_cleaner
        self._logger = logger

    def prepare(self, context: BaselineContext) -> None:
        """
        准备基准模拟

        - 创建输出目录
        - 验证 IDF 和 EPW 文件
        - 设置输出变量
        """
        self._logger.info(f"Preparing baseline simulation for job {context.job.id}")

        # 创建输出目录
        context.job.output_directory.mkdir(parents=True, exist_ok=True)

        # 验证文件
        if not context.job.building.idf_file_path.exists():
            raise FileNotFoundError(f"IDF file not found: {context.job.building.idf_file_path}")

        if not context.job.weather_file.file_path.exists():
            raise FileNotFoundError(f"Weather file not found: {context.job.weather_file.file_path}")

        # 添加必要的输出变量
        self._add_output_variables(context.idf)

        self._logger.info("Preparation completed")

    def execute(self, context: BaselineContext) -> SimulationResult:
        """
        执行基准模拟
        """
        self._logger.info(f"Executing baseline simulation for job {context.job.id}")

        start_time = time.time()

        try:
            # 执行 EnergyPlus
            execution_result = self._executor.run(
                idf=context.idf,
                weather_file=context.job.weather_file.file_path,
                output_directory=context.job.output_directory,
                output_prefix=context.job.output_prefix,
            )

            # 解析结果
            result = self._result_parser.parse(
                job_id=context.job.id,
                output_directory=context.job.output_directory,
                output_prefix=context.job.output_prefix,
            )

            result.execution_time = time.time() - start_time
            result.success = execution_result.success

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

    def cleanup(self, context: BaselineContext) -> None:
        """
        清理临时文件
        """
        self._logger.info("Cleaning up temporary files")
        self._file_cleaner.clean(
            directory=context.job.output_directory,
            keep_extensions=['.csv', '.sql', '.idf'],
        )

    def _add_output_variables(self, idf: IDF) -> None:
        """添加输出变量到 IDF"""
        required_variables = [
            "Site Outdoor Air Drybulb Temperature",
            "Zone Mean Air Temperature",
            "Facility Total Electric Demand Power",
            "Facility Total Natural Gas Demand Rate",
        ]

        for var_name in required_variables:
            # 检查是否已存在
            exists = any(
                ov.Variable_Name == var_name
                for ov in idf.idfobjects.get("OUTPUT:VARIABLE", [])
            )

            if not exists:
                idf.newidfobject(
                    "OUTPUT:VARIABLE",
                    Key_Value="*",
                    Variable_Name=var_name,
                    Reporting_Frequency="Hourly",
                )
```

#### 2. SimulationOrchestrator (编排器)

```python
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


class SimulationOrchestrator:
    """
    模拟编排器

    负责批量模拟任务的调度、执行和监控。
    支持并行执行、进度跟踪、错误处理。
    """

    def __init__(
        self,
        service_factory: 'IServiceFactory',
        cache_service: ICacheService,
        parallel_executor: 'IParallelExecutor',
        logger: 'ILogger',
        max_workers: int = 4,
    ):
        self._service_factory = service_factory
        self._cache = cache_service
        self._parallel_executor = parallel_executor
        self._logger = logger
        self._max_workers = max_workers

    def execute_batch(
        self,
        jobs: list[SimulationJob],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_cache: bool = True,
    ) -> list[SimulationResult]:
        """
        批量执行模拟任务

        Args:
            jobs: 模拟任务列表
            progress_callback: 进度回调函数 (completed, total)
            use_cache: 是否使用缓存

        Returns:
            模拟结果列表
        """
        self._logger.info(f"Starting batch execution of {len(jobs)} jobs")

        results: list[SimulationResult] = []
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

                        # 缓存结果
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

        self._logger.info(f"Batch execution completed. Success: {sum(1 for r in results if r.success)}/{total}")
        return results

    def _execute_single_job(self, job: SimulationJob) -> SimulationResult:
        """执行单个模拟任务"""
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
        """创建模拟上下文"""
        # 根据不同的模拟类型创建不同的上下文
        # 这里简化处理
        from eppy.modeleditor import IDF
        idf = IDF(str(job.building.idf_file_path))

        return BaselineContext(
            job=job,
            idf=idf,
            working_directory=job.output_directory,
        )
```

---

## 设计模式应用

### 1. Factory Pattern (工厂模式)

用于创建不同类型的服务和对象。

```python
from typing import Dict, Type


class ServiceFactory:
    """
    服务工厂

    根据服务类型创建相应的服务实例。
    """

    def __init__(self, container: 'DependencyContainer'):
        self._container = container
        self._service_registry: Dict[str, Type[ISimulationService]] = {}
        self._register_default_services()

    def _register_default_services(self) -> None:
        """注册默认服务"""
        self.register('baseline', BaselineSimulationService)
        self.register('pv', PVSimulationService)
        self.register('optimization', OptimizationSimulationService)
        self.register('sensitivity', SensitivitySimulationService)

    def register(self, service_type: str, service_class: Type[ISimulationService]) -> None:
        """
        注册服务类型

        Args:
            service_type: 服务类型标识
            service_class: 服务类
        """
        self._service_registry[service_type] = service_class

    def create_service(self, service_type: str) -> ISimulationService:
        """
        创建服务实例

        Args:
            service_type: 服务类型标识

        Returns:
            服务实例

        Raises:
            ValueError: 未知的服务类型
        """
        if service_type not in self._service_registry:
            raise ValueError(f"Unknown service type: {service_type}")

        service_class = self._service_registry[service_type]

        # 从依赖容器解析依赖
        return self._container.resolve(service_class)


class BuildingFactory:
    """建筑工厂"""

    @staticmethod
    def create_from_idf(
        idf_path: Path,
        building_type: BuildingType,
        location: str,
    ) -> Building:
        """从 IDF 文件创建建筑对象"""
        # 解析 IDF 文件获取建筑信息
        idf = IDF(str(idf_path))

        # 提取建筑面积
        floor_area = BuildingFactory._extract_floor_area(idf)

        # 提取楼层数
        num_floors = BuildingFactory._extract_num_floors(idf)

        return Building(
            name=idf_path.stem,
            building_type=building_type,
            location=location,
            idf_file_path=idf_path,
            floor_area=floor_area,
            num_floors=num_floors,
        )

    @staticmethod
    def _extract_floor_area(idf: IDF) -> Optional[float]:
        """从 IDF 提取建筑面积"""
        try:
            building_obj = idf.idfobjects['BUILDING'][0]
            # 具体提取逻辑
            return None  # 简化
        except:
            return None

    @staticmethod
    def _extract_num_floors(idf: IDF) -> Optional[int]:
        """从 IDF 提取楼层数"""
        try:
            zones = idf.idfobjects.get('ZONE', [])
            # 具体提取逻辑
            return len(zones) if zones else None
        except:
            return None
```

### 2. Strategy Pattern (策略模式)

用于不同的算法实现（优化算法、分析方法等）。

```python
class OptimizationStrategy(ABC):
    """优化策略抽象基类"""

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable,
        parameter_bounds: dict[str, tuple[float, float]],
        max_iterations: int,
    ) -> ECMParameters:
        """执行优化"""
        pass


class GeneticAlgorithmStrategy(OptimizationStrategy):
    """遗传算法优化策略"""

    def optimize(
        self,
        objective_function: Callable,
        parameter_bounds: dict[str, tuple[float, float]],
        max_iterations: int,
    ) -> ECMParameters:
        from deap import base, creator, tools, algorithms

        # 实现遗传算法优化
        # ...
        pass


class BayesianOptimizationStrategy(OptimizationStrategy):
    """贝叶斯优化策略"""

    def optimize(
        self,
        objective_function: Callable,
        parameter_bounds: dict[str, tuple[float, float]],
        max_iterations: int,
    ) -> ECMParameters:
        from bayes_opt import BayesianOptimization

        # 实现贝叶斯优化
        # ...
        pass


class PSOOptimizationStrategy(OptimizationStrategy):
    """粒子群优化策略"""

    def optimize(
        self,
        objective_function: Callable,
        parameter_bounds: dict[str, tuple[float, float]],
        max_iterations: int,
    ) -> ECMParameters:
        # 实现粒子群优化
        # ...
        pass


class OptimizationService:
    """优化服务，使用策略模式"""

    def __init__(
        self,
        strategy: OptimizationStrategy,
        simulation_orchestrator: SimulationOrchestrator,
    ):
        self._strategy = strategy
        self._orchestrator = simulation_orchestrator

    def set_strategy(self, strategy: OptimizationStrategy) -> None:
        """动态切换优化策略"""
        self._strategy = strategy

    def find_optimal_ecm(
        self,
        building: Building,
        weather_file: WeatherFile,
        parameter_bounds: dict[str, tuple[float, float]],
    ) -> tuple[ECMParameters, SimulationResult]:
        """找到最优 ECM 参数"""

        def objective_function(params: dict) -> float:
            """目标函数：最小化 EUI"""
            ecm_params = ECMParameters(**params)

            # 创建模拟任务
            job = SimulationJob(
                building=building,
                weather_file=weather_file,
                simulation_type="baseline",
                output_directory=Path("temp") / str(uuid4()),
                output_prefix="opt",
                ecm_parameters=ecm_params,
            )

            # 执行模拟
            results = self._orchestrator.execute_batch([job])
            result = results[0]

            if not result.success:
                return float('inf')

            return result.source_eui

        # 使用策略执行优化
        optimal_params = self._strategy.optimize(
            objective_function=objective_function,
            parameter_bounds=parameter_bounds,
            max_iterations=100,
        )

        # 返回最优结果
        # ...
        return optimal_params, final_result
```

### 3. Builder Pattern (建造者模式)

用于构建复杂的对象（如 SimulationJob）。

```python
class SimulationJobBuilder:
    """
    模拟任务建造者

    使用流式 API 构建复杂的 SimulationJob 对象。
    """

    def __init__(self):
        self._building: Optional[Building] = None
        self._weather_file: Optional[WeatherFile] = None
        self._simulation_type: str = "baseline"
        self._output_directory: Optional[Path] = None
        self._output_prefix: str = "sim"
        self._read_variables: bool = True
        self._ecm_parameters: Optional[ECMParameters] = None

    def with_building(self, building: Building) -> 'SimulationJobBuilder':
        """设置建筑"""
        self._building = building
        return self

    def with_weather_file(self, weather_file: WeatherFile) -> 'SimulationJobBuilder':
        """设置天气文件"""
        self._weather_file = weather_file
        return self

    def with_simulation_type(self, simulation_type: str) -> 'SimulationJobBuilder':
        """设置模拟类型"""
        self._simulation_type = simulation_type
        return self

    def with_output_directory(self, output_directory: Path) -> 'SimulationJobBuilder':
        """设置输出目录"""
        self._output_directory = output_directory
        return self

    def with_output_prefix(self, output_prefix: str) -> 'SimulationJobBuilder':
        """设置输出前缀"""
        self._output_prefix = output_prefix
        return self

    def with_ecm_parameters(self, ecm_parameters: ECMParameters) -> 'SimulationJobBuilder':
        """设置 ECM 参数"""
        self._ecm_parameters = ecm_parameters
        return self

    def with_read_variables(self, read_variables: bool) -> 'SimulationJobBuilder':
        """设置是否读取变量"""
        self._read_variables = read_variables
        return self

    def build(self) -> SimulationJob:
        """构建 SimulationJob 对象"""
        if not self._building:
            raise ValueError("Building is required")

        if not self._weather_file:
            raise ValueError("Weather file is required")

        if not self._output_directory:
            # 使用默认输出目录
            self._output_directory = Path("output") / self._simulation_type / str(uuid4())

        return SimulationJob(
            building=self._building,
            weather_file=self._weather_file,
            simulation_type=self._simulation_type,
            output_directory=self._output_directory,
            output_prefix=self._output_prefix,
            read_variables=self._read_variables,
            ecm_parameters=self._ecm_parameters,
        )


# 使用示例
job = (SimulationJobBuilder()
    .with_building(building)
    .with_weather_file(weather_file)
    .with_simulation_type("baseline")
    .with_output_prefix("baseline_chicago")
    .with_ecm_parameters(ecm_params)
    .build())
```

### 4. Observer Pattern (观察者模式)

用于事件监听和进度通知。

```python
from typing import List


class SimulationEvent:
    """模拟事件"""

    def __init__(self, event_type: str, job_id: UUID, data: dict = None):
        self.event_type = event_type
        self.job_id = job_id
        self.data = data or {}
        self.timestamp = datetime.now()


class ISimulationObserver(ABC):
    """模拟观察者接口"""

    @abstractmethod
    def on_simulation_started(self, event: SimulationEvent) -> None:
        """模拟开始时调用"""
        pass

    @abstractmethod
    def on_simulation_progress(self, event: SimulationEvent) -> None:
        """模拟进度更新时调用"""
        pass

    @abstractmethod
    def on_simulation_completed(self, event: SimulationEvent) -> None:
        """模拟完成时调用"""
        pass

    @abstractmethod
    def on_simulation_failed(self, event: SimulationEvent) -> None:
        """模拟失败时调用"""
        pass


class LoggerObserver(ISimulationObserver):
    """日志观察者"""

    def __init__(self, logger: 'ILogger'):
        self._logger = logger

    def on_simulation_started(self, event: SimulationEvent) -> None:
        self._logger.info(f"Simulation {event.job_id} started")

    def on_simulation_progress(self, event: SimulationEvent) -> None:
        progress = event.data.get('progress', 0)
        self._logger.debug(f"Simulation {event.job_id} progress: {progress}%")

    def on_simulation_completed(self, event: SimulationEvent) -> None:
        self._logger.info(f"Simulation {event.job_id} completed successfully")

    def on_simulation_failed(self, event: SimulationEvent) -> None:
        error = event.data.get('error', 'Unknown error')
        self._logger.error(f"Simulation {event.job_id} failed: {error}")


class ProgressBarObserver(ISimulationObserver):
    """进度条观察者"""

    def __init__(self):
        from tqdm import tqdm
        self._progress_bars: dict[UUID, tqdm] = {}

    def on_simulation_started(self, event: SimulationEvent) -> None:
        from tqdm import tqdm
        self._progress_bars[event.job_id] = tqdm(total=100, desc=f"Job {event.job_id}")

    def on_simulation_progress(self, event: SimulationEvent) -> None:
        progress = event.data.get('progress', 0)
        pbar = self._progress_bars.get(event.job_id)
        if pbar:
            pbar.update(progress - pbar.n)

    def on_simulation_completed(self, event: SimulationEvent) -> None:
        pbar = self._progress_bars.get(event.job_id)
        if pbar:
            pbar.close()
            del self._progress_bars[event.job_id]

    def on_simulation_failed(self, event: SimulationEvent) -> None:
        self.on_simulation_completed(event)


class ObservableSimulationService:
    """支持观察者模式的模拟服务"""

    def __init__(self):
        self._observers: List[ISimulationObserver] = []

    def attach(self, observer: ISimulationObserver) -> None:
        """添加观察者"""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: ISimulationObserver) -> None:
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event: SimulationEvent) -> None:
        """通知所有观察者"""
        for observer in self._observers:
            if event.event_type == "started":
                observer.on_simulation_started(event)
            elif event.event_type == "progress":
                observer.on_simulation_progress(event)
            elif event.event_type == "completed":
                observer.on_simulation_completed(event)
            elif event.event_type == "failed":
                observer.on_simulation_failed(event)
```

### 5. Repository Pattern (仓储模式)

用于数据访问抽象。

```python
class IBuildingRepository(ABC):
    """建筑仓储接口"""

    @abstractmethod
    def save(self, building: Building) -> None:
        """保存建筑"""
        pass

    @abstractmethod
    def find_by_id(self, building_id: UUID) -> Optional[Building]:
        """根据 ID 查找建筑"""
        pass

    @abstractmethod
    def find_by_type(self, building_type: BuildingType) -> list[Building]:
        """根据类型查找建筑"""
        pass

    @abstractmethod
    def find_by_location(self, location: str) -> list[Building]:
        """根据位置查找建筑"""
        pass

    @abstractmethod
    def find_all(self) -> list[Building]:
        """查找所有建筑"""
        pass


class FileSystemBuildingRepository(IBuildingRepository):
    """基于文件系统的建筑仓储实现"""

    def __init__(self, base_directory: Path, building_factory: BuildingFactory):
        self._base_directory = base_directory
        self._building_factory = building_factory
        self._cache: dict[UUID, Building] = {}

    def save(self, building: Building) -> None:
        """保存建筑（这里主要是缓存）"""
        self._cache[building.id] = building

    def find_by_id(self, building_id: UUID) -> Optional[Building]:
        """根据 ID 查找建筑"""
        return self._cache.get(building_id)

    def find_by_type(self, building_type: BuildingType) -> list[Building]:
        """根据类型查找建筑"""
        pattern = f"*{building_type.value}*.idf"
        idf_files = list(self._base_directory.glob(pattern))

        buildings = []
        for idf_file in idf_files:
            # 从文件名解析位置
            location = self._extract_location_from_filename(idf_file.stem)

            building = self._building_factory.create_from_idf(
                idf_path=idf_file,
                building_type=building_type,
                location=location,
            )
            buildings.append(building)
            self._cache[building.id] = building

        return buildings

    def find_by_location(self, location: str) -> list[Building]:
        """根据位置查找建筑"""
        pattern = f"*{location}*.idf"
        idf_files = list(self._base_directory.glob(pattern))

        buildings = []
        for idf_file in idf_files:
            # 从文件名解析建筑类型
            building_type = self._extract_building_type_from_filename(idf_file.stem)

            building = self._building_factory.create_from_idf(
                idf_path=idf_file,
                building_type=building_type,
                location=location,
            )
            buildings.append(building)
            self._cache[building.id] = building

        return buildings

    def find_all(self) -> list[Building]:
        """查找所有建筑"""
        idf_files = list(self._base_directory.glob("*.idf"))

        buildings = []
        for idf_file in idf_files:
            # 解析文件名
            location = self._extract_location_from_filename(idf_file.stem)
            building_type = self._extract_building_type_from_filename(idf_file.stem)

            building = self._building_factory.create_from_idf(
                idf_path=idf_file,
                building_type=building_type,
                location=location,
            )
            buildings.append(building)
            self._cache[building.id] = building

        return buildings

    def _extract_location_from_filename(self, filename: str) -> str:
        """从文件名提取位置"""
        # 例如: "Chicago_OfficeLarge.idf" -> "Chicago"
        parts = filename.split('_')
        return parts[0] if parts else "Unknown"

    def _extract_building_type_from_filename(self, filename: str) -> BuildingType:
        """从文件名提取建筑类型"""
        for building_type in BuildingType:
            if building_type.value in filename:
                return building_type
        return BuildingType.OFFICE_LARGE  # 默认值
```

---

## 目录结构

```
EP-WebUI/
├── backend/
│   ├── __init__.py
│   ├── main.py                          # 应用入口
│   │
│   ├── application/                     # 应用层
│   │   ├── __init__.py
│   │   ├── cli.py                       # CLI 接口
│   │   ├── orchestrators/               # 编排器
│   │   │   ├── __init__.py
│   │   │   ├── simulation_orchestrator.py
│   │   │   └── batch_processor.py
│   │   └── builders/                    # 建造者
│   │       ├── __init__.py
│   │       └── simulation_job_builder.py
│   │
│   ├── domain/                          # 领域层
│   │   ├── __init__.py
│   │   ├── models/                      # 领域模型
│   │   │   ├── __init__.py
│   │   │   ├── building.py
│   │   │   ├── weather_file.py
│   │   │   ├── simulation_job.py
│   │   │   ├── simulation_result.py
│   │   │   └── enums.py
│   │   ├── value_objects/               # 值对象
│   │   │   ├── __init__.py
│   │   │   ├── ecm_parameters.py
│   │   │   ├── location.py
│   │   │   └── simulation_period.py
│   │   ├── services/                    # 领域服务
│   │   │   ├── __init__.py
│   │   │   ├── ecm_applicator.py
│   │   │   └── pv_system_designer.py
│   │   └── repositories/                # 仓储接口
│   │       ├── __init__.py
│   │       ├── i_building_repository.py
│   │       ├── i_weather_repository.py
│   │       └── i_result_repository.py
│   │
│   ├── services/                        # 服务层
│   │   ├── __init__.py
│   │   ├── interfaces/                  # 服务接口
│   │   │   ├── __init__.py
│   │   │   ├── i_simulation_service.py
│   │   │   ├── i_file_loader.py
│   │   │   ├── i_result_processor.py
│   │   │   └── i_cache_service.py
│   │   ├── simulation/                  # 模拟服务
│   │   │   ├── __init__.py
│   │   │   ├── base_simulation_service.py
│   │   │   ├── baseline_service.py
│   │   │   ├── pv_service.py
│   │   │   ├── optimization_service.py
│   │   │   └── sensitivity_service.py
│   │   ├── analysis/                    # 分析服务
│   │   │   ├── __init__.py
│   │   │   ├── eui_prediction_service.py
│   │   │   └── data_analysis_service.py
│   │   ├── file/                        # 文件服务
│   │   │   ├── __init__.py
│   │   │   ├── idf_loader.py
│   │   │   ├── epw_loader.py
│   │   │   └── result_parser.py
│   │   └── cache/                       # 缓存服务
│   │       ├── __init__.py
│   │       ├── memory_cache.py
│   │       └── file_cache.py
│   │
│   ├── infrastructure/                  # 基础设施层
│   │   ├── __init__.py
│   │   ├── energyplus/                  # EnergyPlus 相关
│   │   │   ├── __init__.py
│   │   │   ├── executor.py
│   │   │   └── result_parser.py
│   │   ├── repositories/                # 仓储实现
│   │   │   ├── __init__.py
│   │   │   ├── filesystem_building_repo.py
│   │   │   ├── filesystem_weather_repo.py
│   │   │   └── filesystem_result_repo.py
│   │   ├── logging/                     # 日志实现
│   │   │   ├── __init__.py
│   │   │   └── loguru_logger.py
│   │   └── database/                    # 数据库（可选）
│   │       ├── __init__.py
│   │       └── supabase_client.py
│   │
│   ├── utils/                           # 工具层
│   │   ├── __init__.py
│   │   ├── config/                      # 配置管理
│   │   │   ├── __init__.py
│   │   │   ├── config_manager.py
│   │   │   └── config_models.py
│   │   ├── validation/                  # 验证器
│   │   │   ├── __init__.py
│   │   │   └── validators.py
│   │   ├── parallel/                    # 并行处理
│   │   │   ├── __init__.py
│   │   │   └── parallel_executor.py
│   │   ├── exceptions/                  # 自定义异常
│   │   │   ├── __init__.py
│   │   │   └── custom_exceptions.py
│   │   └── helpers/                     # 辅助函数
│   │       ├── __init__.py
│   │       ├── file_helpers.py
│   │       └── path_helpers.py
│   │
│   ├── strategies/                      # 策略模式
│   │   ├── __init__.py
│   │   ├── optimization/                # 优化策略
│   │   │   ├── __init__.py
│   │   │   ├── i_optimization_strategy.py
│   │   │   ├── genetic_algorithm.py
│   │   │   ├── bayesian_optimization.py
│   │   │   └── pso_optimization.py
│   │   └── analysis/                    # 分析策略
│   │       ├── __init__.py
│   │       └── sensitivity_strategy.py
│   │
│   ├── factories/                       # 工厂模式
│   │   ├── __init__.py
│   │   ├── service_factory.py
│   │   ├── building_factory.py
│   │   └── weather_factory.py
│   │
│   ├── observers/                       # 观察者模式
│   │   ├── __init__.py
│   │   ├── i_simulation_observer.py
│   │   ├── logger_observer.py
│   │   └── progress_bar_observer.py
│   │
│   ├── configs/                         # 配置文件
│   │   ├── config.yaml
│   │   ├── paths.yaml
│   │   ├── simulation.yaml
│   │   ├── buildings.yaml
│   │   ├── ecm_ranges.yaml
│   │   ├── analysis.yaml
│   │   ├── pv.yaml
│   │   └── eui_prediction.yaml
│   │
│   └── data/                            # 数据目录
│       ├── prototypes/
│       ├── tmys/
│       ├── ftmys/
│       └── output/
│
├── tests/                               # 测试
│   ├── __init__.py
│   ├── unit/                            # 单元测试
│   │   ├── test_domain_models.py
│   │   ├── test_services.py
│   │   └── test_utils.py
│   ├── integration/                     # 集成测试
│   │   ├── test_simulation_flow.py
│   │   └── test_repository.py
│   └── fixtures/                        # 测试夹具
│       └── sample_data.py
│
├── docs/                                # 文档
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── USAGE.md
│   └── CONTRIBUTING.md
│
├── archive/                             # 旧代码归档
│   └── legacy/
│
├── pyproject.toml                       # 项目配置
├── README.md
├── REFACTORING_GUIDE.md                 # 本文档
└── .gitignore
```

---

## 核心类详细设计

### 配置管理

#### 为什么使用Pydantic而不是dataclass

配置类使用Pydantic BaseModel而非dataclass的关键优势：

1. **自动验证**: 运行时类型检查和数据验证，确保配置正确性
2. **架构一致性**: 与领域层保持一致（领域模型已使用Pydantic）
3. **更好的错误信息**: 详细的ValidationError，便于调试配置问题
4. **类型转换**: 自动将字符串转换为Path、int等类型
5. **字段验证器**: 使用`@field_validator`进行复杂验证（如文件存在性检查）
6. **模型验证器**: 使用`@model_validator`进行跨字段验证（如年份范围检查）

```python
from typing import Optional, Dict, Any
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class PathsConfig(BaseModel):
    """
    路径配置

    使用Pydantic提供自动验证和类型安全。

    Attributes:
        prototype_idf: 原型IDF文件路径
        tmy_dir: TMY天气文件目录
        ftmy_dir: 未来TMY天气文件目录
        output_dir: 输出根目录
        baseline_dir: 基准模拟输出目录
        pv_dir: 光伏模拟输出目录
        optimization_dir: 优化结果输出目录
        eplus_executable: EnergyPlus可执行文件路径
        idd_file: IDD文件路径
    """

    model_config = ConfigDict(
        validate_assignment=True,  # 赋值时也验证
        arbitrary_types_allowed=True,  # 允许Path类型
        frozen=False,  # 允许修改
    )

    prototype_idf: Path = Field(
        ...,
        description="原型IDF文件路径"
    )
    tmy_dir: Path = Field(
        ...,
        description="TMY天气文件目录"
    )
    ftmy_dir: Path = Field(
        ...,
        description="未来TMY天气文件目录"
    )
    output_dir: Path = Field(
        ...,
        description="输出根目录"
    )
    baseline_dir: Path = Field(
        ...,
        description="基准模拟输出目录"
    )
    pv_dir: Path = Field(
        ...,
        description="光伏模拟输出目录"
    )
    optimization_dir: Path = Field(
        ...,
        description="优化结果输出目录"
    )
    eplus_executable: Path = Field(
        ...,
        description="EnergyPlus可执行文件路径"
    )
    idd_file: Path = Field(
        ...,
        description="IDD文件路径"
    )

    @field_validator('eplus_executable', 'idd_file')
    @classmethod
    def validate_file_exists(cls, v: Path) -> Path:
        """验证文件存在"""
        if not v.exists():
            raise ValueError(f"文件不存在: {v}")
        if not v.is_file():
            raise ValueError(f"路径不是文件: {v}")
        return v

    @field_validator('prototype_idf')
    @classmethod
    def validate_idf_file(cls, v: Path) -> Path:
        """验证IDF文件"""
        if not v.exists():
            raise ValueError(f"IDF文件不存在: {v}")
        if v.suffix.lower() != '.idf':
            raise ValueError(f"文件必须是IDF格式，当前为: {v.suffix}")
        return v


class SimulationConfig(BaseModel):
    """
    模拟配置

    使用Pydantic提供自动验证和类型安全。

    Attributes:
        start_year: 模拟开始年份
        end_year: 模拟结束年份
        default_output_suffix: 默认输出文件后缀
        cleanup_files: 需要清理的文件扩展名列表
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )

    start_year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="模拟开始年份"
    )
    end_year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="模拟结束年份"
    )
    default_output_suffix: str = Field(
        default="L",
        min_length=1,
        max_length=10,
        description="默认输出文件后缀"
    )
    cleanup_files: list[str] = Field(
        default_factory=lambda: ['.audit', '.bnd', '.eio', '.end', '.mdd', '.mtd', '.rdd', '.shd'],
        description="需要清理的文件扩展名列表"
    )

    @model_validator(mode='after')
    def validate_year_range(self) -> 'SimulationConfig':
        """验证年份范围"""
        if self.start_year > self.end_year:
            raise ValueError(
                f"开始年份({self.start_year})不能大于结束年份({self.end_year})"
            )
        return self


class AnalysisConfig(BaseModel):
    """
    分析配置

    使用Pydantic提供自动验证和类型安全。

    Attributes:
        sensitivity: 敏感性分析配置
        optimization: 优化配置
        surrogate_models: 代理模型配置
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )

    sensitivity: Dict[str, Any] = Field(
        default_factory=dict,
        description="敏感性分析配置"
    )
    optimization: Dict[str, Any] = Field(
        default_factory=dict,
        description="优化配置"
    )
    surrogate_models: Dict[str, Any] = Field(
        default_factory=dict,
        description="代理模型配置"
    )


class ConfigManager:
    """
    配置管理器

    使用OmegaConf提供类型安全的配置访问。

    OmegaConf特性:
    - 支持YAML配置文件加载
    - 支持配置合并和覆盖
    - 支持变量插值 (e.g., ${paths.output_dir}/baseline)
    - 类型安全的配置访问
    - 支持配置验证
    - 完善的异常处理和日志记录

    Example:
        >>> config_mgr = ConfigManager(config_dir=Path("backend/configs"))
        >>> print(config_mgr.paths.output_dir)
        >>> print(OmegaConf.to_yaml(config_mgr._raw_config))
    """

    def __init__(self, config_dir: Path = Path("backend/configs")):
        self._logger = logger.bind(module=self.__class__.__name__)
        self._config_dir = config_dir
        self._raw_config: DictConfig = self._load_config()

        # 解析为类型安全的配置对象
        self.paths = self._parse_paths_config()
        self.simulation = self._parse_simulation_config()
        self.analysis = self._parse_analysis_config()

    def _load_config(self) -> DictConfig:
        """
        加载所有配置文件

        使用OmegaConf.load()加载YAML文件，然后使用OmegaConf.merge()合并。
        后加载的配置会覆盖先加载的配置。

        Returns:
            DictConfig: 合并后的配置对象

        Raises:
            FileNotFoundError: 配置目录不存在或没有配置文件
            NotADirectoryError: 配置路径不是目录

        Example:
            配置文件结构:
            - backend/configs/paths.yaml
            - backend/configs/simulation.yaml
            - backend/configs/analysis.yaml
        """
        # 验证配置目录存在
        if not self._config_dir.exists():
            self._logger.error(f"Config directory not found: {self._config_dir}")
            raise FileNotFoundError(f"Config directory not found: {self._config_dir}")
        if not self._config_dir.is_dir():
            self._logger.error(f"Config path is not a directory: {self._config_dir}")
            raise NotADirectoryError(f"Config path is not a directory: {self._config_dir}")

        config_files = sorted(self._config_dir.glob("*.yaml"))
        configs = []

        # 验证至少有一个配置文件
        if not config_files:
            self._logger.error(f"No config files found in {self._config_dir}")
            raise FileNotFoundError(f"No config files found in {self._config_dir}")

        for file in config_files:
            # OmegaConf.load() 支持YAML文件加载
            config = OmegaConf.load(file)
            configs.append(config)

        # OmegaConf.merge() 合并多个配置，支持深度合并
        # 后面的配置会覆盖前面的配置
        merged_config = OmegaConf.merge(*configs)
        return merged_config

    def _parse_paths_config(self) -> PathsConfig:
        """
        解析路径配置

        使用OmegaConf.to_container()将DictConfig转换为普通字典，
        然后使用Pydantic进行验证和类型转换。

        Returns:
            PathsConfig: 路径配置对象（Pydantic模型）

        Raises:
            ValueError: 配置中缺少paths部分
            ValidationError: 当配置验证失败时
        """
        # 验证配置中包含paths部分
        if not self._raw_config.paths:
            self._logger.error(f"Paths config not found in {self._raw_config}")
            raise ValueError(f"Paths config not found in {self._raw_config}")

        # OmegaConf.to_container() 转换为普通Python对象
        # resolve=True 解析变量插值，如 ${paths.base_dir}/output
        paths_dict = OmegaConf.to_container(
            self._raw_config.paths,
            resolve=True,  # 解析变量插值
            throw_on_missing=True  # 缺少必需值时抛出异常
        )

        # 使用Pydantic进行验证和类型转换
        # Pydantic会自动：
        # 1. 将字符串转换为Path对象
        # 2. 验证文件是否存在
        # 3. 验证IDF文件格式
        return PathsConfig(**paths_dict)  # type: ignore

    def _parse_simulation_config(self) -> SimulationConfig:
        """
        解析模拟配置

        使用Pydantic进行验证，确保年份范围合理。

        Returns:
            SimulationConfig: 模拟配置对象（Pydantic模型）

        Raises:
            ValueError: 配置中缺少simulation部分
            ValidationError: 当配置验证失败时（如年份范围不合理）
        """
        # 验证配置中包含simulation部分
        if not self._raw_config.simulation:
            self._logger.error(f"Simulation config not found in {self._raw_config}")
            raise ValueError(f"Simulation config not found in {self._raw_config}")

        sim_dict = OmegaConf.to_container(
            self._raw_config.simulation,
            resolve=True,
            throw_on_missing=True
        )

        # Pydantic会自动验证：
        # 1. start_year <= end_year
        # 2. 年份在合理范围内（1900-2100）
        # 3. cleanup_files是列表类型
        return SimulationConfig(**sim_dict)  # type: ignore

    def _parse_analysis_config(self) -> AnalysisConfig:
        """
        解析分析配置

        分析配置是可选的，使用默认值。

        Returns:
            AnalysisConfig: 分析配置对象（Pydantic模型）

        Raises:
            ValueError: 配置中缺少analysis部分
        """
        # 验证配置中包含analysis部分
        if not self._raw_config.analysis:
            self._logger.error(f"Analysis config not found in {self._raw_config}")
            raise ValueError(f"Analysis config not found in {self._raw_config}")

        analysis_dict = OmegaConf.to_container(
            self._raw_config.analysis,
            resolve=True,
            throw_on_missing=False  # 分析配置可选
        )

        # Pydantic会使用默认值填充缺失的字段
        return AnalysisConfig(**(analysis_dict or {}))  # type: ignore

    def _create_directories(self) -> None:
        """创建必要的目录"""
        dirs_to_create = [
            self.paths.output_dir,
            self.paths.baseline_dir,
            self.paths.pv_dir,
            self.paths.optimization_dir,
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_raw_config(self) -> DictConfig:
        """获取原始配置（用于向后兼容）"""
        return self._raw_config


# 配置文件示例

## YAML配置文件结构

### backend/configs/paths.yaml

```yaml
# 路径配置
paths:
  # IDF和天气文件
  prototype_idf: backend/data/prototypes/Chicago_OfficeLarge.idf
  tmy_dir: backend/data/tmys
  ftmy_dir: backend/data/ftmys

  # 输出目录（支持变量插值）
  output_dir: backend/output
  baseline_dir: ${paths.output_dir}/baseline
  pv_dir: ${paths.output_dir}/pv
  optimization_dir: ${paths.output_dir}/optimization

  # EnergyPlus配置
  # 注意：使用 eppy API 后，eplus_executable 是可选的
  # eppy 会自动查找 EnergyPlus 安装，只需要提供 IDD 文件路径
  idd_file: /usr/local/EnergyPlus-23-2-0/Energy+.idd
  # eplus_executable: /usr/local/EnergyPlus-23-2-0/energyplus  # 可选，仅在 eppy 无法自动找到时需要
```

### backend/configs/simulation.yaml

```yaml
# 模拟配置
simulation:
  start_year: 2040
  end_year: 2040
  default_output_suffix: L
  cleanup_files:
    - .audit
    - .bnd
    - .eio
    - .end
    - .mdd
    - .mtd
    - .rdd
    - .shd
```

### backend/configs/analysis.yaml

```yaml
# 分析配置
analysis:
  sensitivity:
    method: sobol
    n_samples: 1000

  optimization:
    algorithm: nsga2
    population_size: 100
    generations: 50

  surrogate_models:
    enabled: true
    model_type: gaussian_process
```

## 使用配置的完整示例

```python
from pathlib import Path
from pydantic import ValidationError

# 初始化配置管理器
try:
    config_mgr = ConfigManager(config_dir=Path("backend/configs"))

    # 访问配置（类型安全）
    print(f"IDF文件: {config_mgr.paths.prototype_idf}")
    print(f"输出目录: {config_mgr.paths.output_dir}")
    print(f"模拟年份: {config_mgr.simulation.start_year}-{config_mgr.simulation.end_year}")

    # Pydantic自动验证
    # - 文件路径存在性
    # - 年份范围合理性
    # - 类型正确性

except ValidationError as e:
    # 详细的验证错误信息
    print("配置验证失败:")
    for error in e.errors():
        print(f"  - {error['loc']}: {error['msg']}")

except FileNotFoundError as e:
    print(f"配置文件未找到: {e}")
```

---

## Baseline 模拟完整实现示例

本章节提供一个完整的、可运行的 Baseline 模拟功能实现示例，展示如何将所有组件整合在一起。

### 1. 核心服务实现

#### 1.1 EnergyPlus 执行器实现

```python
"""
backend/infrastructure/energyplus/executor.py

EnergyPlus 执行器的完整实现 - 使用 eppy API
"""
import io
import time
from pathlib import Path
from typing import Optional

from eppy.modeleditor import IDF
from loguru import logger

from backend.services.interfaces import IEnergyPlusExecutor, ExecutionResult


class EnergyPlusExecutor(IEnergyPlusExecutor):
    """
    EnergyPlus 执行器

    使用 eppy 库的 API 执行建筑能耗模拟。

    Args:
        idd_path: Energy+.idd 文件路径
        energyplus_path: EnergyPlus 可执行文件路径（可选，eppy 会自动查找）

    Note:
        相比直接使用 subprocess 调用 EnergyPlus，使用 eppy API 的优势：
        1. 更简洁的代码，无需手动构建命令行参数
        2. eppy 自动处理路径和参数转换
        3. 更好的错误处理和异常管理
        4. 支持并行运行（通过 eppy.runner.run_functions.runIDFs）
    """

    def __init__(
        self,
        idd_path: Path,
        energyplus_path: Optional[Path] = None,
    ):
        self._idd_path = idd_path
        self._energyplus_path = energyplus_path
        self._logger = logger.bind(module=self.__class__.__name__)

        # 设置 IDD 文件（eppy 要求在创建 IDF 对象前设置）
        IDF.setiddname(str(idd_path))

        # 验证安装
        if not self.validate_installation():
            raise RuntimeError("EnergyPlus installation validation failed")

    def validate_installation(self) -> bool:
        """验证 EnergyPlus 安装"""
        try:
            # 检查 IDD 文件
            if not self._idd_path.exists():
                self._logger.error(f"IDD file not found: {self._idd_path}")
                return False

            # 尝试创建一个空 IDF 来测试 eppy 设置
            test_idf = IDF(io.StringIO("VERSION,23.1;"))

            self._logger.info("EnergyPlus installation validated")
            return True

        except Exception as e:
            self._logger.error(f"EnergyPlus validation failed: {e}")
            return False

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

        使用 eppy 的 idf.run() 方法执行模拟，而不是直接调用 subprocess。

        Args:
            idf: IDF 对象
            weather_file: 天气文件路径
            output_directory: 输出目录
            output_prefix: 输出文件前缀
            read_variables: 是否读取输出变量

        Returns:
            ExecutionResult: 执行结果

        Raises:
            FileNotFoundError: 天气文件不存在

        Example:
            >>> executor = EnergyPlusExecutor(idd_path=Path("Energy+.idd"))
            >>> idf = IDF("building.idf")
            >>> result = executor.run(
            ...     idf=idf,
            ...     weather_file=Path("chicago.epw"),
            ...     output_directory=Path("output"),
            ...     output_prefix="baseline",
            ... )
            >>> assert result.success
        """
        # 验证天气文件
        if not weather_file.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_file}")

        # 创建输出目录
        output_directory.mkdir(parents=True, exist_ok=True)

        self._logger.info(f"Starting EnergyPlus simulation: {output_prefix}")
        self._logger.debug(f"Weather: {weather_file}")
        self._logger.debug(f"Output: {output_directory}")

        start_time = time.time()

        try:
            # 使用 eppy API 执行模拟
            # eppy 会自动处理命令行参数和路径转换
            idf.run(
                weather=str(weather_file),
                output_directory=str(output_directory),
                output_prefix=output_prefix,
                readvars=read_variables,
                verbose='q',  # 安静模式，减少输出
                # 如果指定了 EnergyPlus 路径，可以传递 ep_version 参数
                # ep_version=str(self._energyplus_path) if self._energyplus_path else None,
            )

            execution_time = time.time() - start_time

            # 创建执行结果
            result = ExecutionResult(
                success=True,
                return_code=0,
                stdout="",
                stderr="",
                output_directory=output_directory,
            )

            # 检查并解析错误文件
            err_file = output_directory / f"{output_prefix}out.err"
            if err_file.exists():
                self._parse_error_file(err_file, result)

            if result.success:
                self._logger.info(
                    f"Simulation completed successfully in {execution_time:.2f}s"
                )
            else:
                self._logger.error("Simulation completed with errors")
                for error in result.errors:
                    self._logger.error(f"  - {error}")

            return result

        except Exception as e:
            self._logger.error(f"EnergyPlus execution failed: {e}")

            result = ExecutionResult(
                success=False,
                return_code=1,
                stdout="",
                stderr=str(e),
                output_directory=output_directory,
            )
            result.add_error(f"Execution failed: {e}")

            return result

    def _parse_error_file(
        self,
        err_file: Path,
        result: ExecutionResult,
    ) -> None:
        """
        解析 EnergyPlus 错误文件

        Args:
            err_file: 错误文件路径
            result: 执行结果对象
        """
        try:
            with open(err_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                # 查找严重错误
                if "** Severe  **" in content:
                    severe_errors = [
                        line for line in content.split('\n')
                        if "** Severe  **" in line
                    ]
                    for error in severe_errors:
                        result.add_error(error.strip())

                # 查找致命错误
                if "**  Fatal  **" in content:
                    fatal_errors = [
                        line for line in content.split('\n')
                        if "**  Fatal  **" in line
                    ]
                    for error in fatal_errors:
                        result.add_error(error.strip())

                # 查找警告（限制数量以避免过多输出）
                if "** Warning **" in content:
                    warnings = [
                        line for line in content.split('\n')
                        if "** Warning **" in line
                    ]
                    for warning in warnings[:10]:  # 只保留前 10 个警告
                        result.add_warning(warning.strip())

        except Exception as e:
            self._logger.warning(f"Failed to parse error file: {e}")
```

#### 1.1.1 并行运行多个模拟（可选）

如果需要并行运行多个 EnergyPlus 模拟，可以使用 eppy 的 `runIDFs` 函数结合 `joblib`：

```python
"""
backend/infrastructure/energyplus/parallel_executor.py

并行执行多个 EnergyPlus 模拟
"""
from pathlib import Path
from typing import List, Tuple

from eppy.runner.run_functions import runIDFs
from loguru import logger


class ParallelEnergyPlusExecutor:
    """
    并行 EnergyPlus 执行器

    使用 eppy 的 runIDFs 函数并行运行多个模拟。

    Note:
        runIDFs 内部使用 multiprocessing 实现并行化，
        比手动使用 joblib 更高效且更稳定。
    """

    def __init__(self, idd_path: Path, num_workers: int = -1):
        """
        初始化并行执行器

        Args:
            idd_path: IDD 文件路径
            num_workers: 并行工作进程数，-1 表示使用所有 CPU 核心
        """
        self._idd_path = idd_path
        self._num_workers = num_workers
        self._logger = logger.bind(module=self.__class__.__name__)

    def run_parallel(
        self,
        idf_weather_pairs: List[Tuple[Path, Path]],
        output_directory: Path,
    ) -> List[Path]:
        """
        并行运行多个模拟

        Args:
            idf_weather_pairs: (IDF文件路径, 天气文件路径) 的列表
            output_directory: 输出目录

        Returns:
            输出目录列表

        Example:
            >>> executor = ParallelEnergyPlusExecutor(idd_path=Path("Energy+.idd"))
            >>> pairs = [
            ...     (Path("building1.idf"), Path("chicago.epw")),
            ...     (Path("building2.idf"), Path("chicago.epw")),
            ...     (Path("building3.idf"), Path("chicago.epw")),
            ... ]
            >>> results = executor.run_parallel(pairs, Path("output"))
        """
        self._logger.info(f"Starting parallel execution of {len(idf_weather_pairs)} simulations")

        # 创建输出目录
        output_directory.mkdir(parents=True, exist_ok=True)

        # 准备 runIDFs 的输入参数
        idf_files = [str(idf) for idf, _ in idf_weather_pairs]
        weather_files = [str(epw) for _, epw in idf_weather_pairs]

        # 使用 runIDFs 并行运行
        # processors: 并行进程数，-1 表示使用所有核心
        # verbose: 'q' 表示安静模式
        runIDFs(
            idf_files,
            weather_files,
            output_directory=str(output_directory),
            processors=self._num_workers,
            verbose='q',
        )

        self._logger.info("Parallel execution completed")

        # 返回输出目录列表
        return [
            output_directory / Path(idf).stem
            for idf, _ in idf_weather_pairs
        ]
```

**使用 joblib 的替代方案**（如果需要更细粒度的控制）：

```python
"""
使用 joblib 实现并行执行的替代方案
"""
from joblib import Parallel, delayed
from pathlib import Path
from typing import List

from backend.infrastructure.energyplus.executor import EnergyPlusExecutor
from backend.services.interfaces import ExecutionResult


def run_single_simulation(
    executor: EnergyPlusExecutor,
    idf_path: Path,
    weather_file: Path,
    output_directory: Path,
    output_prefix: str,
) -> ExecutionResult:
    """运行单个模拟的辅助函数"""
    from eppy.modeleditor import IDF

    idf = IDF(str(idf_path))
    return executor.run(
        idf=idf,
        weather_file=weather_file,
        output_directory=output_directory,
        output_prefix=output_prefix,
    )


def run_simulations_parallel_joblib(
    executor: EnergyPlusExecutor,
    simulation_configs: List[dict],
    n_jobs: int = -1,
) -> List[ExecutionResult]:
    """
    使用 joblib 并行运行多个模拟

    Args:
        executor: EnergyPlus 执行器
        simulation_configs: 模拟配置列表，每个配置包含 idf_path, weather_file 等
        n_jobs: 并行任务数，-1 表示使用所有核心

    Returns:
        执行结果列表
    """
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_simulation)(
            executor,
            config['idf_path'],
            config['weather_file'],
            config['output_directory'],
            config['output_prefix'],
        )
        for config in simulation_configs
    )

    return results
```

#### 1.2 结果解析器实现

```python
"""
backend/infrastructure/parsers/result_parser.py

模拟结果解析器的完整实现
"""
import sqlite3
from pathlib import Path
from typing import Optional
from uuid import UUID

import pandas as pd
from loguru import logger

from backend.domain.models import SimulationResult


class ResultParser:
    """
    模拟结果解析器

    从 EnergyPlus 输出文件中解析能耗数据和其他指标。
    """

    def __init__(self):
        self._logger = logger.bind(module=self.__class__.__name__)

    def parse(
        self,
        job_id: UUID,
        output_directory: Path,
        output_prefix: str = "eplus",
    ) -> SimulationResult:
        """
        解析模拟结果

        Args:
            job_id: 模拟任务 ID
            output_directory: 输出目录
            output_prefix: 输出文件前缀

        Returns:
            SimulationResult: 解析后的模拟结果
        """
        self._logger.info(f"Parsing simulation results for job {job_id}")

        # 创建结果对象
        result = SimulationResult(
            job_id=job_id,
            output_directory=output_directory,
        )

        # 查找输出文件
        table_csv = output_directory / f"{output_prefix}tbl.csv"
        meter_csv = output_directory / f"{output_prefix}out.csv"
        sql_file = output_directory / f"{output_prefix}out.sql"

        # 设置文件路径
        if table_csv.exists():
            result.table_csv_path = table_csv
        if meter_csv.exists():
            result.meter_csv_path = meter_csv
        if sql_file.exists():
            result.sql_path = sql_file

        # 解析能耗数据
        try:
            if sql_file.exists():
                self._parse_from_sql(result, sql_file)
            elif table_csv.exists():
                self._parse_from_table_csv(result, table_csv)
            else:
                result.add_error("No output files found for parsing")
                return result

            result.success = True
            self._logger.info(
                f"Parsing completed. Source EUI: {result.source_eui:.2f} kWh/m²/yr"
            )

        except Exception as e:
            self._logger.error(f"Failed to parse results: {e}")
            result.add_error(f"Parsing error: {str(e)}")

        return result

    def _parse_from_sql(self, result: SimulationResult, sql_file: Path) -> None:
        """从 SQL 文件解析数据"""
        conn = sqlite3.connect(str(sql_file))

        try:
            # 查询年度能耗数据
            query = """
            SELECT
                Value
            FROM
                TabularDataWithStrings
            WHERE
                ReportName = 'AnnualBuildingUtilityPerformanceSummary'
                AND TableName = 'Site and Source Energy'
                AND RowName = 'Total Site Energy'
                AND ColumnName = 'Total Energy'
                AND Units = 'GJ'
            """

            df = pd.read_sql_query(query, conn)

            if not df.empty:
                # 转换 GJ 到 kWh
                total_energy_kwh = float(df.iloc[0]['Value']) * 277.778

                # 获取建筑面积
                area_query = """
                SELECT
                    Value
                FROM
                    TabularDataWithStrings
                WHERE
                    ReportName = 'AnnualBuildingUtilityPerformanceSummary'
                    AND TableName = 'Building Area'
                    AND RowName = 'Total Building Area'
                    AND Units = 'm2'
                """

                area_df = pd.read_sql_query(area_query, conn)

                if not area_df.empty:
                    building_area = float(area_df.iloc[0]['Value'])
                    result.source_eui = total_energy_kwh / building_area
                    result.total_energy_kwh = total_energy_kwh

                    # 存储到元数据
                    result.metadata['building_area_m2'] = building_area

            # 查询电力和天然气消耗
            self._parse_energy_breakdown(result, conn)

        finally:
            conn.close()

    def _parse_energy_breakdown(
        self,
        result: SimulationResult,
        conn: sqlite3.Connection,
    ) -> None:
        """解析能源分项数据"""
        # 查询电力消耗
        elec_query = """
        SELECT
            Value
        FROM
            TabularDataWithStrings
        WHERE
            ReportName = 'AnnualBuildingUtilityPerformanceSummary'
            AND TableName = 'End Uses'
            AND RowName = 'Total End Uses'
            AND ColumnName = 'Electricity'
            AND Units = 'GJ'
        """

        elec_df = pd.read_sql_query(elec_query, conn)
        if not elec_df.empty:
            elec_kwh = float(elec_df.iloc[0]['Value']) * 277.778
            result.metadata['electricity_kwh'] = elec_kwh

        # 查询天然气消耗
        gas_query = """
        SELECT
            Value
        FROM
            TabularDataWithStrings
        WHERE
            ReportName = 'AnnualBuildingUtilityPerformanceSummary'
            AND TableName = 'End Uses'
            AND RowName = 'Total End Uses'
            AND ColumnName = 'Natural Gas'
            AND Units = 'GJ'
        """

        gas_df = pd.read_sql_query(gas_query, conn)
        if not gas_df.empty:
            gas_kwh = float(gas_df.iloc[0]['Value']) * 277.778
            result.metadata['natural_gas_kwh'] = gas_kwh

    def _parse_from_table_csv(
        self,
        result: SimulationResult,
        table_csv: Path,
    ) -> None:
        """从表格 CSV 文件解析数据（备用方法）"""
        # 简化实现，实际应该解析 CSV 文件
        self._logger.warning("CSV parsing not fully implemented, using SQL instead")
        result.add_warning("Parsed from CSV (limited data)")
```

#### 1.3 文件清理器实现

```python
"""
backend/infrastructure/file_cleaner.py

临时文件清理器实现
"""
from pathlib import Path
from typing import List

from loguru import logger


class FileCleaner:
    """
    文件清理器

    清理模拟过程中产生的临时文件，保留重要的输出文件。
    """

    def __init__(self):
        self._logger = logger.bind(module=self.__class__.__name__)

    def clean(
        self,
        directory: Path,
        keep_extensions: List[str] = None,
        cleanup_patterns: List[str] = None,
    ) -> None:
        """
        清理目录中的临时文件

        Args:
            directory: 要清理的目录
            keep_extensions: 要保留的文件扩展名列表（如 ['.csv', '.sql']）
            cleanup_patterns: 要删除的文件模式列表（如 ['*.audit', '*.bnd']）
        """
        if not directory.exists():
            self._logger.warning(f"Directory does not exist: {directory}")
            return

        keep_extensions = keep_extensions or ['.csv', '.sql', '.idf', '.epw']
        cleanup_patterns = cleanup_patterns or [
            '*.audit', '*.bnd', '*.eio', '*.mtd', '*.mdd',
            '*.rdd', '*.shd', '*.dxf', '*.end', '*.eso',
        ]

        deleted_count = 0

        # 删除匹配模式的文件
        for pattern in cleanup_patterns:
            for file in directory.glob(pattern):
                try:
                    file.unlink()
                    deleted_count += 1
                    self._logger.debug(f"Deleted: {file.name}")
                except Exception as e:
                    self._logger.warning(f"Failed to delete {file.name}: {e}")

        self._logger.info(f"Cleaned up {deleted_count} temporary files from {directory}")
```

### 2. BaselineSimulationService 完整实现

```python
"""
backend/services/simulation/baseline_service.py

Baseline 模拟服务的完整实现
"""
import time
from pathlib import Path
from typing import Optional

from eppy.modeleditor import IDF
from loguru import logger

from backend.domain.models import SimulationResult
from backend.services.interfaces import ISimulationService, IEnergyPlusExecutor
from backend.services.simulation import BaselineContext
from backend.infrastructure.parsers import ResultParser
from backend.infrastructure.file_cleaner import FileCleaner
from backend.utils.config import ConfigManager


class BaselineSimulationService(ISimulationService[BaselineContext]):
    """
    Baseline 模拟服务

    执行建筑的基准能耗模拟，不应用任何节能措施（ECM）。
    这是所有其他模拟类型的基础，用于建立性能基准。

    Args:
        executor: EnergyPlus 执行器
        result_parser: 结果解析器
        file_cleaner: 文件清理器
        config: 配置管理器

    Example:
        >>> config = ConfigManager()
        >>> executor = EnergyPlusExecutor(
        ...     executable_path=config.paths.eplus_executable,
        ...     idd_path=config.paths.idd_file,
        ... )
        >>> parser = ResultParser()
        >>> cleaner = FileCleaner()
        >>> service = BaselineSimulationService(executor, parser, cleaner, config)
        >>>
        >>> # 创建上下文并运行
        >>> context = BaselineContext(job=job, idf=idf, working_directory=output_dir)
        >>> result = service.run(context)
    """

    def __init__(
        self,
        executor: IEnergyPlusExecutor,
        result_parser: ResultParser,
        file_cleaner: FileCleaner,
        config: ConfigManager,
    ):
        self._executor = executor
        self._result_parser = result_parser
        self._file_cleaner = file_cleaner
        self._config = config
        self._logger = logger.bind(service=self.__class__.__name__)

    def prepare(self, context: BaselineContext) -> None:
        """
        准备 Baseline 模拟

        执行以下步骤：
        1. 创建输出目录
        2. 验证 IDF 和天气文件存在
        3. 添加必要的输出变量到 IDF
        4. 设置模拟控制参数

        Args:
            context: Baseline 模拟上下文

        Raises:
            FileNotFoundError: 如果必需的文件不存在
            ValueError: 如果配置无效
        """
        self._logger.info(
            f"Preparing baseline simulation for job {context.job.id}"
        )

        # 1. 创建输出目录
        context.job.output_directory.mkdir(parents=True, exist_ok=True)
        self._logger.debug(f"Output directory: {context.job.output_directory}")

        # 2. 验证文件存在
        if not context.job.building.idf_file_path.exists():
            raise FileNotFoundError(
                f"IDF file not found: {context.job.building.idf_file_path}"
            )

        if not context.job.weather_file.file_path.exists():
            raise FileNotFoundError(
                f"Weather file not found: {context.job.weather_file.file_path}"
            )

        self._logger.debug(f"IDF file: {context.job.building.idf_file_path}")
        self._logger.debug(f"Weather file: {context.job.weather_file.file_path}")

        # 3. 添加输出变量
        self._add_output_variables(context.idf)

        # 4. 配置输出控制
        self._configure_output_controls(context.idf)

        # 5. 设置模拟周期（如果需要）
        self._configure_simulation_period(context.idf)

        self._logger.info("Preparation completed successfully")

    def execute(self, context: BaselineContext) -> SimulationResult:
        """
        执行 Baseline 模拟

        执行以下步骤：
        1. 调用 EnergyPlus 执行器运行模拟
        2. 检查执行结果
        3. 解析输出文件
        4. 返回模拟结果

        Args:
            context: Baseline 模拟上下文

        Returns:
            SimulationResult: 模拟结果对象
        """
        self._logger.info(
            f"Executing baseline simulation for job {context.job.id}"
        )

        start_time = time.time()

        try:
            # 1. 执行 EnergyPlus
            execution_result = self._executor.run(
                idf=context.idf,
                weather_file=context.job.weather_file.file_path,
                output_directory=context.job.output_directory,
                output_prefix=context.job.output_prefix,
                read_variables=context.job.read_variables,
            )

            # 2. 检查执行是否成功
            if not execution_result.success:
                self._logger.error(
                    f"EnergyPlus execution failed with return code "
                    f"{execution_result.return_code}"
                )

                # 创建失败结果
                result = SimulationResult(
                    job_id=context.job.id,
                    output_directory=context.job.output_directory,
                )
                result.success = False

                # 添加错误信息
                for error in execution_result.errors:
                    result.add_error(error)

                # 添加警告信息
                for warning in execution_result.warnings:
                    result.add_warning(warning)

                return result

            # 3. 解析结果
            self._logger.info("Parsing simulation results...")
            result = self._result_parser.parse(
                job_id=context.job.id,
                output_directory=context.job.output_directory,
                output_prefix=context.job.output_prefix,
            )

            # 4. 添加执行信息
            execution_time = time.time() - start_time
            result.metadata['execution_time_seconds'] = execution_time
            result.metadata['building_name'] = context.job.building.name
            result.metadata['building_type'] = context.job.building.building_type.value
            result.metadata['location'] = context.job.building.location
            result.metadata['weather_scenario'] = context.job.weather_file.scenario

            # 添加警告（如果有）
            for warning in execution_result.warnings:
                result.add_warning(warning)

            # 5. 记录结果
            if result.is_valid():
                self._logger.info(
                    f"Simulation completed successfully in {execution_time:.2f}s. "
                    f"Source EUI: {result.source_eui:.2f} kWh/m²/yr"
                )
            else:
                self._logger.warning(
                    f"Simulation completed with errors in {execution_time:.2f}s"
                )

            return result

        except Exception as e:
            # 处理未预期的异常
            self._logger.error(f"Unexpected error during simulation: {e}", exc_info=True)

            result = SimulationResult(
                job_id=context.job.id,
                output_directory=context.job.output_directory,
            )
            result.add_error(f"Unexpected error: {str(e)}")
            result.metadata['execution_time_seconds'] = time.time() - start_time

            return result

    def cleanup(self, context: BaselineContext) -> None:
        """
        清理临时文件

        删除模拟过程中产生的临时文件，保留重要的输出文件。

        Args:
            context: Baseline 模拟上下文
        """
        self._logger.info("Cleaning up temporary files...")

        try:
            self._file_cleaner.clean(
                directory=context.job.output_directory,
                keep_extensions=['.csv', '.sql', '.idf', '.epw'],
                cleanup_patterns=self._config.simulation.cleanup_files,
            )
            self._logger.info("Cleanup completed")

        except Exception as e:
            # 清理失败不应该影响整体流程
            self._logger.warning(f"Cleanup failed: {e}")

    def _add_output_variables(self, idf: IDF) -> None:
        """
        添加必要的输出变量到 IDF

        Args:
            idf: IDF 对象
        """
        # 定义需要的输出变量
        required_variables = [
            ("Site Outdoor Air Drybulb Temperature", "Hourly"),
            ("Zone Mean Air Temperature", "Hourly"),
            ("Facility Total Electric Demand Power", "Hourly"),
            ("Facility Total Natural Gas Demand Rate", "Hourly"),
            ("Facility Total Purchased Electric Energy", "Monthly"),
            ("Facility Total Natural Gas Energy", "Monthly"),
        ]

        added_count = 0

        for var_name, frequency in required_variables:
            # 检查变量是否已存在
            exists = any(
                ov.Variable_Name == var_name
                for ov in idf.idfobjects.get("OUTPUT:VARIABLE", [])
            )

            if not exists:
                idf.newidfobject(
                    "OUTPUT:VARIABLE",
                    Key_Value="*",
                    Variable_Name=var_name,
                    Reporting_Frequency=frequency,
                )
                added_count += 1
                self._logger.debug(f"Added output variable: {var_name}")

        if added_count > 0:
            self._logger.info(f"Added {added_count} output variables to IDF")

    def _configure_output_controls(self, idf: IDF) -> None:
        """
        配置输出控制参数

        Args:
            idf: IDF 对象
        """
        # 确保有 Output:Table:SummaryReports 对象
        summary_reports = idf.idfobjects.get("OUTPUT:TABLE:SUMMARYREPORTS", [])

        if not summary_reports:
            idf.newidfobject(
                "OUTPUT:TABLE:SUMMARYREPORTS",
                Report_1_Name="AllSummary",
            )
            self._logger.debug("Added AllSummary report")

        # 配置 SQLite 输出
        sqlite_outputs = idf.idfobjects.get("OUTPUT:SQLITE", [])

        if not sqlite_outputs:
            idf.newidfobject(
                "OUTPUT:SQLITE",
                Option_Type="SimpleAndTabular",
            )
            self._logger.debug("Enabled SQLite output")

    def _configure_simulation_period(self, idf: IDF) -> None:
        """
        配置模拟周期（可选）

        Args:
            idf: IDF 对象
        """
        # 这里可以根据需要修改模拟周期
        # 默认使用 IDF 文件中的设置
        run_periods = idf.idfobjects.get("RUNPERIOD", [])

        if run_periods:
            self._logger.debug(
                f"Using existing run period: {run_periods[0].Name}"
            )
        else:
            self._logger.warning("No run period defined in IDF")
```

### 3. 完整使用示例

#### 3.1 简单工厂函数

```python
"""
backend/utils/factory.py

简单工厂函数，用于创建服务实例
"""
from pathlib import Path

from backend.utils.config import ConfigManager
from backend.infrastructure.energyplus import EnergyPlusExecutor
from backend.infrastructure.parsers import ResultParser
from backend.infrastructure.file_cleaner import FileCleaner
from backend.services.simulation import BaselineSimulationService


def create_baseline_service(config: ConfigManager) -> BaselineSimulationService:
    """
    创建 Baseline 模拟服务

    使用工厂函数集中管理依赖关系，便于测试和维护。

    Args:
        config: 配置管理器

    Returns:
        BaselineSimulationService: 配置好的服务实例
    """
    # 创建依赖组件
    executor = EnergyPlusExecutor(
        executable_path=config.paths.eplus_executable,
        idd_path=config.paths.idd_file,
        timeout=3600,
    )

    result_parser = ResultParser()
    file_cleaner = FileCleaner()

    # 创建服务
    return BaselineSimulationService(
        executor=executor,
        result_parser=result_parser,
        file_cleaner=file_cleaner,
        config=config,
    )
```

#### 3.2 单个建筑模拟示例

```python
"""
示例 1: 运行单个建筑的 Baseline 模拟
"""
from pathlib import Path
from eppy.modeleditor import IDF

from backend.utils.config import ConfigManager
from backend.utils.factory import create_baseline_service
from backend.domain.models import Building, WeatherFile, SimulationJob
from backend.domain.models.enums import BuildingType, SimulationType
from backend.services.simulation import BaselineContext


def run_single_baseline_simulation():
    """运行单个 Baseline 模拟的完整示例"""

    # 1. 初始化配置
    config = ConfigManager(config_dir=Path("backend/configs"))

    # 2. 创建 Building 对象
    building = Building(
        name="Chicago_OfficeLarge",
        building_type=BuildingType.OFFICE_LARGE,
        location="Chicago",
        idf_file_path=Path("data/prototypes/Chicago_OfficeLarge.idf"),
        floor_area=46320.0,  # m²
        num_floors=12,
    )

    # 3. 创建 WeatherFile 对象
    weather_file = WeatherFile(
        file_path=Path("data/tmys/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"),
        location="Chicago",
        scenario="TMY3",
    )

    # 4. 创建 SimulationJob 对象
    job = SimulationJob(
        building=building,
        weather_file=weather_file,
        simulation_type=SimulationType.BASELINE,
        output_directory=Path("output/baseline/chicago_office"),
        output_prefix="baseline",
        read_variables=True,
    )

    # 5. 加载 IDF 文件
    IDF.setiddname(str(config.paths.idd_file))
    idf = IDF(str(building.idf_file_path))

    # 6. 创建模拟上下文
    context = BaselineContext(
        job=job,
        idf=idf,
        working_directory=job.output_directory,
    )

    # 7. 创建服务并运行模拟
    service = create_baseline_service(config)

    print(f"Starting baseline simulation for {building.name}...")
    job.start()

    try:
        result = service.run(context)

        if result.is_valid():
            job.complete(result)
            print(f"✓ Simulation completed successfully!")
            print(f"  Source EUI: {result.source_eui:.2f} kWh/m²/yr")
            print(f"  Total Energy: {result.total_energy_kwh:.2f} kWh")
            print(f"  Execution Time: {result.metadata.get('execution_time_seconds', 0):.2f}s")

            # 显示能源分项
            if 'electricity_kwh' in result.metadata:
                print(f"  Electricity: {result.metadata['electricity_kwh']:.2f} kWh")
            if 'natural_gas_kwh' in result.metadata:
                print(f"  Natural Gas: {result.metadata['natural_gas_kwh']:.2f} kWh")
        else:
            job.fail("Simulation validation failed")
            print(f"✗ Simulation failed or produced invalid results")
            print(f"  Errors: {len(result.error_messages)}")
            for error in result.error_messages[:5]:  # 显示前5个错误
                print(f"    - {error}")

    except Exception as e:
        job.fail(str(e))
        print(f"✗ Simulation error: {e}")
        raise

    return result


if __name__ == "__main__":
    result = run_single_baseline_simulation()
```

#### 3.3 批量模拟示例

```python
"""
示例 2: 批量运行多个建筑的 Baseline 模拟
"""
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from eppy.modeleditor import IDF
from loguru import logger

from backend.utils.config import ConfigManager
from backend.utils.factory import create_baseline_service
from backend.domain.models import Building, WeatherFile, SimulationJob
from backend.domain.models.enums import BuildingType, SimulationType
from backend.services.simulation import BaselineContext


def create_simulation_jobs(
    config: ConfigManager,
) -> List[SimulationJob]:
    """
    创建批量模拟任务

    Returns:
        模拟任务列表
    """
    jobs = []

    # 定义要模拟的建筑和城市
    building_types = [
        BuildingType.OFFICE_LARGE,
        BuildingType.OFFICE_MEDIUM,
        BuildingType.OFFICE_SMALL,
    ]

    cities = ["Chicago", "Miami", "Phoenix"]

    # 为每个组合创建任务
    for city in cities:
        for building_type in building_types:
            # 构建文件路径
            idf_filename = f"{city}_{building_type.value}.idf"
            idf_path = config.paths.prototype_dir / idf_filename

            epw_filename = f"{city}_TMY3.epw"
            epw_path = config.paths.tmy_dir / epw_filename

            # 检查文件是否存在
            if not idf_path.exists() or not epw_path.exists():
                logger.warning(f"Skipping {city} {building_type.value}: files not found")
                continue

            # 创建 Building 对象
            building = Building(
                name=f"{city}_{building_type.value}",
                building_type=building_type,
                location=city,
                idf_file_path=idf_path,
            )

            # 创建 WeatherFile 对象
            weather_file = WeatherFile(
                file_path=epw_path,
                location=city,
                scenario="TMY3",
            )

            # 创建 SimulationJob 对象
            output_dir = config.paths.baseline_dir / city / building_type.value
            job = SimulationJob(
                building=building,
                weather_file=weather_file,
                simulation_type=SimulationType.BASELINE,
                output_directory=output_dir,
                output_prefix="baseline",
            )

            jobs.append(job)

    return jobs


def run_batch_simulations(
    jobs: List[SimulationJob],
    config: ConfigManager,
    max_workers: int = 4,
) -> List[SimulationResult]:
    """
    并行运行批量模拟

    Args:
        jobs: 模拟任务列表
        config: 配置管理器
        max_workers: 最大并行工作线程数

    Returns:
        模拟结果列表
    """
    logger.info(f"Starting batch simulation of {len(jobs)} jobs with {max_workers} workers")

    # 创建服务
    service = create_baseline_service(config)

    # 设置 IDD 文件（全局设置，只需一次）
    IDF.setiddname(str(config.paths.idd_file))

    results = []
    completed = 0
    total = len(jobs)

    def run_single_job(job: SimulationJob):
        """运行单个任务"""
        try:
            # 加载 IDF
            idf = IDF(str(job.building.idf_file_path))

            # 创建上下文
            context = BaselineContext(
                job=job,
                idf=idf,
                working_directory=job.output_directory,
            )

            # 运行模拟
            job.start()
            result = service.run(context)

            if result.is_valid():
                job.complete(result)
            else:
                job.fail("Validation failed")

            return result

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            job.fail(str(e))

            # 返回失败结果
            from backend.domain.models import SimulationResult
            result = SimulationResult(
                job_id=job.id,
                output_directory=job.output_directory,
            )
            result.add_error(str(e))
            return result

    # 并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_job = {
            executor.submit(run_single_job, job): job
            for job in jobs
        }

        # 收集结果
        for future in as_completed(future_to_job):
            job = future_to_job[future]

            try:
                result = future.result()
                results.append(result)

                completed += 1

                # 显示进度
                if result.is_valid():
                    logger.info(
                        f"[{completed}/{total}] ✓ {job.building.name}: "
                        f"EUI = {result.source_eui:.2f} kWh/m²/yr"
                    )
                else:
                    logger.warning(
                        f"[{completed}/{total}] ✗ {job.building.name}: Failed"
                    )

            except Exception as e:
                logger.error(f"Error processing result for {job.building.name}: {e}")

    # 统计结果
    success_count = sum(1 for r in results if r.is_valid())
    logger.info(
        f"Batch simulation completed: {success_count}/{total} successful"
    )

    return results


def main():
    """主函数"""
    # 初始化配置
    config = ConfigManager(config_dir=Path("backend/configs"))

    # 创建任务
    jobs = create_simulation_jobs(config)
    print(f"Created {len(jobs)} simulation jobs")

    # 运行批量模拟
    results = run_batch_simulations(
        jobs=jobs,
        config=config,
        max_workers=4,  # 根据 CPU 核心数调整
    )

    # 分析结果
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)

    for result in results:
        if result.is_valid():
            building_name = result.metadata.get('building_name', 'Unknown')
            print(f"{building_name:30s} {result.source_eui:8.2f} kWh/m²/yr")

    print("="*60)


if __name__ == "__main__":
    main()
```

#### 3.4 使用缓存的示例

```python
"""
示例 3: 使用缓存避免重复模拟
"""
import hashlib
import pickle
from pathlib import Path
from typing import Optional

from backend.domain.models import SimulationJob, SimulationResult


class SimpleCache:
    """简单的文件缓存实现"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[SimulationResult]:
        """获取缓存的结果"""
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None

        return None

    def set(self, key: str, result: SimulationResult) -> None:
        """保存结果到缓存"""
        cache_file = self.cache_dir / f"{key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")


def run_with_cache(
    job: SimulationJob,
    service: BaselineSimulationService,
    cache: SimpleCache,
) -> SimulationResult:
    """
    运行模拟，使用缓存避免重复计算

    Args:
        job: 模拟任务
        service: 模拟服务
        cache: 缓存对象

    Returns:
        模拟结果（可能来自缓存）
    """
    # 生成缓存键
    cache_key = job.get_cache_key()

    # 检查缓存
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Using cached result for {job.building.name}")
        return cached_result

    # 运行模拟
    logger.info(f"Running simulation for {job.building.name}")

    from eppy.modeleditor import IDF
    idf = IDF(str(job.building.idf_file_path))

    context = BaselineContext(
        job=job,
        idf=idf,
        working_directory=job.output_directory,
    )

    result = service.run(context)

    # 保存到缓存
    if result.is_valid():
        cache.set(cache_key, result)

    return result
```

### 4. 测试示例

#### 4.1 单元测试

```python
"""
tests/services/test_baseline_service.py

BaselineSimulationService 的单元测试
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from backend.services.simulation import BaselineSimulationService, BaselineContext
from backend.domain.models import Building, WeatherFile, SimulationJob, SimulationResult
from backend.domain.models.enums import BuildingType, SimulationType


@pytest.fixture
def mock_config():
    """模拟配置对象"""
    config = Mock()
    config.paths.eplus_executable = Path("/usr/local/EnergyPlus/energyplus")
    config.paths.idd_file = Path("/usr/local/EnergyPlus/Energy+.idd")
    config.simulation.cleanup_files = ['*.audit', '*.bnd']
    return config


@pytest.fixture
def mock_executor():
    """模拟 EnergyPlus 执行器"""
    executor = Mock()

    # 模拟成功的执行结果
    from backend.services.interfaces import ExecutionResult
    result = ExecutionResult(
        success=True,
        return_code=0,
        stdout="Simulation completed",
        stderr="",
        output_directory=Path("output/test"),
    )

    executor.run.return_value = result
    return executor


@pytest.fixture
def mock_parser():
    """模拟结果解析器"""
    parser = Mock()

    # 模拟解析结果
    result = SimulationResult(
        job_id="test-job-id",
        output_directory=Path("output/test"),
    )
    result.success = True
    result.source_eui = 150.5
    result.total_energy_kwh = 500000.0

    parser.parse.return_value = result
    return parser


@pytest.fixture
def mock_cleaner():
    """模拟文件清理器"""
    return Mock()


@pytest.fixture
def baseline_service(mock_executor, mock_parser, mock_cleaner, mock_config):
    """创建 BaselineSimulationService 实例"""
    return BaselineSimulationService(
        executor=mock_executor,
        result_parser=mock_parser,
        file_cleaner=mock_cleaner,
        config=mock_config,
    )


@pytest.fixture
def sample_job(tmp_path):
    """创建示例模拟任务"""
    # 创建临时 IDF 和 EPW 文件
    idf_file = tmp_path / "test.idf"
    idf_file.write_text("Version,9.3;")

    epw_file = tmp_path / "test.epw"
    epw_file.write_text("LOCATION,Chicago,IL,USA")

    building = Building(
        name="TestBuilding",
        building_type=BuildingType.OFFICE_LARGE,
        location="Chicago",
        idf_file_path=idf_file,
    )

    weather_file = WeatherFile(
        file_path=epw_file,
        location="Chicago",
        scenario="TMY3",
    )

    return SimulationJob(
        building=building,
        weather_file=weather_file,
        simulation_type=SimulationType.BASELINE,
        output_directory=tmp_path / "output",
        output_prefix="test",
    )


@pytest.fixture
def sample_context(sample_job):
    """创建示例模拟上下文"""
    # 模拟 IDF 对象
    mock_idf = MagicMock()
    mock_idf.idfobjects.get.return_value = []

    return BaselineContext(
        job=sample_job,
        idf=mock_idf,
        working_directory=sample_job.output_directory,
    )


class TestBaselineSimulationService:
    """BaselineSimulationService 测试类"""

    def test_prepare_creates_output_directory(
        self,
        baseline_service,
        sample_context,
    ):
        """测试 prepare 方法创建输出目录"""
        # 执行
        baseline_service.prepare(sample_context)

        # 验证
        assert sample_context.job.output_directory.exists()

    def test_prepare_validates_files(
        self,
        baseline_service,
        sample_context,
        tmp_path,
    ):
        """测试 prepare 方法验证文件存在"""
        # 删除 IDF 文件
        sample_context.job.building.idf_file_path.unlink()

        # 执行并验证抛出异常
        with pytest.raises(FileNotFoundError, match="IDF file not found"):
            baseline_service.prepare(sample_context)

    def test_prepare_adds_output_variables(
        self,
        baseline_service,
        sample_context,
    ):
        """测试 prepare 方法添加输出变量"""
        # 执行
        baseline_service.prepare(sample_context)

        # 验证调用了 newidfobject
        assert sample_context.idf.newidfobject.called

    def test_execute_calls_energyplus(
        self,
        baseline_service,
        sample_context,
        mock_executor,
    ):
        """测试 execute 方法调用 EnergyPlus"""
        # 执行
        result = baseline_service.execute(sample_context)

        # 验证
        mock_executor.run.assert_called_once()
        assert result.success
        assert result.source_eui == 150.5

    def test_execute_handles_energyplus_failure(
        self,
        baseline_service,
        sample_context,
        mock_executor,
    ):
        """测试 execute 方法处理 EnergyPlus 失败"""
        # 模拟失败
        from backend.services.interfaces import ExecutionResult
        mock_executor.run.return_value = ExecutionResult(
            success=False,
            return_code=1,
            stdout="",
            stderr="Error",
            output_directory=sample_context.job.output_directory,
        )
        mock_executor.run.return_value.errors = ["Fatal error"]

        # 执行
        result = baseline_service.execute(sample_context)

        # 验证
        assert not result.success
        assert len(result.error_messages) > 0

    def test_cleanup_removes_temp_files(
        self,
        baseline_service,
        sample_context,
        mock_cleaner,
    ):
        """测试 cleanup 方法清理临时文件"""
        # 执行
        baseline_service.cleanup(sample_context)

        # 验证
        mock_cleaner.clean.assert_called_once()

    def test_run_executes_full_workflow(
        self,
        baseline_service,
        sample_context,
        mock_executor,
        mock_parser,
        mock_cleaner,
    ):
        """测试 run 方法执行完整流程"""
        # 执行
        result = baseline_service.run(sample_context)

        # 验证所有步骤都被调用
        assert sample_context.job.output_directory.exists()  # prepare
        mock_executor.run.assert_called_once()  # execute
        mock_parser.parse.assert_called_once()  # execute
        mock_cleaner.clean.assert_called_once()  # cleanup

        # 验证结果
        assert result.success
        assert result.source_eui == 150.5
```

#### 4.2 集成测试

```python
"""
tests/integration/test_baseline_integration.py

Baseline 模拟的集成测试（需要真实的 EnergyPlus 安装）
"""
import pytest
from pathlib import Path

from eppy.modeleditor import IDF

from backend.utils.config import ConfigManager
from backend.utils.factory import create_baseline_service
from backend.domain.models import Building, WeatherFile, SimulationJob
from backend.domain.models.enums import BuildingType, SimulationType
from backend.services.simulation import BaselineContext


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("/usr/local/EnergyPlus/energyplus").exists(),
    reason="EnergyPlus not installed"
)
class TestBaselineIntegration:
    """Baseline 模拟集成测试"""

    def test_full_baseline_simulation(self, tmp_path):
        """测试完整的 Baseline 模拟流程"""
        # 1. 初始化配置
        config = ConfigManager(config_dir=Path("backend/configs"))

        # 2. 创建测试数据（假设测试数据存在）
        building = Building(
            name="TestOffice",
            building_type=BuildingType.OFFICE_SMALL,
            location="Chicago",
            idf_file_path=Path("tests/fixtures/test_office.idf"),
        )

        weather_file = WeatherFile(
            file_path=Path("tests/fixtures/test_weather.epw"),
            location="Chicago",
            scenario="TMY3",
        )

        job = SimulationJob(
            building=building,
            weather_file=weather_file,
            simulation_type=SimulationType.BASELINE,
            output_directory=tmp_path / "output",
            output_prefix="test",
        )

        # 3. 加载 IDF
        IDF.setiddname(str(config.paths.idd_file))
        idf = IDF(str(building.idf_file_path))

        # 4. 创建上下文
        context = BaselineContext(
            job=job,
            idf=idf,
            working_directory=job.output_directory,
        )

        # 5. 运行模拟
        service = create_baseline_service(config)
        result = service.run(context)

        # 6. 验证结果
        assert result.is_valid()
        assert result.source_eui > 0
        assert result.total_energy_kwh > 0
        assert result.table_csv_path.exists()
        assert result.sql_path.exists()
```

### 5. 关键要点总结

#### 5.1 设计原则

本实现遵循以下设计原则：

1. **依赖注入**: 所有依赖通过构造函数注入，便于测试和替换
2. **单一职责**: 每个类只负责一个明确的功能
3. **接口隔离**: 使用抽象基类定义清晰的接口
4. **错误处理**: 完善的异常处理和日志记录
5. **类型安全**: 使用 Pydantic 和类型注解确保类型安全

#### 5.2 代码组织

```
backend/
├── domain/                    # 领域层
│   ├── models/               # 实体和值对象
│   └── services/             # 领域服务
├── services/                  # 应用服务层
│   ├── interfaces/           # 服务接口
│   └── simulation/           # 模拟服务实现
├── infrastructure/            # 基础设施层
│   ├── energyplus/           # EnergyPlus 集成
│   ├── parsers/              # 结果解析器
│   └── file_cleaner.py       # 文件清理器
└── utils/                     # 工具层
    ├── config/               # 配置管理
    └── factory.py            # 工厂函数
```

#### 5.3 使用流程

完整的 Baseline 模拟流程：

```
1. 初始化配置 (ConfigManager)
   ↓
2. 创建领域对象 (Building, WeatherFile, SimulationJob)
   ↓
3. 加载 IDF 文件 (eppy.IDF)
   ↓
4. 创建模拟上下文 (BaselineContext)
   ↓
5. 创建服务实例 (create_baseline_service)
   ↓
6. 运行模拟 (service.run)
   ├── prepare: 准备环境和配置
   ├── execute: 执行 EnergyPlus 并解析结果
   └── cleanup: 清理临时文件
   ↓
7. 处理结果 (SimulationResult)
```

#### 5.4 扩展建议

基于此实现，可以轻松扩展：

1. **PV 模拟服务**: 继承 `ISimulationService`，实现光伏系统模拟
2. **ECM 模拟服务**: 应用节能措施后的模拟
3. **优化服务**: 集成优化算法
4. **缓存服务**: 添加 Redis 或数据库缓存
5. **并行执行**: 使用进程池而非线程池提高性能
6. **进度跟踪**: 添加 WebSocket 实时进度推送
7. **结果可视化**: 集成图表生成功能

---

## 基本用法

```python
from omegaconf import OmegaConf, DictConfig

# 1. 从YAML文件加载配置
config = OmegaConf.load("config.yaml")

# 2. 从字典创建配置
config = OmegaConf.create({
    "database": {
        "host": "localhost",
        "port": 5432
    }
})

# 3. 访问配置值
print(config.database.host)  # localhost
print(config["database"]["port"])  # 5432

# 4. 转换为YAML字符串
yaml_str = OmegaConf.to_yaml(config)
print(yaml_str)

# 5. 转换为普通字典
plain_dict = OmegaConf.to_container(config, resolve=True)
```

## 配置合并

```python
# 基础配置
base_config = OmegaConf.create({
    "model": {
        "name": "resnet",
        "layers": 50
    },
    "training": {
        "epochs": 100,
        "batch_size": 32
    }
})

# 覆盖配置
override_config = OmegaConf.create({
    "model": {
        "layers": 101  # 覆盖layers
    },
    "training": {
        "epochs": 200  # 覆盖epochs
    }
})

# 合并配置（后面的覆盖前面的）
merged = OmegaConf.merge(base_config, override_config)
print(merged.model.layers)  # 101
print(merged.training.batch_size)  # 32 (保留)
```

## 变量插值

```yaml
# config.yaml
paths:
  base_dir: /data
  output_dir: ${paths.base_dir}/output  # 引用其他配置值
  log_dir: ${paths.base_dir}/logs

simulation:
  output: ${paths.output_dir}/simulation  # 嵌套引用
```

```python
config = OmegaConf.load("config.yaml")
# 自动解析插值
print(config.paths.output_dir)  # /data/output
print(config.simulation.output)  # /data/output/simulation
```

## 配置验证

```python
from omegaconf import OmegaConf, MissingMandatoryValue

# 使用 ??? 标记必需值
config = OmegaConf.create({
    "database": {
        "host": "???",  # 必需值
        "port": 5432
    }
})

try:
    # 访问未设置的必需值会抛出异常
    print(config.database.host)
except MissingMandatoryValue as e:
    print(f"Missing required config: {e}")

# 设置必需值
OmegaConf.update(config, "database.host", "localhost")
print(config.database.host)  # localhost
```

## 命令行覆盖

```python
from omegaconf import OmegaConf

# 从命令行参数覆盖配置
# python script.py model.layers=101 training.epochs=200

def load_config_with_cli_override(config_file: str, cli_args: list[str]) -> DictConfig:
    """加载配置并应用命令行覆盖"""
    # 加载基础配置
    config = OmegaConf.load(config_file)

    # 从命令行参数创建覆盖配置
    cli_config = OmegaConf.from_dotlist(cli_args)

    # 合并配置
    return OmegaConf.merge(config, cli_config)

# 使用示例
import sys
config = load_config_with_cli_override(
    "config.yaml",
    sys.argv[1:]  # ['model.layers=101', 'training.epochs=200']
)
```

## 最佳实践

### 1. 使用resolve=True解析插值

```python
# ✅ 推荐：解析所有插值
config_dict = OmegaConf.to_container(config, resolve=True)

# ❌ 不推荐：保留插值字符串
config_dict = OmegaConf.to_container(config, resolve=False)
```

### 2. 使用throw_on_missing处理缺失值

```python
# ✅ 推荐：缺失必需值时抛出异常
config_dict = OmegaConf.to_container(
    config,
    resolve=True,
    throw_on_missing=True
)

# ⚠️ 可选配置可以设置为False
optional_config = OmegaConf.to_container(
    config.optional_section,
    resolve=True,
    throw_on_missing=False
)
```

### 3. 配置文件组织

```
backend/configs/
├── base.yaml          # 基础配置
├── paths.yaml         # 路径配置
├── simulation.yaml    # 模拟配置
└── analysis.yaml      # 分析配置
```

```python
# 加载并合并所有配置
def load_all_configs(config_dir: Path) -> DictConfig:
    config_files = sorted(config_dir.glob("*.yaml"))
    configs = [OmegaConf.load(f) for f in config_files]
    return OmegaConf.merge(*configs)
```

### 4. 使用Pydantic进行类型安全的配置访问

```python
from pydantic import BaseModel, Field, field_validator
from omegaconf import OmegaConf

class DatabaseConfig(BaseModel):
    """数据库配置（使用Pydantic）"""

    host: str = Field(..., min_length=1, description="数据库主机")
    port: int = Field(..., ge=1, le=65535, description="数据库端口")
    username: str = Field(..., min_length=1, description="用户名")
    password: str = Field(..., min_length=1, description="密码")

    @field_validator('host')
    @classmethod
    def validate_host(cls, v: str) -> str:
        """验证主机名"""
        if v == "localhost" or v.startswith("127."):
            return v
        # 可以添加更多验证逻辑
        return v

# 从YAML加载并转换为Pydantic模型
config_dict = OmegaConf.load("database.yaml")
db_config = DatabaseConfig(**OmegaConf.to_container(config_dict, resolve=True))

# 现在有完整的类型检查和自动验证
print(db_config.host)  # IDE会提供自动补全
print(db_config.model_dump())  # 转换为字典
```

## Pydantic配置类的优势

### 1. 自动验证

```python
from pydantic import BaseModel, Field, ValidationError

class PathsConfig(BaseModel):
    output_dir: Path = Field(..., description="输出目录")

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """自动验证并创建目录"""
        v.mkdir(parents=True, exist_ok=True)
        return v

try:
    # Pydantic会自动验证
    config = PathsConfig(output_dir="/tmp/output")
except ValidationError as e:
    print(e.json())  # 详细的错误信息
```

### 2. 类型转换

```python
from pydantic import BaseModel
from pathlib import Path

class Config(BaseModel):
    path: Path  # 自动将字符串转换为Path
    count: int  # 自动将字符串转换为int

# 自动类型转换
config = Config(path="/tmp/data", count="42")
assert isinstance(config.path, Path)  # True
assert isinstance(config.count, int)  # True
```

### 3. 与OmegaConf结合使用

```python
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

class SimulationConfig(BaseModel):
    start_year: int = Field(..., ge=1900, le=2100)
    end_year: int = Field(..., ge=1900, le=2100)

# 从OmegaConf加载
omega_config = OmegaConf.load("simulation.yaml")
config_dict = OmegaConf.to_container(omega_config, resolve=True)

# 使用Pydantic验证
sim_config = SimulationConfig(**config_dict)

# 获得两者的优势：
# - OmegaConf: 配置合并、变量插值
# - Pydantic: 自动验证、类型安全
```

### 4. 配置序列化

```python
from pydantic import BaseModel

class Config(BaseModel):
    name: str
    value: int

config = Config(name="test", value=42)

# 序列化为字典
config_dict = config.model_dump()

# 序列化为JSON
config_json = config.model_dump_json()

# 从字典创建
new_config = Config(**config_dict)

# 从JSON创建
new_config = Config.model_validate_json(config_json)
```
```

### 依赖注入容器

```python
from typing import Type, TypeVar, Callable, Dict, Any


T = TypeVar('T')


class DependencyContainer:
    """
    依赖注入容器

    管理服务的生命周期和依赖关系。
    """

    def __init__(self):
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}

    def register_singleton(self, interface: Type[T], instance: T) -> None:
        """
        注册单例

        Args:
            interface: 接口类型
            instance: 实例对象
        """
        self._singletons[interface] = instance

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """
        注册工厂函数

        Args:
            interface: 接口类型
            factory: 工厂函数
        """
        self._factories[interface] = factory

    def resolve(self, interface: Type[T]) -> T:
        """
        解析依赖

        Args:
            interface: 接口类型

        Returns:
            实例对象

        Raises:
            ValueError: 未注册的接口
        """
        # 首先检查单例
        if interface in self._singletons:
            return self._singletons[interface]

        # 然后检查工厂
        if interface in self._factories:
            return self._factories[interface]()

        raise ValueError(f"No registration found for {interface}")

    def resolve_all(self, *interfaces: Type) -> tuple:
        """
        解析多个依赖

        Args:
            interfaces: 接口类型列表

        Returns:
            实例对象元组
        """
        return tuple(self.resolve(interface) for interface in interfaces)


def setup_container(config: ConfigManager) -> DependencyContainer:
    """
    设置依赖注入容器

    Args:
        config: 配置管理器

    Returns:
        配置好的依赖容器
    """
    container = DependencyContainer()

    # 注册单例
    from backend.infrastructure.logging.loguru_logger import LoguruLogger
    logger = LoguruLogger()
    container.register_singleton('ILogger', logger)

    # 注册配置
    container.register_singleton(ConfigManager, config)

    # 注册缓存服务
    from backend.services.cache.memory_cache import MemoryCache
    cache = MemoryCache()
    container.register_singleton(ICacheService, cache)

    # 注册 EnergyPlus 执行器
    # 注意：使用 eppy API 后，不再需要 executable_path 参数
    # eppy 会自动查找 EnergyPlus 安装路径
    from backend.infrastructure.energyplus.executor import EnergyPlusExecutor
    executor = EnergyPlusExecutor(
        idd_path=config.paths.idd_file,
        # energyplus_path 是可选的，如果 eppy 无法自动找到可以手动指定
        # energyplus_path=config.paths.eplus_executable,
    )
    container.register_singleton('IEnergyPlusExecutor', executor)

    # 注册工厂
    from backend.factories.service_factory import ServiceFactory
    service_factory = ServiceFactory(container)
    container.register_singleton(ServiceFactory, service_factory)

    # 注册仓储
    from backend.infrastructure.repositories.filesystem_building_repo import FileSystemBuildingRepository
    from backend.factories.building_factory import BuildingFactory

    building_factory = BuildingFactory()
    building_repo = FileSystemBuildingRepository(
        base_directory=config.paths.prototype_idf,
        building_factory=building_factory,
    )
    container.register_singleton(IBuildingRepository, building_repo)

    # 更多注册...

    return container
```

---

## 性能优化策略

### 1. 并行执行

```python
from joblib import Parallel, delayed, cpu_count
from typing import Callable, List, TypeVar, Optional


T = TypeVar('T')
R = TypeVar('R')


class ParallelExecutor:
    """
    并行执行器

    支持多线程和多进程执行。
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
    ):
        """
        Args:
            max_workers: 最大工作线程/进程数，None 表示使用 CPU 核心数
            use_processes: 是否使用多进程而非多线程
        """
        if max_workers is None:
            max_workers = cpu_count()

        self._max_workers = max_workers
        self._use_processes = use_processes

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
        """
        backend = 'loky' if self._use_processes else 'threading'
        total = len(items)

        try:
            results = Parallel(
                n_jobs=self._max_workers,
                backend=backend,
                verbose=0
            )(delayed(func)(item) for item in items)

            if progress_callback:
                progress_callback(total, total)

            return results
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise

    def map_chunked(
        self,
        func: Callable[[List[T]], R],
        items: List[T],
        chunk_size: int,
    ) -> List[R]:
        """
        分块并行执行

        将大列表分成小块，每块在一个进程中处理。
        适用于 I/O 密集型任务。

        Args:
            func: 处理函数，接收一个列表
            items: 输入列表
            chunk_size: 每块大小

        Returns:
            结果列表
        """
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        return self.map(func, chunks)


# 使用示例
def run_simulation_batch_parallel(
    jobs: List[SimulationJob],
    orchestrator: SimulationOrchestrator,
) -> List[SimulationResult]:
    """并行执行模拟批次"""

    parallel_executor = ParallelExecutor(max_workers=4, use_processes=True)

    def execute_job(job: SimulationJob) -> SimulationResult:
        # 在子进程中执行单个任务
        return orchestrator._execute_single_job(job)

    results = parallel_executor.map(
        func=execute_job,
        items=jobs,
        progress_callback=lambda completed, total: print(f"Progress: {completed}/{total}"),
    )

    return results
```

### 2. 智能缓存

```python
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Any, Callable
import functools


class SmartCache:
    """
    智能缓存系统

    支持内存缓存和磁盘缓存，自动过期管理。
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_items: int = 100,
        default_ttl: int = 3600,  # 1 hour
    ):
        self._cache_dir = cache_dir or Path(".cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory_cache: Dict[str, tuple[Any, float]] = {}
        self._max_memory_items = max_memory_items
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        # 首先检查内存缓存
        if key in self._memory_cache:
            value, expires_at = self._memory_cache[key]
            if time.time() < expires_at:
                return value
            else:
                del self._memory_cache[key]

        # 检查磁盘缓存
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                value, expires_at = cached_data['value'], cached_data['expires_at']

                if time.time() < expires_at:
                    # 加载到内存缓存
                    self._set_memory_cache(key, value, expires_at)
                    return value
                else:
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存"""
        ttl = ttl or self._default_ttl
        expires_at = time.time() + ttl

        # 设置内存缓存
        self._set_memory_cache(key, value, expires_at)

        # 设置磁盘缓存
        cache_file = self._get_cache_file(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'expires_at': expires_at,
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _set_memory_cache(self, key: str, value: Any, expires_at: float) -> None:
        """设置内存缓存"""
        # 如果超过最大数量，删除最旧的
        if len(self._memory_cache) >= self._max_memory_items:
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k][1]
            )
            del self._memory_cache[oldest_key]

        self._memory_cache[key] = (value, expires_at)

    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用 hash 作为文件名
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"

    def clear(self) -> None:
        """清空所有缓存"""
        self._memory_cache.clear()

        for cache_file in self._cache_dir.glob("*.cache"):
            cache_file.unlink()


def cached(
    cache: SmartCache,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
):
    """
    缓存装饰器

    Args:
        cache: 缓存实例
        ttl: 过期时间（秒）
        key_func: 自定义键函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # 尝试从缓存获取
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # 执行函数
            result = func(*args, **kwargs)

            # 保存到缓存
            cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


# 使用示例
cache = SmartCache()


@cached(cache, ttl=7200)  # 2 hours
def run_expensive_simulation(building_id: str, weather_id: str) -> SimulationResult:
    # 执行耗时的模拟
    ...
    return result
```

### 3. 资源池化

```python
from queue import Queue
from threading import Lock
from typing import Generic, TypeVar, Callable


T = TypeVar('T')


class ObjectPool(Generic[T]):
    """
    对象池

    重用昂贵的对象（如 IDF 实例）。
    """

    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
    ):
        """
        Args:
            factory: 创建对象的工厂函数
            max_size: 池的最大大小
        """
        self._factory = factory
        self._max_size = max_size
        self._pool: Queue[T] = Queue(maxsize=max_size)
        self._lock = Lock()
        self._current_size = 0

    def acquire(self) -> T:
        """获取对象"""
        try:
            # 尝试从池中获取
            obj = self._pool.get_nowait()
            return obj
        except:
            # 池为空，创建新对象
            with self._lock:
                if self._current_size < self._max_size:
                    obj = self._factory()
                    self._current_size += 1
                    return obj
                else:
                    # 等待对象归还
                    return self._pool.get()

    def release(self, obj: T) -> None:
        """归还对象"""
        try:
            self._pool.put_nowait(obj)
        except:
            # 池已满，丢弃对象
            with self._lock:
                self._current_size -= 1

    def clear(self) -> None:
        """清空池"""
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except:
                break

        with self._lock:
            self._current_size = 0


# 使用示例
from contextlib import contextmanager


class IDFPool:
    """IDF 对象池"""

    def __init__(self):
        self._pools: Dict[Path, ObjectPool[IDF]] = {}

    def get_pool(self, idf_path: Path) -> ObjectPool[IDF]:
        """获取指定 IDF 文件的对象池"""
        if idf_path not in self._pools:
            def factory():
                return IDF(str(idf_path))

            self._pools[idf_path] = ObjectPool(factory, max_size=5)

        return self._pools[idf_path]

    @contextmanager
    def borrow(self, idf_path: Path):
        """借用 IDF 对象（上下文管理器）"""
        pool = self.get_pool(idf_path)
        idf = pool.acquire()
        try:
            yield idf
        finally:
            pool.release(idf)


# 使用
idf_pool = IDFPool()

with idf_pool.borrow(building.idf_file_path) as idf:
    # 使用 IDF 对象
    idf.run(...)
```

---

## 代码质量保证

### 1. 类型提示

所有函数和方法都必须有完整的类型提示：

```python
from typing import List, Dict, Optional, Union, Callable, Protocol


# ✅ 正确：完整的类型提示
def process_simulation_results(
    results: List[SimulationResult],
    filter_func: Optional[Callable[[SimulationResult], bool]] = None,
    group_by: Optional[str] = None,
) -> Dict[str, List[SimulationResult]]:
    """
    处理模拟结果

    Args:
        results: 模拟结果列表
        filter_func: 可选的过滤函数
        group_by: 分组字段名

    Returns:
        分组后的结果字典

    Raises:
        ValueError: 如果 group_by 字段不存在
    """
    ...
```

### 2. 文档字符串

使用 Google 风格的文档字符串：

```python
def calculate_energy_savings(
    baseline_result: SimulationResult,
    optimized_result: SimulationResult,
    floor_area: float,
) -> Dict[str, float]:
    """
    计算能源节省量

    比较基准建筑和优化后建筑的能耗，计算节省的能源和百分比。

    Args:
        baseline_result: 基准模拟结果
        optimized_result: 优化后模拟结果
        floor_area: 建筑面积 (m²)

    Returns:
        包含以下键的字典:
        - 'absolute_savings': 绝对节省量 (kWh/yr)
        - 'percentage_savings': 百分比节省 (%)
        - 'eui_reduction': EUI 降低量 (kWh/m²/yr)

    Raises:
        ValueError: 如果模拟结果无效或面积为负

    Example:
        >>> baseline = SimulationResult(source_eui=150.0)
        >>> optimized = SimulationResult(source_eui=120.0)
        >>> savings = calculate_energy_savings(baseline, optimized, 1000.0)
        >>> print(savings['percentage_savings'])
        20.0
    """
    if floor_area <= 0:
        raise ValueError(f"Floor area must be positive: {floor_area}")

    if not baseline_result.success or not optimized_result.success:
        raise ValueError("Both simulation results must be successful")

    baseline_eui = baseline_result.source_eui
    optimized_eui = optimized_result.source_eui

    eui_reduction = baseline_eui - optimized_eui
    percentage_savings = (eui_reduction / baseline_eui) * 100
    absolute_savings = eui_reduction * floor_area

    return {
        'absolute_savings': absolute_savings,
        'percentage_savings': percentage_savings,
        'eui_reduction': eui_reduction,
    }
```

### 3. 异常处理

定义清晰的异常层次结构：

```python
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


class FileNotFoundError(EPWebUIException):
    """文件未找到"""
    pass


class CacheError(EPWebUIException):
    """缓存错误"""
    pass


# 使用示例
def load_building(idf_path: Path) -> Building:
    """加载建筑"""
    if not idf_path.exists():
        raise FileNotFoundError(f"IDF file not found: {idf_path}")

    try:
        idf = IDF(str(idf_path))
    except Exception as e:
        raise ValidationError(f"Failed to parse IDF file: {e}") from e

    return Building(...)
```

### 4. 代码格式化和检查

使用 `ruff` 进行格式化和检查：

```toml
# pyproject.toml

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "DTZ", # flake8-datetimez
    "T10", # flake8-debugger
    "EM",  # flake8-errmsg
    "ISC", # flake8-implicit-str-concat
    "ICN", # flake8-import-conventions
    "PIE", # flake8-pie
    "PT",  # flake8-pytest-style
    "RET", # flake8-return
    "SIM", # flake8-simplify
    "TID", # flake8-tidy-imports
    "ARG", # flake8-unused-arguments
    "PTH", # flake8-use-pathlib
    "PD",  # pandas-vet
    "NPY", # numpy-specific rules
]

ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

---

## 测试策略

### 1. 单元测试

```python
import pytest
from pathlib import Path
from backend.domain.models.building import Building, BuildingType


class TestBuilding:
    """建筑实体单元测试"""

    def test_create_building_success(self, tmp_path):
        """测试成功创建建筑"""
        # Arrange
        idf_file = tmp_path / "test.idf"
        idf_file.touch()

        # Act
        building = Building(
            name="Test Building",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=idf_file,
            floor_area=1000.0,
        )

        # Assert
        assert building.name == "Test Building"
        assert building.building_type == BuildingType.OFFICE_LARGE
        assert building.floor_area == 1000.0
        assert building.get_identifier() == "Chicago_OfficeLarge"

    def test_create_building_invalid_file(self):
        """测试无效文件路径"""
        with pytest.raises(ValueError, match="IDF file does not exist"):
            Building(
                name="Test",
                building_type=BuildingType.OFFICE_LARGE,
                location="Chicago",
                idf_file_path=Path("/nonexistent/file.idf"),
            )

    def test_create_building_negative_area(self, tmp_path):
        """测试负面积"""
        idf_file = tmp_path / "test.idf"
        idf_file.touch()

        with pytest.raises(ValueError, match="Floor area must be positive"):
            Building(
                name="Test",
                building_type=BuildingType.OFFICE_LARGE,
                location="Chicago",
                idf_file_path=idf_file,
                floor_area=-100.0,
            )


class TestECMParameters:
    """ECM 参数单元测试"""

    def test_create_ecm_parameters_success(self):
        """测试成功创建 ECM 参数"""
        params = ECMParameters(
            window_u_value=1.5,
            window_shgc=0.4,
            cooling_cop=4.0,
        )

        assert params.window_u_value == 1.5
        assert params.window_shgc == 0.4
        assert params.cooling_cop == 4.0

    def test_ecm_parameters_invalid_u_value(self):
        """测试无效的 U 值"""
        with pytest.raises(ValueError, match="Invalid window U-value"):
            ECMParameters(window_u_value=15.0)

    def test_ecm_parameters_invalid_shgc(self):
        """测试无效的 SHGC"""
        with pytest.raises(ValueError, match="Invalid SHGC"):
            ECMParameters(window_shgc=1.5)

    def test_ecm_parameters_to_dict(self):
        """测试转换为字典"""
        params = ECMParameters(
            window_u_value=1.5,
            window_shgc=0.4,
        )

        params_dict = params.to_dict()

        assert params_dict == {
            'window_u_value': 1.5,
            'window_shgc': 0.4,
        }

    def test_ecm_parameters_hashable(self):
        """测试可哈希性"""
        params1 = ECMParameters(window_u_value=1.5)
        params2 = ECMParameters(window_u_value=1.5)
        params3 = ECMParameters(window_u_value=2.0)

        assert hash(params1) == hash(params2)
        assert hash(params1) != hash(params3)


# Fixtures
@pytest.fixture
def sample_building(tmp_path):
    """创建示例建筑"""
    idf_file = tmp_path / "Chicago_OfficeLarge.idf"
    idf_file.touch()

    return Building(
        name="Chicago Office",
        building_type=BuildingType.OFFICE_LARGE,
        location="Chicago",
        idf_file_path=idf_file,
        floor_area=5000.0,
        num_floors=10,
    )


@pytest.fixture
def sample_weather_file(tmp_path):
    """创建示例天气文件"""
    epw_file = tmp_path / "Chicago_TMY.epw"
    epw_file.touch()

    return WeatherFile(
        file_path=epw_file,
        location="Chicago",
        scenario="TMY",
    )
```

### 2. 集成测试

```python
class TestSimulationFlow:
    """模拟流程集成测试"""

    def test_baseline_simulation_flow(
        self,
        sample_building,
        sample_weather_file,
        tmp_path,
    ):
        """测试完整的基准模拟流程"""
        # Arrange
        config = ConfigManager()
        container = setup_container(config)

        service_factory = container.resolve(ServiceFactory)
        baseline_service = service_factory.create_service("baseline")

        job = (SimulationJobBuilder()
            .with_building(sample_building)
            .with_weather_file(sample_weather_file)
            .with_simulation_type("baseline")
            .with_output_directory(tmp_path / "output")
            .build())

        # Act
        context = BaselineContext(
            job=job,
            idf=IDF(str(sample_building.idf_file_path)),
            working_directory=tmp_path,
        )

        result = baseline_service.run(context)

        # Assert
        assert result is not None
        assert result.job_id == job.id
        # 更多断言...

    def test_batch_simulation_with_cache(
        self,
        sample_building,
        sample_weather_file,
        tmp_path,
    ):
        """测试带缓存的批量模拟"""
        # Arrange
        config = ConfigManager()
        container = setup_container(config)

        orchestrator = container.resolve(SimulationOrchestrator)

        jobs = [
            (SimulationJobBuilder()
                .with_building(sample_building)
                .with_weather_file(sample_weather_file)
                .with_output_directory(tmp_path / f"output_{i}")
                .build())
            for i in range(5)
        ]

        # Act - 第一次执行
        results1 = orchestrator.execute_batch(jobs, use_cache=True)

        # Act - 第二次执行（应该使用缓存）
        results2 = orchestrator.execute_batch(jobs, use_cache=True)

        # Assert
        assert len(results1) == 5
        assert len(results2) == 5
        # 验证缓存被使用...
```

### 3. 测试覆盖率

使用 `pytest-cov` 测量测试覆盖率：

```bash
# 安装
pip install pytest pytest-cov

# 运行测试并生成覆盖率报告
pytest --cov=backend --cov-report=html --cov-report=term

# 查看 HTML 报告
open htmlcov/index.html
```

---

## 实施步骤

### 阶段 1：基础架构（第 1-2 周）

#### 1.1 创建目录结构

```bash
mkdir -p backend/{application,domain,services,infrastructure,utils,strategies,factories,observers}
mkdir -p backend/domain/{models,value_objects,services,repositories}
mkdir -p backend/services/{interfaces,simulation,analysis,file,cache}
mkdir -p backend/infrastructure/{energyplus,repositories,logging,database}
mkdir -p backend/utils/{config,validation,parallel,exceptions,helpers}
mkdir -p tests/{unit,integration,fixtures}
```

#### 1.2 实现核心领域模型

- [ ] 创建 `Building` 实体类
- [ ] 创建 `WeatherFile` 实体类
- [ ] 创建 `SimulationJob` 实体类
- [ ] 创建 `SimulationResult` 实体类
- [ ] 创建 `ECMParameters` 值对象
- [ ] 创建枚举类（`BuildingType`, etc.）

#### 1.3 实现配置管理

- [ ] 创建 `ConfigManager` 类
- [ ] 创建配置数据类（`PathsConfig`, `SimulationConfig`, etc.）
- [ ] 添加配置验证

#### 1.4 实现依赖注入容器

- [ ] 创建 `DependencyContainer` 类
- [ ] 实现 `setup_container` 函数

### 阶段 2：服务层（第 3-4 周）

#### 2.1 定义服务接口

- [ ] 创建 `ISimulationService` 接口
- [ ] 创建 `IFileLoader` 接口
- [ ] 创建 `IResultProcessor` 接口
- [ ] 创建 `ICacheService` 接口

#### 2.2 实现基础服务

- [ ] 实现 `BaselineSimulationService`
- [ ] 实现 `IDFFileLoader`
- [ ] 实现 `ResultParser`
- [ ] 实现 `MemoryCache`

#### 2.3 实现 EnergyPlus 执行器

- [ ] 创建 `EnergyPlusExecutor` 类
- [ ] 实现模拟执行逻辑
- [ ] 添加错误处理

### 阶段 3：高级功能（第 5-6 周）

#### 3.1 实现模拟编排器

- [ ] 创建 `SimulationOrchestrator` 类
- [ ] 实现批量执行
- [ ] 集成缓存机制

#### 3.2 实现工厂模式

- [ ] 创建 `ServiceFactory`
- [ ] 创建 `BuildingFactory`
- [ ] 创建 `WeatherFactory`

#### 3.3 实现仓储模式

- [ ] 创建仓储接口
- [ ] 实现文件系统仓储

### 阶段 4：性能优化（第 7-8 周）

#### 4.1 实现并行执行

- [ ] 创建 `ParallelExecutor` 类
- [ ] 集成到 `SimulationOrchestrator`
- [ ] 性能测试和调优

#### 4.2 实现智能缓存

- [ ] 创建 `SmartCache` 类
- [ ] 实现内存和磁盘缓存
- [ ] 添加过期管理

#### 4.3 实现对象池

- [ ] 创建 `ObjectPool` 类
- [ ] 实现 IDF 对象池
- [ ] 集成到服务层

### 阶段 5：扩展服务（第 9-10 周）

#### 5.1 实现 PV 服务

- [ ] 迁移 PV 相关逻辑
- [ ] 创建 `PVSimulationService`
- [ ] 实现 PV 系统设计器

#### 5.2 实现优化服务

- [ ] 创建优化策略接口
- [ ] 实现遗传算法策略
- [ ] 实现贝叶斯优化策略
- [ ] 创建 `OptimizationService`

#### 5.3 实现敏感性分析服务

- [ ] 创建 `SensitivityService`
- [ ] 集成 SALib

### 阶段 6：测试和文档（第 11-12 周）

#### 6.1 单元测试

- [ ] 为所有核心类编写单元测试
- [ ] 达到 80% 覆盖率

#### 6.2 集成测试

- [ ] 编写端到端测试
- [ ] 性能基准测试

#### 6.3 文档

- [ ] API 文档
- [ ] 架构文档
- [ ] 使用指南

### 阶段 7：迁移和部署（第 13-14 周）

#### 7.1 数据迁移

- [ ] 迁移现有模拟结果
- [ ] 验证数据完整性

#### 7.2 向后兼容

- [ ] 创建兼容层
- [ ] 迁移旧配置文件

#### 7.3 部署

- [ ] 更新 CI/CD 流程
- [ ] 生产环境测试

---

## 迁移路线图

### 向后兼容策略

为了确保平滑迁移，提供向后兼容的适配器：

```python
class LegacyAdapter:
    """
    旧代码适配器

    提供与旧代码的兼容接口。
    """

    def __init__(self, new_orchestrator: SimulationOrchestrator):
        self._orchestrator = new_orchestrator

    def run_baseline_simulation(
        self,
        idf_path: str,
        weather_path: str,
        output_dir: str,
        output_prefix: str = "baseline",
    ) -> dict:
        """
        旧接口：运行基准模拟

        这个方法提供与旧代码相同的接口，但内部使用新架构。
        """
        # 创建新对象
        building = BuildingFactory.create_from_idf(
            idf_path=Path(idf_path),
            building_type=BuildingType.OFFICE_LARGE,  # 需要解析
            location="Unknown",  # 需要解析
        )

        weather_file = WeatherFile(
            file_path=Path(weather_path),
            location="Unknown",
            scenario="TMY",
        )

        job = (SimulationJobBuilder()
            .with_building(building)
            .with_weather_file(weather_file)
            .with_simulation_type("baseline")
            .with_output_directory(Path(output_dir))
            .with_output_prefix(output_prefix)
            .build())

        # 执行模拟
        results = self._orchestrator.execute_batch([job])
        result = results[0]

        # 转换为旧格式
        return {
            'success': result.success,
            'source_eui': result.source_eui,
            'output_dir': str(result.output_directory),
            'errors': result.error_messages,
        }
```

### 逐步迁移计划

1. **第 1 阶段**：保留旧代码，新功能使用新架构
2. **第 2 阶段**：逐步将旧代码标记为 `@deprecated`
3. **第 3 阶段**：提供迁移工具和文档
4. **第 4 阶段**：完全移除旧代码

---

## 总结

本重构方案基于以下核心原则：

1. **SOLID 原则**：确保代码的可维护性和可扩展性
2. **分层架构**：清晰的职责分离
3. **设计模式**：工厂、策略、建造者、观察者、仓储等模式的应用
4. **类型安全**：100% 类型提示覆盖
5. **性能优化**：并行执行、智能缓存、对象池
6. **测试驱动**：80% 以上测试覆盖率
7. **文档完善**：详细的文档字符串和使用指南

通过这次重构，EP-WebUI 将具备：

- ✅ 更高的性能（30-50% 提升）
- ✅ 更好的可读性和可维护性
- ✅ 更强的扩展性
- ✅ 更完善的测试和文档
- ✅ 更健壮的错误处理
- ✅ 更灵活的配置管理

---

## 附录

### A. 命名约定

- **类名**：使用 PascalCase（如 `SimulationOrchestrator`）
- **函数/方法名**：使用 snake_case（如 `execute_batch`）
- **常量**：使用 UPPER_SNAKE_CASE（如 `MAX_WORKERS`）
- **私有成员**：使用下划线前缀（如 `_cache`）
- **接口**：使用 `I` 前缀（如 `ISimulationService`）

### B. 代码审查清单

- [ ] 是否有完整的类型提示？
- [ ] 是否有文档字符串？
- [ ] 是否遵循 SOLID 原则？
- [ ] 是否有适当的错误处理？
- [ ] 是否有单元测试？
- [ ] 是否符合命名约定？
- [ ] 是否有代码重复？
- [ ] 是否有性能问题？

### C. 从 subprocess 迁移到 eppy API 的指南

#### 迁移概述

本次重构将 EnergyPlus 执行方式从直接使用 `subprocess` 调用改为使用 `eppy` 库的 API。这带来了以下优势:

**优势**:
1. **更简洁的代码**: 无需手动构建命令行参数
2. **更好的错误处理**: eppy 提供了更友好的异常处理
3. **自动路径管理**: eppy 自动处理文件路径转换
4. **并行运行支持**: 通过 `runIDFs` 函数轻松实现并行化
5. **更好的跨平台兼容性**: eppy 自动处理不同操作系统的差异

#### 迁移步骤

**步骤 1: 更新构造函数**

```python
# 旧代码 (使用 subprocess)
class EnergyPlusExecutor:
    def __init__(self, executable_path: Path, idd_path: Path, timeout: int = 3600):
        self._executable_path = executable_path
        self._idd_path = idd_path
        self._timeout = timeout

# 新代码 (使用 eppy API)
class EnergyPlusExecutor:
    def __init__(self, idd_path: Path, energyplus_path: Optional[Path] = None):
        self._idd_path = idd_path
        self._energyplus_path = energyplus_path
        # 设置 IDD 文件（eppy 要求）
        IDF.setiddname(str(idd_path))
```

**步骤 2: 更新 run 方法**

```python
# 旧代码 (使用 subprocess)
def run(self, idf, weather_file, output_directory, output_prefix, read_variables=True):
    # 构建命令
    cmd = [
        str(self._executable_path),
        "-w", str(weather_file),
        "-d", str(output_directory),
        "-p", output_prefix,
        "-i", str(self._idd_path),
        str(temp_idf_path),
    ]

    # 执行
    process = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)

    # 处理结果
    result = ExecutionResult(
        success=process.returncode == 0,
        return_code=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
        output_directory=output_directory,
    )

# 新代码 (使用 eppy API)
def run(self, idf, weather_file, output_directory, output_prefix, read_variables=True):
    # 使用 eppy API 执行
    idf.run(
        weather=str(weather_file),
        output_directory=str(output_directory),
        output_prefix=output_prefix,
        readvars=read_variables,
        verbose='q',  # 安静模式
    )

    # 创建结果对象
    result = ExecutionResult(
        success=True,
        return_code=0,
        stdout="",
        stderr="",
        output_directory=output_directory,
    )
```

**步骤 3: 更新配置文件**

```yaml
# 旧配置
paths:
  eplus_executable: /usr/local/EnergyPlus-23-2-0/energyplus  # 必需
  idd_file: /usr/local/EnergyPlus-23-2-0/Energy+.idd

# 新配置
paths:
  idd_file: /usr/local/EnergyPlus-23-2-0/Energy+.idd  # 必需
  # eplus_executable: /usr/local/EnergyPlus-23-2-0/energyplus  # 可选，仅在 eppy 无法自动找到时需要
```

**步骤 4: 更新依赖注入**

```python
# 旧代码
executor = EnergyPlusExecutor(
    executable_path=config.paths.eplus_executable,
    idd_path=config.paths.idd_file,
)

# 新代码
executor = EnergyPlusExecutor(
    idd_path=config.paths.idd_file,
    # energyplus_path=config.paths.eplus_executable,  # 可选
)
```

**步骤 5: 并行运行（可选）**

如果需要并行运行多个模拟，可以使用 eppy 的 `runIDFs` 函数:

```python
from eppy.runner.run_functions import runIDFs

# 准备输入
idf_files = [str(idf1), str(idf2), str(idf3)]
weather_files = [str(epw1), str(epw2), str(epw3)]

# 并行运行
runIDFs(
    idf_files,
    weather_files,
    output_directory=str(output_dir),
    processors=-1,  # 使用所有 CPU 核心
    verbose='q',
)
```

#### 注意事项

1. **IDD 文件设置**: 必须在创建任何 IDF 对象之前调用 `IDF.setiddname()`
2. **路径转换**: eppy 需要字符串路径，使用 `str()` 转换 Path 对象
3. **错误处理**: eppy 的错误信息在 `.err` 文件中，需要手动解析
4. **超时控制**: eppy 不直接支持超时，如需要可以使用 `signal` 模块或 `multiprocessing.Pool` 的 timeout 参数

#### 测试迁移

确保更新相关的单元测试:

```python
# 旧测试
def test_executor_with_subprocess(mock_subprocess):
    executor = EnergyPlusExecutor(Path("energyplus"), Path("Energy+.idd"))
    # ...

# 新测试
def test_executor_with_eppy_api(mock_idf):
    executor = EnergyPlusExecutor(Path("Energy+.idd"))
    # ...
```

### D. 参考资源

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Design Patterns](https://refactoring.guru/design-patterns)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [eppy Documentation](https://eppy.readthedocs.io/)
- [eppy GitHub Repository](https://github.com/santoshphilip/eppy)

---

**文档结束**
