# 领域层实现指南

> Domain Layer Implementation Guide
>
> 版本：2.0（使用 Pydantic）
> 更新日期：2025-10-27

---

## 目录

1. [概述](#概述)
2. [为什么使用Pydantic](#为什么使用pydantic)
3. [目录结构](#目录结构)
4. [领域模型（Entities）](#领域模型entities)
5. [值对象（Value Objects）](#值对象value-objects)
6. [领域服务（Domain Services）](#领域服务domain-services)
7. [仓储接口（Repository Interfaces）](#仓储接口repository-interfaces)
8. [使用示例](#使用示例)
9. [测试策略](#测试策略)

---

## 概述

领域层是系统的核心，包含业务规则和领域逻辑。这一层独立于外部依赖,体现了纯粹的业务领域概念。

### 设计原则

- **业务导向**：代码反映业务术语和概念
- **类型安全**：使用Pydantic提供运行时类型验证
- **纯粹性**：不包含基础设施相关代码
- **验证自动化**：Pydantic自动处理数据验证
- **封装性**：隐藏内部实现细节

### 关键特性

- ✅ 类型安全（100% 类型提示 + 运行时验证）
- ✅ 自动数据验证（Pydantic V2）
- ✅ 不可变值对象（frozen=True）
- ✅ JSON序列化支持
- ✅ 完整的文档字符串

---

## 为什么使用Pydantic

### Pydantic的优势

1. **自动验证**：在对象创建和修改时自动验证数据
2. **类型强制**：自动进行类型转换（如字符串转数字）
3. **JSON支持**：内置的序列化/反序列化
4. **性能优异**：Pydantic V2使用Rust实现，性能提升5-50倍
5. **开发体验**：减少样板代码，提高可读性

### Pydantic vs 传统dataclass

```python
# ❌ 传统dataclass - 需要手动验证
@dataclass
class Building:
    floor_area: float

    def __post_init__(self):
        if self.floor_area <= 0:
            raise ValueError("Floor area must be positive")

# ✅ Pydantic - 自动验证
class Building(BaseModel):
    floor_area: float = Field(gt=0)
    # 自动验证，无需手动代码！
```

### 设计原则说明

**重要**：领域模型不应包含硬编码的计算逻辑。例如：

- ❌ EUI目标值不应硬编码在Building模型中
- ✅ EUI应该由模拟服务计算后存储在SimulationResult中
- ❌ 领域模型不应包含复杂的业务计算
- ✅ 领域模型应专注于数据完整性和简单的业务规则

---

## 目录结构

```
backend/domain/
├── __init__.py
├── models/                      # 领域模型（实体）
│   ├── __init__.py
│   ├── building.py             # 建筑实体
│   ├── weather_file.py         # 天气文件实体
│   ├── simulation_job.py       # 模拟任务实体
│   ├── simulation_result.py    # 模拟结果实体
│   └── enums.py                # 枚举类型
├── value_objects/               # 值对象
│   ├── __init__.py
│   ├── ecm_parameters.py       # ECM 参数
│   ├── location.py             # 位置信息
│   └── simulation_period.py    # 模拟时间段
├── services/                    # 领域服务
│   ├── __init__.py
│   ├── ecm_applicator.py       # ECM 应用器
│   └── pv_system_designer.py   # PV 系统设计器
└── repositories/                # 仓储接口
    ├── __init__.py
    ├── i_building_repository.py
    ├── i_weather_repository.py
    └── i_result_repository.py
```

---

## 领域模型（Entities）

### 1. 枚举类型（enums.py）

```python
"""
领域枚举类型

定义系统中使用的所有枚举类型。
"""

from enum import Enum


class BuildingType(str, Enum):
    """
    建筑类型枚举

    继承str是为了与Pydantic更好地集成。
    """

    OFFICE_LARGE = "OfficeLarge"
    OFFICE_MEDIUM = "OfficeMedium"
    OFFICE_SMALL = "OfficeSmall"
    MULTI_FAMILY_RESIDENTIAL = "MultiFamilyResidential"
    SINGLE_FAMILY_RESIDENTIAL = "SingleFamilyResidential"
    RETAIL = "Retail"
    WAREHOUSE = "Warehouse"
    HOSPITAL = "Hospital"
    SCHOOL = "School"

    def __str__(self) -> str:
        return self.value


class SimulationStatus(str, Enum):
    """模拟状态枚举"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """是否为终止状态"""
        return self in {
            SimulationStatus.COMPLETED,
            SimulationStatus.FAILED,
            SimulationStatus.CANCELLED,
        }


class SimulationType(str, Enum):
    """模拟类型枚举"""

    BASELINE = "baseline"
    PV = "pv"
    OPTIMIZATION = "optimization"
    SENSITIVITY = "sensitivity"
    ECM = "ecm"


class ClimateZone(str, Enum):
    """气候区划分"""

    ZONE_1A = "1A"  # Very Hot - Humid
    ZONE_2A = "2A"  # Hot - Humid
    ZONE_3A = "3A"  # Warm - Humid
    ZONE_4A = "4A"  # Mixed - Humid
    ZONE_5A = "5A"  # Cool - Humid
    ZONE_6A = "6A"  # Cold - Humid
    ZONE_7 = "7"    # Very Cold
    ZONE_8 = "8"    # Subarctic
```

### 2. 建筑实体（building.py）

```python
"""
建筑实体 - 使用Pydantic实现

表示一个建筑的完整信息。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict

from .enums import BuildingType


class Building(BaseModel):
    """
    建筑实体类

    使用Pydantic实现自动验证和序列化。

    Attributes:
        id: 建筑唯一标识符
        name: 建筑名称
        building_type: 建筑类型
        location: 建筑位置（城市）
        idf_file_path: IDF 文件路径
        created_at: 创建时间
        modified_at: 修改时间
        metadata: 元数据字典
        floor_area: 建筑面积（平方米）
        num_floors: 楼层数
        year_built: 建造年份

    Example:
        >>> from pathlib import Path
        >>> building = Building(
        ...     name="Chicago Office",
        ...     building_type=BuildingType.OFFICE_LARGE,
        ...     location="Chicago",
        ...     idf_file_path=Path("chicago_office.idf"),
        ...     floor_area=5000.0,
        ...     num_floors=10,
        ... )
        >>> print(building.get_identifier())
        'Chicago_OfficeLarge'
        >>>
        >>> # JSON序列化
        >>> json_str = building.model_dump_json()
        >>> # 从JSON恢复
        >>> building_copy = Building.model_validate_json(json_str)
    """

    model_config = ConfigDict(
        validate_assignment=True,  # 修改属性时也验证
        frozen=False,  # 允许修改
        arbitrary_types_allowed=True,  # 允许Path等类型
        str_strip_whitespace=True,  # 自动去除字符串首尾空格
        use_enum_values=False,  # 保留枚举对象
    )

    # 必需字段
    name: str = Field(
        min_length=1,
        max_length=255,
        description="建筑名称"
    )

    building_type: BuildingType = Field(
        description="建筑类型"
    )

    location: str = Field(
        min_length=1,
        max_length=100,
        description="建筑位置"
    )

    idf_file_path: Path = Field(
        description="IDF文件路径"
    )

    # 自动生成字段
    id: UUID = Field(
        default_factory=uuid4,
        description="建筑唯一标识符"
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )

    modified_at: datetime = Field(
        default_factory=datetime.now,
        description="修改时间"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )

    # 可选字段
    floor_area: Optional[float] = Field(
        default=None,
        gt=0,
        description="建筑面积（平方米）"
    )

    num_floors: Optional[int] = Field(
        default=None,
        gt=0,
        description="楼层数"
    )

    year_built: Optional[int] = Field(
        default=None,
        ge=1800,
        le=2100,
        description="建造年份"
    )

    # 验证器
    @field_validator('idf_file_path')
    @classmethod
    def validate_idf_file(cls, v: Path) -> Path:
        """验证IDF文件存在且有效"""
        if not v.exists():
            raise ValueError(f"IDF file does not exist: {v}")

        if v.suffix.lower() != '.idf':
            raise ValueError(f"File must be an IDF file, got: {v.suffix}")

        return v

    @field_validator('year_built')
    @classmethod
    def validate_year_built(cls, v: Optional[int]) -> Optional[int]:
        """验证建造年份"""
        if v is not None:
            current_year = datetime.now().year
            if not (1800 <= v <= current_year):
                raise ValueError(
                    f"Invalid year built: {v}. "
                    f"Must be between 1800 and {current_year}"
                )
        return v

    # 业务方法
    def get_identifier(self) -> str:
        """
        获取建筑唯一标识符

        Returns:
            格式为 "{location}_{building_type}" 的字符串

        Example:
            >>> building.get_identifier()
            'Chicago_OfficeLarge'
        """
        return f"{self.location}_{self.building_type.value}"

    def update_metadata(self, key: str, value: Any) -> None:
        """
        更新元数据

        Args:
            key: 元数据键
            value: 元数据值

        Example:
            >>> building.update_metadata("architect", "John Doe")
            >>> building.metadata["architect"]
            'John Doe'
        """
        self.metadata[key] = value
        self.modified_at = datetime.now()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        获取元数据

        Args:
            key: 元数据键
            default: 默认值

        Returns:
            元数据值或默认值
        """
        return self.metadata.get(key, default)

    def is_large_building(self) -> bool:
        """
        判断是否为大型建筑

        Returns:
            如果建筑面积 > 10000 m²，返回True
        """
        return self.floor_area is not None and self.floor_area > 10000

    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"Building(name='{self.name}', "
            f"type={self.building_type.value}, "
            f"location='{self.location}')"
        )
```

### 3. 天气文件实体（weather_file.py）

```python
"""
天气文件实体 - 使用Pydantic实现

表示一个 EPW 天气文件。
"""

from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict

from .enums import ClimateZone


class WeatherFile(BaseModel):
    """
    天气文件实体类

    表示一个 EPW 天气文件，可以是 TMY（典型气象年）或 FTMY（未来典型气象年）。

    Attributes:
        id: 唯一标识符
        file_path: EPW 文件路径
        location: 位置名称
        scenario: 情景名称（如 "TMY", "126", "245" 等）
        is_future: 是否为未来气候情景
        climate_zone: 气候区
        heating_degree_days: 供热度日数
        cooling_degree_days: 供冷度日数

    Example:
        >>> from pathlib import Path
        >>> weather = WeatherFile(
        ...     file_path=Path("chicago_tmy.epw"),
        ...     location="Chicago",
        ...     scenario="TMY",
        ... )
        >>> print(weather.get_identifier())
        'Chicago_TMY'
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    # 必需字段
    file_path: Path = Field(
        description="EPW文件路径"
    )

    location: str = Field(
        min_length=1,
        max_length=100,
        description="位置名称"
    )

    scenario: str = Field(
        min_length=1,
        max_length=50,
        description="情景名称（TMY/FTMY/SSP等）"
    )

    # 自动生成字段
    id: UUID = Field(
        default_factory=uuid4,
        description="唯一标识符"
    )

    # 可选字段
    is_future: bool = Field(
        default=False,
        description="是否为未来气候情景"
    )

    climate_zone: Optional[ClimateZone] = Field(
        default=None,
        description="气候区"
    )

    heating_degree_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="供热度日数"
    )

    cooling_degree_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="供冷度日数"
    )

    # 验证器
    @field_validator('file_path')
    @classmethod
    def validate_epw_file(cls, v: Path) -> Path:
        """验证EPW文件存在且有效"""
        if not v.exists():
            raise ValueError(f"Weather file does not exist: {v}")

        if v.suffix.lower() != '.epw':
            raise ValueError(
                f"Weather file must be .epw format, got: {v.suffix}"
            )

        return v

    # 业务方法
    def get_identifier(self) -> str:
        """
        获取天气文件唯一标识符

        Returns:
            格式为 "{location}_{scenario}" 的字符串
        """
        return f"{self.location}_{self.scenario}"

    def is_typical_meteorological_year(self) -> bool:
        """
        是否为典型气象年

        Returns:
            如果是 TMY 则返回 True
        """
        return self.scenario.upper() == "TMY" and not self.is_future

    def get_scenario_description(self) -> str:
        """
        获取情景描述

        Returns:
            情景的可读描述
        """
        if self.is_typical_meteorological_year():
            return "Typical Meteorological Year"

        if self.is_future:
            # 假设未来情景格式为 SSP 场景编号
            scenario_map = {
                "126": "SSP1-2.6 (Low emissions)",
                "245": "SSP2-4.5 (Intermediate emissions)",
                "370": "SSP3-7.0 (Medium-high emissions)",
                "434": "SSP4-3.4 (Intermediate emissions, low overshoot)",
                "585": "SSP5-8.5 (High emissions)",
            }
            return scenario_map.get(self.scenario, f"Future scenario {self.scenario}")

        return self.scenario

    def __str__(self) -> str:
        """字符串表示"""
        return f"WeatherFile(location='{self.location}', scenario='{self.scenario}')"
```

### 4. 模拟任务实体（simulation_job.py）

```python
"""
模拟任务实体 - 使用Pydantic实现

表示一个完整的模拟任务。
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, ForwardRef
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict

from .enums import SimulationStatus, SimulationType

if TYPE_CHECKING:
    from ..value_objects.ecm_parameters import ECMParameters
    from .building import Building
    from .simulation_result import SimulationResult
    from .weather_file import WeatherFile


class SimulationJob(BaseModel):
    """
    模拟任务实体类

    表示一个完整的模拟任务，包含建筑、天气、配置等信息。
    这是一个聚合根（Aggregate Root），管理模拟的生命周期。

    Attributes:
        id: 任务唯一标识符
        building: 建筑对象
        weather_file: 天气文件对象
        simulation_type: 模拟类型
        output_directory: 输出目录
        output_prefix: 输出文件前缀
        read_variables: 是否读取变量
        ecm_parameters: ECM 参数（可选）
        status: 任务状态
        created_at: 创建时间
        started_at: 开始时间
        completed_at: 完成时间
        result: 模拟结果
        error_message: 错误信息

    Example:
        >>> job = SimulationJob(
        ...     building=building,
        ...     weather_file=weather,
        ...     simulation_type=SimulationType.BASELINE,
        ...     output_directory=Path("output"),
        ...     output_prefix="baseline",
        ... )
        >>> job.start()
        >>> job.status
        <SimulationStatus.RUNNING: 'running'>
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
    )

    # 必需字段
    building: 'Building' = Field(
        description="建筑对象"
    )

    weather_file: 'WeatherFile' = Field(
        description="天气文件对象"
    )

    simulation_type: SimulationType = Field(
        description="模拟类型"
    )

    output_directory: Path = Field(
        description="输出目录"
    )

    output_prefix: str = Field(
        min_length=1,
        description="输出文件前缀"
    )

    # 自动生成字段
    id: UUID = Field(
        default_factory=uuid4,
        description="任务唯一标识符"
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )

    status: SimulationStatus = Field(
        default=SimulationStatus.PENDING,
        description="任务状态"
    )

    # 可选字段
    read_variables: bool = Field(
        default=True,
        description="是否读取变量"
    )

    ecm_parameters: Optional['ECMParameters'] = Field(
        default=None,
        description="ECM参数"
    )

    started_at: Optional[datetime] = Field(
        default=None,
        description="开始时间"
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="完成时间"
    )

    result: Optional['SimulationResult'] = Field(
        default=None,
        description="模拟结果"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="错误信息"
    )

    # 业务方法
    def start(self) -> None:
        """
        标记任务为运行中

        Raises:
            ValueError: 如果任务不在 PENDING 状态
        """
        if self.status != SimulationStatus.PENDING:
            raise ValueError(
                f"Cannot start job in {self.status} status. "
                f"Job must be in PENDING status."
            )

        self.status = SimulationStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, result: 'SimulationResult') -> None:
        """
        标记任务为完成

        Args:
            result: 模拟结果

        Raises:
            ValueError: 如果任务不在 RUNNING 状态
        """
        if self.status != SimulationStatus.RUNNING:
            raise ValueError(
                f"Cannot complete job in {self.status} status. "
                f"Job must be in RUNNING status."
            )

        self.status = SimulationStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result

    def fail(self, error_message: str) -> None:
        """
        标记任务为失败

        Args:
            error_message: 错误信息

        Raises:
            ValueError: 如果任务已经在终止状态
        """
        if self.status.is_terminal() and self.status != SimulationStatus.RUNNING:
            raise ValueError(
                f"Cannot fail job in {self.status} status. "
                f"Job is already in a terminal state."
            )

        self.status = SimulationStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message

    def cancel(self) -> None:
        """
        取消任务

        Raises:
            ValueError: 如果任务已经在终止状态
        """
        if self.status.is_terminal():
            raise ValueError(
                f"Cannot cancel job in {self.status} status. "
                f"Job is already in a terminal state."
            )

        self.status = SimulationStatus.CANCELLED
        self.completed_at = datetime.now()

    def get_cache_key(self) -> str:
        """
        获取缓存键

        Returns:
            用于缓存的唯一键
        """
        building_id = self.building.get_identifier()
        weather_id = self.weather_file.get_identifier()
        ecm_hash = hash(str(self.ecm_parameters)) if self.ecm_parameters else 0
        return f"{building_id}_{weather_id}_{self.simulation_type.value}_{ecm_hash}"

    def get_duration(self) -> Optional[float]:
        """
        获取任务执行时长

        Returns:
            执行时长（秒），如果任务未完成则返回 None
        """
        if self.started_at is None or self.completed_at is None:
            return None

        duration = self.completed_at - self.started_at
        return duration.total_seconds()

    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"SimulationJob(id={self.id}, type={self.simulation_type.value}, "
            f"status={self.status.value})"
        )
```

### 5. 模拟结果实体（simulation_result.py）

```python
"""
模拟结果实体 - 使用Pydantic实现

包含模拟的输出数据和元数据。

注意：EUI等能耗指标应该由模拟服务计算后存储，不在领域模型中计算。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator


class SimulationResult(BaseModel):
    """
    模拟结果实体类

    包含模拟的输出数据和元数据。

    **重要**：EUI等能耗指标由模拟服务计算后赋值，不在此处计算。

    Attributes:
        id: 结果唯一标识符
        job_id: 关联的任务 ID
        output_directory: 输出目录
        table_csv_path: Table CSV 文件路径
        meter_csv_path: Meter CSV 文件路径
        sql_path: SQL 文件路径
        source_eui: 源能耗强度（kWh/m²/yr）- 由ResultParser计算
        site_eui: 场地能耗强度（kWh/m²/yr）- 由ResultParser计算
        total_energy_kwh: 总能耗（kWh）- 由ResultParser计算
        execution_time: 执行时间（秒）
        success: 是否成功
        error_messages: 错误信息列表
        warning_messages: 警告信息列表
        created_at: 创建时间
        metadata: 元数据

    Example:
        >>> result = SimulationResult(
        ...     job_id=job.id,
        ...     output_directory=Path("output"),
        ... )
        >>> # EUI由ResultParser计算后赋值
        >>> result.source_eui = parser.parse_source_eui()
        >>> result.site_eui = parser.parse_site_eui()
        >>> result.success = True
        >>>
        >>> if result.is_valid():
        ...     print(f"Success! EUI: {result.source_eui}")
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=True,
    )

    # 必需字段
    job_id: UUID = Field(
        description="关联的任务ID"
    )

    output_directory: Path = Field(
        description="输出目录"
    )

    # 自动生成字段
    id: UUID = Field(
        default_factory=uuid4,
        description="结果唯一标识符"
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )

    error_messages: list[str] = Field(
        default_factory=list,
        description="错误信息列表"
    )

    warning_messages: list[str] = Field(
        default_factory=list,
        description="警告信息列表"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )

    # 可选字段 - 由模拟服务填充
    table_csv_path: Optional[Path] = Field(
        default=None,
        description="Table CSV文件路径"
    )

    meter_csv_path: Optional[Path] = Field(
        default=None,
        description="Meter CSV文件路径"
    )

    sql_path: Optional[Path] = Field(
        default=None,
        description="SQL文件路径"
    )

    source_eui: Optional[float] = Field(
        default=None,
        ge=0,
        description="源能耗强度（kWh/m²/yr）- 由ResultParser计算"
    )

    site_eui: Optional[float] = Field(
        default=None,
        ge=0,
        description="场地能耗强度（kWh/m²/yr）- 由ResultParser计算"
    )

    total_energy_kwh: Optional[float] = Field(
        default=None,
        ge=0,
        description="总能耗（kWh）- 由ResultParser计算"
    )

    execution_time: Optional[float] = Field(
        default=None,
        ge=0,
        description="执行时间（秒）"
    )

    success: bool = Field(
        default=False,
        description="是否成功"
    )

    # 业务方法
    def add_error(self, message: str) -> None:
        """
        添加错误信息

        Args:
            message: 错误信息
        """
        self.error_messages.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """
        添加警告信息

        Args:
            message: 警告信息
        """
        self.warning_messages.append(message)

    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.error_messages) > 0

    def is_valid(self) -> bool:
        """
        验证结果是否有效

        Returns:
            如果结果有效则返回 True

        Note:
            验证包括：
            - 成功标志
            - EUI 值存在且合理
        """
        if not self.success:
            return False

        # 检查 EUI 值
        if self.source_eui is None:
            return False

        # 合理性检查（EUI通常不会超过1000 kWh/m²/yr）
        if self.source_eui > 1000:
            self.add_warning(
                f"Source EUI seems unreasonably high: {self.source_eui} kWh/m²/yr"
            )

        return True

    def get_eui_summary(self) -> dict[str, Optional[float]]:
        """
        获取 EUI 摘要

        Returns:
            包含各种 EUI 指标的字典
        """
        return {
            "source_eui": self.source_eui,
            "site_eui": self.site_eui,
            "total_energy_kwh": self.total_energy_kwh,
        }

    def __str__(self) -> str:
        """字符串表示"""
        status = "Success" if self.success else "Failed"
        return (
            f"SimulationResult(id={self.id}, job_id={self.job_id}, "
            f"status={status}, source_eui={self.source_eui})"
        )
```

---

## 值对象（Value Objects）

### 1. ECM 参数（ecm_parameters.py）

```python
"""
ECM 参数值对象 - 使用Pydantic实现

表示能效措施（Energy Conservation Measures）的参数。
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ECMParameters(BaseModel):
    """
    Energy Conservation Measures (ECM) 参数值对象

    使用Pydantic实现不可变值对象，包含所有能效改造措施的参数。

    Attributes:
        window_u_value: 窗户传热系数（W/m²K）
        window_shgc: 太阳得热系数（0-1）
        wall_insulation: 墙体保温 R-value
        roof_insulation: 屋顶保温 R-value
        infiltration_rate: 渗透率（ACH）
        natural_ventilation_area: 自然通风面积（m²）
        cooling_cop: 制冷系统 COP
        heating_efficiency: 供热效率（0-1）
        cooling_setpoint: 制冷设定温度（°C）
        heating_setpoint: 供热设定温度（°C）
        lighting_power_density: 照明功率密度（W/m²）
        lighting_reduction_factor: 照明削减因子（0-1）

    Example:
        >>> ecm = ECMParameters(
        ...     window_u_value=1.5,
        ...     window_shgc=0.4,
        ...     cooling_cop=4.0,
        ... )
        >>> ecm_dict = ecm.to_dict()
        >>> # 可哈希
        >>> hash(ecm)
        1234567890
    """

    model_config = ConfigDict(
        frozen=True,  # 不可变
        validate_assignment=True,
    )

    # 围护结构参数
    window_u_value: Optional[float] = Field(
        default=None,
        gt=0.1,
        lt=10.0,
        description="窗户传热系数（W/m²K）"
    )

    window_shgc: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="太阳得热系数（0-1）"
    )

    wall_insulation: Optional[float] = Field(
        default=None,
        gt=0,
        description="墙体保温 R-value"
    )

    roof_insulation: Optional[float] = Field(
        default=None,
        gt=0,
        description="屋顶保温 R-value"
    )

    # 通风参数
    infiltration_rate: Optional[float] = Field(
        default=None,
        gt=0,
        description="渗透率（ACH）"
    )

    natural_ventilation_area: Optional[float] = Field(
        default=None,
        ge=0,
        description="自然通风面积（m²）"
    )

    # HVAC 参数
    cooling_cop: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=10.0,
        description="制冷系统 COP"
    )

    heating_efficiency: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="供热效率（0-1）"
    )

    cooling_setpoint: Optional[float] = Field(
        default=None,
        ge=18.0,
        le=30.0,
        description="制冷设定温度（°C）"
    )

    heating_setpoint: Optional[float] = Field(
        default=None,
        ge=15.0,
        le=25.0,
        description="供热设定温度（°C）"
    )

    # 照明参数
    lighting_power_density: Optional[float] = Field(
        default=None,
        ge=0,
        description="照明功率密度（W/m²）"
    )

    lighting_reduction_factor: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="照明削减因子（0-1）"
    )

    # 业务方法
    def to_dict(self) -> dict[str, float]:
        """
        转换为字典，只包含非 None 值

        Returns:
            包含所有非 None 参数的字典
        """
        return {
            k: v for k, v in self.model_dump().items()
            if v is not None
        }

    def merge(self, other: 'ECMParameters') -> 'ECMParameters':
        """
        合并两个 ECM 参数对象

        Args:
            other: 另一个 ECM 参数对象

        Returns:
            合并后的新 ECM 参数对象

        Note:
            other 中的非 None 值会覆盖 self 中的值
        """
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return ECMParameters(**merged_dict)

    def __hash__(self) -> int:
        """支持哈希，用于缓存键"""
        return hash(tuple(sorted(self.to_dict().items())))

    def __str__(self) -> str:
        """字符串表示"""
        non_none_params = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"ECMParameters({', '.join(non_none_params)})"
```

### 2. 位置值对象（location.py）

```python
"""
位置值对象 - 使用Pydantic实现

表示地理位置信息。
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from ..models.enums import ClimateZone


class Location(BaseModel):
    """
    位置值对象

    表示一个地理位置的信息。

    Attributes:
        city: 城市名称
        country: 国家名称
        climate_zone: 气候区
        latitude: 纬度
        longitude: 经度

    Example:
        >>> location = Location(
        ...     city="Chicago",
        ...     country="USA",
        ...     climate_zone=ClimateZone.ZONE_5A,
        ...     latitude=41.8781,
        ...     longitude=-87.6298,
        ... )
        >>> str(location)
        'Chicago, USA'
    """

    model_config = ConfigDict(
        frozen=True,  # 不可变
        validate_assignment=True,
    )

    city: str = Field(
        min_length=1,
        description="城市名称"
    )

    country: str = Field(
        min_length=1,
        description="国家名称"
    )

    climate_zone: Optional[ClimateZone] = Field(
        default=None,
        description="气候区"
    )

    latitude: Optional[float] = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="纬度"
    )

    longitude: Optional[float] = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="经度"
    )

    # 业务方法
    def get_coordinates(self) -> Optional[tuple[float, float]]:
        """
        获取坐标

        Returns:
            (latitude, longitude) 元组，如果坐标不可用则返回 None
        """
        if self.latitude is not None and self.longitude is not None:
            return (self.latitude, self.longitude)
        return None

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.city}, {self.country}"
```

### 3. 模拟时间段（simulation_period.py）

```python
"""
模拟时间段值对象 - 使用Pydantic实现

表示模拟的时间范围。
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class SimulationPeriod(BaseModel):
    """
    模拟时间段值对象

    表示模拟的起止时间。

    Attributes:
        start_year: 开始年份
        end_year: 结束年份
        start_month: 开始月份（1-12）
        end_month: 结束月份（1-12）
        start_day: 开始日期（1-31）
        end_day: 结束日期（1-31）

    Example:
        >>> period = SimulationPeriod(
        ...     start_year=2040,
        ...     end_year=2040,
        ... )
        >>> period.get_duration_years()
        1
    """

    model_config = ConfigDict(
        frozen=True,  # 不可变
        validate_assignment=True,
    )

    start_year: int = Field(
        ge=1900,
        le=2100,
        description="开始年份"
    )

    end_year: int = Field(
        ge=1900,
        le=2100,
        description="结束年份"
    )

    start_month: int = Field(
        default=1,
        ge=1,
        le=12,
        description="开始月份"
    )

    end_month: int = Field(
        default=12,
        ge=1,
        le=12,
        description="结束月份"
    )

    start_day: int = Field(
        default=1,
        ge=1,
        le=31,
        description="开始日期"
    )

    end_day: int = Field(
        default=31,
        ge=1,
        le=31,
        description="结束日期"
    )

    @model_validator(mode='after')
    def validate_period(self) -> 'SimulationPeriod':
        """验证时间段的有效性"""
        # 验证年份顺序
        if self.start_year > self.end_year:
            raise ValueError(
                f"Start year ({self.start_year}) must be <= end year ({self.end_year})"
            )

        # 如果是同一年，检查时间顺序
        if self.start_year == self.end_year:
            if self.start_month > self.end_month:
                raise ValueError(
                    f"Start month ({self.start_month}) must be <= end month ({self.end_month}) "
                    f"for the same year"
                )

            if self.start_month == self.end_month and self.start_day > self.end_day:
                raise ValueError(
                    f"Start day ({self.start_day}) must be <= end day ({self.end_day}) "
                    f"for the same month"
                )

        return self

    # 业务方法
    def get_duration_years(self) -> int:
        """
        获取持续年数

        Returns:
            持续年数
        """
        return self.end_year - self.start_year + 1

    def is_full_year(self) -> bool:
        """
        是否为完整年份

        Returns:
            如果模拟覆盖完整年份则返回 True
        """
        return (
            self.start_month == 1
            and self.start_day == 1
            and self.end_month == 12
            and self.end_day == 31
        )

    def __str__(self) -> str:
        """字符串表示"""
        if self.is_full_year() and self.start_year == self.end_year:
            return f"Year {self.start_year}"

        start = f"{self.start_year}-{self.start_month:02d}-{self.start_day:02d}"
        end = f"{self.end_year}-{self.end_month:02d}-{self.end_day:02d}"
        return f"{start} to {end}"
```

---

## 领域服务（Domain Services）

领域服务封装了不属于特定实体或值对象的业务逻辑。这些服务是无状态的，专注于协调领域对象之间的交互。

### 1. ECM 应用器（ecm_applicator.py）

#### 接口定义

```python
"""
ECM 应用器领域服务

负责将 ECM 参数应用到 IDF 文件。
"""

from abc import ABC, abstractmethod
from typing import Any

from ..value_objects.ecm_parameters import ECMParameters


class IECMApplicator(ABC):
    """
    ECM 应用器接口

    负责将 ECM 参数应用到建筑模型（IDF）中。
    """

    @abstractmethod
    def apply(self, idf: Any, parameters: ECMParameters) -> None:
        """
        应用 ECM 参数到 IDF

        Args:
            idf: IDF 对象
            parameters: ECM 参数

        Raises:
            ValueError: 如果参数无效
            RuntimeError: 如果应用失败
        """
        pass

    @abstractmethod
    def validate(self, parameters: ECMParameters) -> bool:
        """
        验证 ECM 参数是否可以应用

        Args:
            parameters: ECM 参数

        Returns:
            如果参数有效则返回 True
        """
        pass
```

#### 完整实现

```python
"""
ECM 应用器实现

使用eppy库操作IDF文件，应用能效改造措施。
"""

from typing import Any, Optional
from loguru import logger

from ..value_objects.ecm_parameters import ECMParameters
from .i_ecm_applicator import IECMApplicator


class ECMApplicator(IECMApplicator):
    """
    ECM 应用器实现

    使用领域驱动设计原则，将ECM参数应用到IDF模型中。

    Example:
        >>> from eppy.modeleditor import IDF
        >>> idf = IDF("building.idf")
        >>> applicator = ECMApplicator()
        >>>
        >>> ecm_params = ECMParameters(
        ...     window_u_value=1.5,
        ...     window_shgc=0.4,
        ...     cooling_cop=4.0,
        ... )
        >>>
        >>> if applicator.validate(ecm_params):
        ...     applicator.apply(idf, ecm_params)
        ...     idf.save()
    """

    def validate(self, parameters: ECMParameters) -> bool:
        """
        验证 ECM 参数是否有效

        Args:
            parameters: ECM 参数

        Returns:
            如果所有参数都有效则返回 True

        Note:
            Pydantic已经在创建时进行了基本验证，
            这里主要进行业务规则验证。
        """
        # Pydantic已经验证了基本约束
        # 这里可以添加额外的业务规则验证

        params_dict = parameters.to_dict()

        if not params_dict:
            logger.warning("No ECM parameters provided")
            return False

        # 验证参数组合的合理性
        if parameters.cooling_setpoint and parameters.heating_setpoint:
            if parameters.cooling_setpoint <= parameters.heating_setpoint:
                logger.error(
                    f"Cooling setpoint ({parameters.cooling_setpoint}°C) must be "
                    f"greater than heating setpoint ({parameters.heating_setpoint}°C)"
                )
                return False

        logger.info(f"ECM parameters validated: {len(params_dict)} parameters")
        return True

    def apply(self, idf: Any, parameters: ECMParameters) -> None:
        """
        应用 ECM 参数到 IDF

        Args:
            idf: IDF 对象（eppy的IDF实例）
            parameters: ECM 参数

        Raises:
            ValueError: 如果参数无效
            RuntimeError: 如果应用失败

        Note:
            这个方法按照优先级顺序应用ECM参数：
            1. 围护结构（窗户、墙体、屋顶）
            2. 渗透率
            3. 自然通风
            4. HVAC系统
            5. 照明系统
        """
        if not self.validate(parameters):
            raise ValueError("Invalid ECM parameters")

        try:
            # 1. 应用围护结构参数
            if parameters.window_u_value is not None or parameters.window_shgc is not None:
                self._apply_window_parameters(idf, parameters)

            if parameters.wall_insulation is not None:
                self._apply_wall_insulation(idf, parameters.wall_insulation)

            if parameters.roof_insulation is not None:
                self._apply_roof_insulation(idf, parameters.roof_insulation)

            # 2. 应用渗透率
            if parameters.infiltration_rate is not None:
                self._apply_infiltration(idf, parameters.infiltration_rate)

            # 3. 应用自然通风
            if parameters.natural_ventilation_area is not None:
                self._apply_natural_ventilation(idf, parameters.natural_ventilation_area)

            # 4. 应用 HVAC 参数
            if any([
                parameters.cooling_cop is not None,
                parameters.heating_efficiency is not None,
                parameters.cooling_setpoint is not None,
                parameters.heating_setpoint is not None,
            ]):
                self._apply_hvac_parameters(idf, parameters)

            # 5. 应用照明参数
            if parameters.lighting_power_density is not None or parameters.lighting_reduction_factor is not None:
                self._apply_lighting_parameters(idf, parameters)

            logger.info("ECM parameters applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply ECM parameters: {e}")
            raise RuntimeError(f"Failed to apply ECM parameters: {e}")

    def _apply_window_parameters(
        self,
        idf: Any,
        parameters: ECMParameters,
    ) -> None:
        """
        应用窗户参数

        Args:
            idf: IDF 对象
            parameters: ECM 参数
        """
        # 获取所有窗户材料
        window_materials = idf.idfobjects.get("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", [])

        if not window_materials:
            logger.warning("No window materials found in IDF")
            return

        for material in window_materials:
            if parameters.window_u_value is not None:
                material.UFactor = parameters.window_u_value
                logger.debug(f"Set window U-value to {parameters.window_u_value} for {material.Name}")

            if parameters.window_shgc is not None:
                material.Solar_Heat_Gain_Coefficient = parameters.window_shgc
                logger.debug(f"Set window SHGC to {parameters.window_shgc} for {material.Name}")

    def _apply_wall_insulation(self, idf: Any, r_value: float) -> None:
        """
        应用墙体保温

        Args:
            idf: IDF 对象
            r_value: R-value（保温值）
        """
        # 获取所有墙体材料
        materials = idf.idfobjects.get("MATERIAL", [])

        for material in materials:
            # 识别保温材料（通常名称中包含"Insulation"）
            if "insulation" in material.Name.lower() and "wall" in material.Name.lower():
                # R-value = Thickness / Conductivity
                # 保持导热系数不变，调整厚度
                conductivity = material.Conductivity
                new_thickness = r_value * conductivity
                material.Thickness = new_thickness
                logger.debug(f"Set wall insulation thickness to {new_thickness}m for {material.Name}")

    def _apply_roof_insulation(self, idf: Any, r_value: float) -> None:
        """
        应用屋顶保温

        Args:
            idf: IDF 对象
            r_value: R-value（保温值）
        """
        materials = idf.idfobjects.get("MATERIAL", [])

        for material in materials:
            if "insulation" in material.Name.lower() and "roof" in material.Name.lower():
                conductivity = material.Conductivity
                new_thickness = r_value * conductivity
                material.Thickness = new_thickness
                logger.debug(f"Set roof insulation thickness to {new_thickness}m for {material.Name}")

    def _apply_infiltration(self, idf: Any, infiltration_rate: float) -> None:
        """
        应用渗透率

        Args:
            idf: IDF 对象
            infiltration_rate: 渗透率（ACH - Air Changes per Hour）
        """
        infiltration_objects = idf.idfobjects.get("ZONEINFILTRATION:DESIGNFLOWRATE", [])

        if not infiltration_objects:
            logger.warning("No infiltration objects found in IDF")
            return

        for infiltration in infiltration_objects:
            # 设置为空气交换率方法
            infiltration.Design_Flow_Rate_Calculation_Method = "AirChanges/Hour"
            infiltration.Air_Changes_per_Hour = infiltration_rate
            logger.debug(f"Set infiltration rate to {infiltration_rate} ACH for {infiltration.Name}")

    def _apply_natural_ventilation(self, idf: Any, nv_area: float) -> None:
        """
        应用自然通风

        Args:
            idf: IDF 对象
            nv_area: 自然通风面积（m²）
        """
        ventilation_objects = idf.idfobjects.get("ZONEVENTILATION:DESIGNFLOWRATE", [])

        if not ventilation_objects:
            logger.info("No natural ventilation objects found, would need to create them")
            return

        for ventilation in ventilation_objects:
            # 更新通风面积
            if hasattr(ventilation, "Design_Flow_Rate_Calculation_Method"):
                ventilation.Design_Flow_Rate_Calculation_Method = "Flow/Area"
                ventilation.Flow_Rate_per_Zone_Floor_Area = nv_area
                logger.debug(f"Set natural ventilation area to {nv_area} m² for {ventilation.Name}")

    def _apply_hvac_parameters(self, idf: Any, parameters: ECMParameters) -> None:
        """
        应用 HVAC 参数

        Args:
            idf: IDF 对象
            parameters: ECM 参数
        """
        # 1. 应用制冷系统 COP
        if parameters.cooling_cop is not None:
            chillers = idf.idfobjects.get("CHILLER:ELECTRIC:EIR", [])
            for chiller in chillers:
                # EIR = 1 / COP
                chiller.Reference_COP = parameters.cooling_cop
                logger.debug(f"Set cooling COP to {parameters.cooling_cop} for {chiller.Name}")

        # 2. 应用供热效率
        if parameters.heating_efficiency is not None:
            boilers = idf.idfobjects.get("BOILER:HOTWATER", [])
            for boiler in boilers:
                boiler.Nominal_Thermal_Efficiency = parameters.heating_efficiency
                logger.debug(f"Set heating efficiency to {parameters.heating_efficiency} for {boiler.Name}")

        # 3. 应用温度设定点
        if parameters.cooling_setpoint is not None or parameters.heating_setpoint is not None:
            thermostats = idf.idfobjects.get("HVACTEMPLATE:THERMOSTAT", [])

            for thermostat in thermostats:
                if parameters.cooling_setpoint is not None:
                    thermostat.Constant_Cooling_Setpoint = parameters.cooling_setpoint
                    logger.debug(f"Set cooling setpoint to {parameters.cooling_setpoint}°C")

                if parameters.heating_setpoint is not None:
                    thermostat.Constant_Heating_Setpoint = parameters.heating_setpoint
                    logger.debug(f"Set heating setpoint to {parameters.heating_setpoint}°C")

    def _apply_lighting_parameters(self, idf: Any, parameters: ECMParameters) -> None:
        """
        应用照明参数

        Args:
            idf: IDF 对象
            parameters: ECM 参数
        """
        lights = idf.idfobjects.get("LIGHTS", [])

        if not lights:
            logger.warning("No lighting objects found in IDF")
            return

        for light in lights:
            # 应用照明功率密度
            if parameters.lighting_power_density is not None:
                if hasattr(light, "Design_Level_Calculation_Method"):
                    light.Design_Level_Calculation_Method = "Watts/Area"
                    light.Watts_per_Zone_Floor_Area = parameters.lighting_power_density
                    logger.debug(
                        f"Set lighting power density to {parameters.lighting_power_density} W/m² "
                        f"for {light.Name}"
                    )

            # 应用照明削减因子
            if parameters.lighting_reduction_factor is not None:
                # 削减因子应用于现有功率
                current_power = getattr(light, "Watts_per_Zone_Floor_Area", 10.0)  # 默认10 W/m²
                new_power = current_power * (1 - parameters.lighting_reduction_factor)
                light.Watts_per_Zone_Floor_Area = new_power
                logger.debug(
                    f"Applied lighting reduction factor {parameters.lighting_reduction_factor}, "
                    f"new power: {new_power} W/m² for {light.Name}"
                )
```

### 2. PV 系统设计器（pv_system_designer.py）

#### 接口定义

```python
"""
PV 系统设计器领域服务

负责光伏系统的设计和容量计算。
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..models.building import Building
from ..value_objects.location import Location


class IPVSystemDesigner(ABC):
    """
    PV 系统设计器接口

    负责设计建筑的光伏系统。
    """

    @abstractmethod
    def design_pv_system(
        self,
        building: Building,
        location: Location,
        coverage_ratio: float = 0.7,
    ) -> dict[str, float]:
        """
        设计 PV 系统

        Args:
            building: 建筑对象
            location: 位置信息
            coverage_ratio: 屋顶覆盖率（0-1）

        Returns:
            PV系统设计参数字典

        Raises:
            ValueError: 如果参数无效
        """
        pass

    @abstractmethod
    def calculate_system_capacity(
        self,
        roof_area: float,
        coverage_ratio: float,
        panel_efficiency: float,
    ) -> float:
        """
        计算系统容量

        Args:
            roof_area: 屋顶面积（m²）
            coverage_ratio: 覆盖率（0-1）
            panel_efficiency: 面板效率（0-1）

        Returns:
            系统容量（kW）
        """
        pass
```

#### 完整实现

```python
"""
PV 系统设计器实现

使用领域驱动设计原则进行光伏系统设计。
"""

from typing import Optional
import math
from loguru import logger

from ..models.building import Building
from ..value_objects.location import Location
from .i_pv_system_designer import IPVSystemDesigner


class PVSystemDesigner(IPVSystemDesigner):
    """
    PV 系统设计器实现

    根据建筑特征和位置信息设计光伏系统。

    Attributes:
        default_panel_efficiency: 默认面板效率（18%）
        default_system_losses: 默认系统损失（14%）
        standard_test_irradiance: 标准测试辐照度（1000 W/m²）

    Example:
        >>> designer = PVSystemDesigner()
        >>>
        >>> building = Building(
        ...     name="Office",
        ...     building_type=BuildingType.OFFICE_LARGE,
        ...     location="Chicago",
        ...     idf_file_path=Path("office.idf"),
        ...     floor_area=5000.0,
        ... )
        >>>
        >>> location = Location(
        ...     city="Chicago",
        ...     country="USA",
        ...     latitude=41.8781,
        ...     longitude=-87.6298,
        ... )
        >>>
        >>> pv_design = designer.design_pv_system(
        ...     building=building,
        ...     location=location,
        ...     coverage_ratio=0.7,
        ... )
        >>>
        >>> print(f"System capacity: {pv_design['system_capacity_kw']} kW")
        >>> print(f"Annual generation: {pv_design['estimated_annual_generation_kwh']} kWh")
    """

    def __init__(
        self,
        default_panel_efficiency: float = 0.18,
        default_system_losses: float = 0.14,
    ):
        """
        初始化 PV 系统设计器

        Args:
            default_panel_efficiency: 默认面板效率（0-1）
            default_system_losses: 默认系统损失（0-1）
        """
        if not (0 < default_panel_efficiency < 1):
            raise ValueError("Panel efficiency must be between 0 and 1")

        if not (0 < default_system_losses < 1):
            raise ValueError("System losses must be between 0 and 1")

        self.default_panel_efficiency = default_panel_efficiency
        self.default_system_losses = default_system_losses
        self.standard_test_irradiance = 1000.0  # W/m²

    def design_pv_system(
        self,
        building: Building,
        location: Location,
        coverage_ratio: float = 0.7,
    ) -> dict[str, float]:
        """
        设计 PV 系统

        Args:
            building: 建筑对象
            location: 位置信息
            coverage_ratio: 屋顶覆盖率（0-1）

        Returns:
            PV系统设计参数字典，包含：
            - roof_area: 屋顶面积（m²）
            - available_area: 可用面积（m²）
            - system_capacity_kw: 系统容量（kW）
            - panel_efficiency: 面板效率
            - system_losses: 系统损失
            - estimated_annual_generation_kwh: 估算年发电量（kWh）

        Raises:
            ValueError: 如果参数无效
        """
        if not (0 < coverage_ratio <= 1):
            raise ValueError("Coverage ratio must be between 0 and 1")

        if building.floor_area is None:
            raise ValueError("Building floor area is required for PV system design")

        # 1. 计算屋顶面积
        roof_area = self._calculate_roof_area(building)
        logger.info(f"Calculated roof area: {roof_area} m²")

        # 2. 计算可用面积（考虑覆盖率）
        available_area = roof_area * coverage_ratio

        # 3. 计算系统容量
        system_capacity_kw = self.calculate_system_capacity(
            roof_area=available_area,
            coverage_ratio=1.0,  # 已经应用过覆盖率
            panel_efficiency=self.default_panel_efficiency,
        )

        # 4. 估算年发电量
        estimated_annual_generation = self._estimate_annual_generation(
            system_capacity_kw=system_capacity_kw,
            location=location,
        )

        # 5. 返回设计参数
        design = {
            "roof_area": roof_area,
            "available_area": available_area,
            "system_capacity_kw": system_capacity_kw,
            "panel_efficiency": self.default_panel_efficiency,
            "system_losses": self.default_system_losses,
            "estimated_annual_generation_kwh": estimated_annual_generation,
        }

        logger.info(f"PV system designed: {system_capacity_kw:.2f} kW capacity")
        return design

    def calculate_system_capacity(
        self,
        roof_area: float,
        coverage_ratio: float,
        panel_efficiency: float,
    ) -> float:
        """
        计算系统容量

        使用公式：
        Capacity (kW) = Area × Coverage × Efficiency × Irradiance / 1000

        Args:
            roof_area: 屋顶面积（m²）
            coverage_ratio: 覆盖率（0-1）
            panel_efficiency: 面板效率（0-1）

        Returns:
            系统容量（kW）

        Raises:
            ValueError: 如果参数无效
        """
        if roof_area <= 0:
            raise ValueError("Roof area must be positive")

        if not (0 < coverage_ratio <= 1):
            raise ValueError("Coverage ratio must be between 0 and 1")

        if not (0 < panel_efficiency < 1):
            raise ValueError("Panel efficiency must be between 0 and 1")

        # 计算系统容量（kW）
        capacity_kw = (
            roof_area
            * coverage_ratio
            * panel_efficiency
            * self.standard_test_irradiance
            / 1000.0
        )

        return round(capacity_kw, 2)

    def _calculate_roof_area(self, building: Building) -> float:
        """
        计算建筑屋顶面积

        Args:
            building: 建筑对象

        Returns:
            屋顶面积（m²）

        Note:
            简化假设：屋顶面积 = 建筑面积 / 楼层数
            实际项目中应该从IDF文件中提取准确的屋顶面积
        """
        floor_area = building.floor_area
        if floor_area is None:
            raise ValueError("Building floor area is required")

        num_floors = building.num_floors or 1
        roof_area = floor_area / num_floors

        return roof_area

    def _estimate_annual_generation(
        self,
        system_capacity_kw: float,
        location: Location,
    ) -> float:
        """
        估算年发电量

        使用简化的方法基于纬度估算。
        实际项目应该使用更精确的太阳辐射数据。

        Args:
            system_capacity_kw: 系统容量（kW）
            location: 位置信息

        Returns:
            估算年发电量（kWh）

        Note:
            使用公式：
            Generation = Capacity × Peak Sun Hours × 365 × (1 - Losses)

            Peak Sun Hours 基于纬度估算：
            - 低纬度（0-30°）: 5-6小时
            - 中纬度（30-50°）: 4-5小时
            - 高纬度（50°+）: 3-4小时
        """
        # 获取纬度
        latitude = location.latitude
        if latitude is None:
            logger.warning("Latitude not provided, using default peak sun hours")
            peak_sun_hours = 4.5  # 默认值
        else:
            # 根据纬度估算峰值日照时数
            abs_latitude = abs(latitude)
            if abs_latitude <= 30:
                peak_sun_hours = 5.5
            elif abs_latitude <= 50:
                peak_sun_hours = 4.5
            else:
                peak_sun_hours = 3.5

        # 计算年发电量
        annual_generation = (
            system_capacity_kw
            * peak_sun_hours
            * 365
            * (1 - self.default_system_losses)
        )

        return round(annual_generation, 2)
```

---

## 仓储接口（Repository Interfaces）

仓储接口定义在领域层，实现在基础设施层。详细的仓储实现请参考 [05_REPOSITORY_LAYER.md](05_REPOSITORY_LAYER.md)。

### 1. 建筑仓储接口（i_building_repository.py）

```python
"""
建筑仓储接口

定义建筑数据访问的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from ..models.building import Building
from ..models.enums import BuildingType


class IBuildingRepository(ABC):
    """
    建筑仓储接口

    定义建筑数据的持久化和检索操作。
    """

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

    @abstractmethod
    def delete(self, building_id: UUID) -> bool:
        """删除建筑"""
        pass
```

### 2-3. 其他仓储接口

省略，请参考 [05_REPOSITORY_LAYER.md](05_REPOSITORY_LAYER.md)。

---

## 使用示例

### 创建和使用领域对象

```python
"""
领域层使用示例 - 使用Pydantic
"""

from pathlib import Path
from backend.domain.models import Building, BuildingType, WeatherFile
from backend.domain.models import SimulationJob, SimulationType
from backend.domain.value_objects import ECMParameters, Location

# 1. 创建建筑对象（自动验证）
building = Building(
    name="Chicago Office Building",
    building_type=BuildingType.OFFICE_LARGE,
    location="Chicago",
    idf_file_path=Path("data/chicago_office.idf"),
    floor_area=5000.0,
    num_floors=10,
    year_built=2015,
)

# 2. JSON序列化
json_str = building.model_dump_json(indent=2)
print(json_str)

# 3. 从JSON恢复
building_copy = Building.model_validate_json(json_str)

# 4. 添加元数据
building.update_metadata("architect", "John Doe")

# 5. 创建天气文件对象
weather = WeatherFile(
    file_path=Path("data/chicago_tmy.epw"),
    location="Chicago",
    scenario="TMY",
)

print(f"Weather: {weather.get_scenario_description()}")

# 6. 创建 ECM 参数（不可变）
ecm_params = ECMParameters(
    window_u_value=1.5,
    window_shgc=0.4,
    cooling_cop=4.0,
    lighting_reduction_factor=0.2,
)

# 尝试修改会报错（frozen=True）
# ecm_params.window_u_value = 2.0  # ❌ ValidationError!

# 7. 合并参数
ecm_params2 = ECMParameters(
    cooling_cop=5.0,  # 覆盖
    heating_efficiency=0.9,  # 新增
)
merged_params = ecm_params.merge(ecm_params2)

# 8. 创建模拟任务
job = SimulationJob(
    building=building,
    weather_file=weather,
    simulation_type=SimulationType.BASELINE,
    output_directory=Path("output/baseline"),
    output_prefix="chicago_baseline",
    ecm_parameters=merged_params,
)

# 9. 模拟生命周期管理
try:
    # 开始任务
    job.start()
    print(f"Job started: {job.status}")

    # ... 执行模拟（由服务层完成）...

    # 创建结果对象
    from backend.domain.models import SimulationResult

    result = SimulationResult(
        job_id=job.id,
        output_directory=job.output_directory,
    )

    # ⚠️ 重要：EUI由ResultParser计算后赋值，不在此处计算
    # 假设ResultParser已经解析了输出文件
    result.source_eui = 145.5  # 来自ResultParser
    result.site_eui = 138.2    # 来自ResultParser
    result.total_energy_kwh = 727500.0  # 来自ResultParser
    result.execution_time = 120.5
    result.success = True

    # 验证结果
    if result.is_valid():
        # 完成任务
        job.complete(result)
        print(f"Job completed: {job.status}")
        print(f"Duration: {job.get_duration()}s")
        print(f"EUI Summary: {result.get_eui_summary()}")
    else:
        job.fail("Result validation failed")

except Exception as e:
    job.fail(str(e))
    print(f"Job failed: {job.error_message}")

# 10. 数据验证示例
try:
    # ❌ 无效的floor_area会自动报错
    invalid_building = Building(
        name="Invalid Building",
        building_type=BuildingType.OFFICE_LARGE,
        location="Chicago",
        idf_file_path=Path("data/test.idf"),
        floor_area=-100,  # 负数！
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### 使用领域服务

```python
"""
领域服务使用示例
"""

from pathlib import Path
from eppy.modeleditor import IDF

from backend.domain.models import Building, BuildingType
from backend.domain.value_objects import ECMParameters, Location
from backend.domain.services import ECMApplicator, PVSystemDesigner

# ========================================
# 1. 使用 ECM 应用器
# ========================================

# 加载IDF文件
IDF.setiddname("path/to/Energy+.idd")
idf = IDF("data/chicago_office.idf")

# 创建ECM应用器
ecm_applicator = ECMApplicator()

# 定义ECM参数
ecm_params = ECMParameters(
    window_u_value=1.5,       # 低导热窗户
    window_shgc=0.4,          # 低太阳得热系数
    cooling_cop=4.5,          # 高效制冷系统
    heating_efficiency=0.92,  # 高效供热系统
    lighting_reduction_factor=0.3,  # 照明节能30%
)

# 验证并应用ECM参数
if ecm_applicator.validate(ecm_params):
    print("✓ ECM parameters are valid")

    try:
        # 应用ECM参数到IDF
        ecm_applicator.apply(idf, ecm_params)
        print("✓ ECM parameters applied successfully")

        # 保存修改后的IDF
        idf.saveas("output/chicago_office_ecm.idf")
        print("✓ Modified IDF saved")

    except RuntimeError as e:
        print(f"✗ Failed to apply ECM: {e}")
else:
    print("✗ Invalid ECM parameters")

# ========================================
# 2. 使用 PV 系统设计器
# ========================================

# 创建建筑对象
building = Building(
    name="Chicago Office",
    building_type=BuildingType.OFFICE_LARGE,
    location="Chicago",
    idf_file_path=Path("data/chicago_office.idf"),
    floor_area=5000.0,  # 5000平方米
    num_floors=10,      # 10层
)

# 创建位置对象
location = Location(
    city="Chicago",
    country="USA",
    latitude=41.8781,
    longitude=-87.6298,
)

# 创建PV系统设计器（使用默认参数）
pv_designer = PVSystemDesigner(
    default_panel_efficiency=0.20,  # 20%效率（高效面板）
    default_system_losses=0.14,     # 14%系统损失
)

# 设计PV系统
try:
    pv_design = pv_designer.design_pv_system(
        building=building,
        location=location,
        coverage_ratio=0.7,  # 70%屋顶覆盖率
    )

    # 打印设计结果
    print("\n=== PV System Design ===")
    print(f"Roof area: {pv_design['roof_area']:.2f} m²")
    print(f"Available area (70% coverage): {pv_design['available_area']:.2f} m²")
    print(f"System capacity: {pv_design['system_capacity_kw']:.2f} kW")
    print(f"Panel efficiency: {pv_design['panel_efficiency']*100:.1f}%")
    print(f"Estimated annual generation: {pv_design['estimated_annual_generation_kwh']:,.0f} kWh")

    # 计算性能指标
    annual_generation = pv_design['estimated_annual_generation_kwh']
    system_capacity = pv_design['system_capacity_kw']
    capacity_factor = annual_generation / (system_capacity * 8760)
    print(f"Capacity factor: {capacity_factor*100:.1f}%")

except ValueError as e:
    print(f"✗ PV design failed: {e}")

# ========================================
# 3. 组合使用：ECM + PV
# ========================================

def optimize_building_with_ecm_and_pv(
    building: Building,
    location: Location,
    ecm_params: ECMParameters,
    pv_coverage: float = 0.7,
) -> dict:
    """
    对建筑应用ECM和PV优化

    Args:
        building: 建筑对象
        location: 位置信息
        ecm_params: ECM参数
        pv_coverage: PV覆盖率

    Returns:
        优化结果字典
    """
    results = {}

    # 1. 应用ECM
    IDF.setiddname("path/to/Energy+.idd")
    idf = IDF(str(building.idf_file_path))

    ecm_applicator = ECMApplicator()
    if ecm_applicator.validate(ecm_params):
        ecm_applicator.apply(idf, ecm_params)
        results['ecm_applied'] = True
        results['ecm_measures'] = len(ecm_params.to_dict())
    else:
        results['ecm_applied'] = False
        results['error'] = "Invalid ECM parameters"
        return results

    # 2. 设计PV系统
    pv_designer = PVSystemDesigner()
    pv_design = pv_designer.design_pv_system(
        building=building,
        location=location,
        coverage_ratio=pv_coverage,
    )
    results['pv_design'] = pv_design

    # 3. 保存优化后的IDF
    output_path = Path("output") / f"{building.name}_optimized.idf"
    idf.saveas(str(output_path))
    results['output_file'] = output_path

    return results

# 使用组合优化
optimization_results = optimize_building_with_ecm_and_pv(
    building=building,
    location=location,
    ecm_params=ecm_params,
    pv_coverage=0.7,
)

print("\n=== Optimization Results ===")
print(f"ECM applied: {optimization_results['ecm_applied']}")
print(f"Number of ECM measures: {optimization_results['ecm_measures']}")
print(f"PV system capacity: {optimization_results['pv_design']['system_capacity_kw']:.2f} kW")
print(f"Output file: {optimization_results['output_file']}")
```

---

## 测试策略

### 单元测试示例

```python
"""
领域层单元测试 - 使用Pydantic
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from backend.domain.models import Building, BuildingType
from backend.domain.value_objects import ECMParameters


class TestBuilding:
    """建筑实体测试"""

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

    def test_create_building_invalid_floor_area(self, tmp_path):
        """测试无效楼层面积（Pydantic自动验证）"""
        idf_file = tmp_path / "test.idf"
        idf_file.touch()

        with pytest.raises(ValidationError) as exc_info:
            Building(
                name="Test",
                building_type=BuildingType.OFFICE_LARGE,
                location="Chicago",
                idf_file_path=idf_file,
                floor_area=-100,  # 负数
            )

        # 验证错误信息
        assert "floor_area" in str(exc_info.value)

    def test_create_building_invalid_file(self):
        """测试无效文件路径（自定义验证器）"""
        with pytest.raises(ValidationError) as exc_info:
            Building(
                name="Test",
                building_type=BuildingType.OFFICE_LARGE,
                location="Chicago",
                idf_file_path=Path("/nonexistent/file.idf"),
            )

        assert "does not exist" in str(exc_info.value)

    def test_json_serialization(self, tmp_path):
        """测试JSON序列化"""
        idf_file = tmp_path / "test.idf"
        idf_file.touch()

        building = Building(
            name="Test",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=idf_file,
            floor_area=1000.0,
        )

        # 序列化
        json_str = building.model_dump_json()

        # 反序列化
        building_copy = Building.model_validate_json(json_str)

        # 验证
        assert building_copy.name == building.name
        assert building_copy.floor_area == building.floor_area


class TestECMParameters:
    """ECM 参数测试"""

    def test_create_ecm_parameters(self):
        """测试创建 ECM 参数"""
        params = ECMParameters(
            window_u_value=1.5,
            window_shgc=0.4,
        )

        assert params.window_u_value == 1.5
        assert params.window_shgc == 0.4

    def test_invalid_u_value(self):
        """测试无效的 U 值（Pydantic验证）"""
        with pytest.raises(ValidationError) as exc_info:
            ECMParameters(window_u_value=15.0)  # 超出范围

        assert "window_u_value" in str(exc_info.value)

    def test_immutability(self):
        """测试不可变性（frozen=True）"""
        params = ECMParameters(window_u_value=1.5)

        with pytest.raises(ValidationError):
            params.window_u_value = 2.0  # 不能修改！

    def test_merge(self):
        """测试合并"""
        params1 = ECMParameters(window_u_value=1.5)
        params2 = ECMParameters(window_shgc=0.4)

        merged = params1.merge(params2)
        assert merged.window_u_value == 1.5
        assert merged.window_shgc == 0.4


class TestECMApplicator:
    """ECM 应用器测试"""

    def test_validate_valid_parameters(self):
        """测试验证有效的ECM参数"""
        from backend.domain.services import ECMApplicator

        applicator = ECMApplicator()
        params = ECMParameters(
            window_u_value=1.5,
            window_shgc=0.4,
        )

        assert applicator.validate(params) is True

    def test_validate_invalid_temperature_setpoints(self):
        """测试验证无效的温度设定点（制冷 <= 供热）"""
        from backend.domain.services import ECMApplicator

        applicator = ECMApplicator()
        params = ECMParameters(
            cooling_setpoint=20.0,  # 制冷设定点
            heating_setpoint=22.0,  # 供热设定点更高（无效！）
        )

        assert applicator.validate(params) is False

    def test_validate_empty_parameters(self):
        """测试验证空参数"""
        from backend.domain.services import ECMApplicator

        applicator = ECMApplicator()
        params = ECMParameters()  # 空参数

        assert applicator.validate(params) is False

    def test_apply_window_parameters(self, mocker):
        """测试应用窗户参数到IDF"""
        from backend.domain.services import ECMApplicator

        # Mock IDF对象
        mock_idf = mocker.MagicMock()
        mock_material = mocker.MagicMock()
        mock_material.Name = "Window Material"
        mock_idf.idfobjects.get.return_value = [mock_material]

        # 创建应用器和参数
        applicator = ECMApplicator()
        params = ECMParameters(
            window_u_value=1.5,
            window_shgc=0.4,
        )

        # 应用参数
        applicator.apply(mock_idf, params)

        # 验证材料属性被更新
        assert mock_material.UFactor == 1.5
        assert mock_material.Solar_Heat_Gain_Coefficient == 0.4

    def test_apply_invalid_parameters_raises_error(self):
        """测试应用无效参数抛出错误"""
        from backend.domain.services import ECMApplicator

        applicator = ECMApplicator()
        params = ECMParameters()  # 空参数（无效）

        mock_idf = None  # IDF对象（不会被使用）

        with pytest.raises(ValueError, match="Invalid ECM parameters"):
            applicator.apply(mock_idf, params)


class TestPVSystemDesigner:
    """PV 系统设计器测试"""

    def test_initialize_with_valid_parameters(self):
        """测试使用有效参数初始化"""
        from backend.domain.services import PVSystemDesigner

        designer = PVSystemDesigner(
            default_panel_efficiency=0.20,
            default_system_losses=0.14,
        )

        assert designer.default_panel_efficiency == 0.20
        assert designer.default_system_losses == 0.14

    def test_initialize_with_invalid_efficiency_raises_error(self):
        """测试使用无效效率初始化抛出错误"""
        from backend.domain.services import PVSystemDesigner

        with pytest.raises(ValueError, match="Panel efficiency must be between 0 and 1"):
            PVSystemDesigner(default_panel_efficiency=1.5)  # >1

        with pytest.raises(ValueError, match="Panel efficiency must be between 0 and 1"):
            PVSystemDesigner(default_panel_efficiency=0.0)  # =0

    def test_calculate_system_capacity(self):
        """测试计算系统容量"""
        from backend.domain.services import PVSystemDesigner

        designer = PVSystemDesigner()

        capacity = designer.calculate_system_capacity(
            roof_area=100.0,        # 100 m²
            coverage_ratio=0.7,     # 70%覆盖
            panel_efficiency=0.18,  # 18%效率
        )

        # Expected: 100 × 0.7 × 0.18 × 1000 / 1000 = 12.6 kW
        assert capacity == 12.6

    def test_design_pv_system(self, tmp_path):
        """测试设计完整的PV系统"""
        from backend.domain.services import PVSystemDesigner

        # 创建测试IDF文件
        idf_file = tmp_path / "test.idf"
        idf_file.touch()

        # 创建建筑对象
        building = Building(
            name="Test Building",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=idf_file,
            floor_area=5000.0,  # 5000 m²
            num_floors=10,      # 10层
        )

        # 创建位置对象
        from backend.domain.value_objects import Location

        location = Location(
            city="Chicago",
            country="USA",
            latitude=41.8781,
            longitude=-87.6298,
        )

        # 设计PV系统
        designer = PVSystemDesigner()
        pv_design = designer.design_pv_system(
            building=building,
            location=location,
            coverage_ratio=0.7,
        )

        # 验证设计结果
        assert pv_design['roof_area'] == 500.0  # 5000 / 10 = 500 m²
        assert pv_design['available_area'] == 350.0  # 500 × 0.7 = 350 m²
        assert pv_design['system_capacity_kw'] > 0
        assert pv_design['estimated_annual_generation_kwh'] > 0

        # 验证返回的字典包含所有必要的键
        expected_keys = [
            'roof_area',
            'available_area',
            'system_capacity_kw',
            'panel_efficiency',
            'system_losses',
            'estimated_annual_generation_kwh',
        ]
        for key in expected_keys:
            assert key in pv_design

    def test_design_pv_system_without_floor_area_raises_error(self, tmp_path):
        """测试没有楼层面积时设计PV系统抛出错误"""
        from backend.domain.services import PVSystemDesigner
        from backend.domain.value_objects import Location

        idf_file = tmp_path / "test.idf"
        idf_file.touch()

        building = Building(
            name="Test",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=idf_file,
            # floor_area 未提供
        )

        location = Location(city="Chicago", country="USA")

        designer = PVSystemDesigner()

        with pytest.raises(ValueError, match="Building floor area is required"):
            designer.design_pv_system(building, location)

    def test_estimate_annual_generation_by_latitude(self):
        """测试基于纬度估算年发电量"""
        from backend.domain.services import PVSystemDesigner
        from backend.domain.value_objects import Location

        designer = PVSystemDesigner()

        # 低纬度（5.5小时峰值日照）
        low_lat = Location(city="Singapore", country="Singapore", latitude=1.35)
        gen_low = designer._estimate_annual_generation(10.0, low_lat)

        # 中纬度（4.5小时峰值日照）
        mid_lat = Location(city="Chicago", country="USA", latitude=41.88)
        gen_mid = designer._estimate_annual_generation(10.0, mid_lat)

        # 高纬度（3.5小时峰值日照）
        high_lat = Location(city="Oslo", country="Norway", latitude=59.91)
        gen_high = designer._estimate_annual_generation(10.0, high_lat)

        # 低纬度应该有更高的发电量
        assert gen_low > gen_mid > gen_high


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 集成测试示例

```python
"""
领域层集成测试

测试多个领域对象和服务之间的交互。
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from backend.domain.models import Building, BuildingType, WeatherFile
from backend.domain.models import SimulationJob, SimulationType
from backend.domain.value_objects import ECMParameters, Location
from backend.domain.services import ECMApplicator, PVSystemDesigner


class TestSimulationJobLifecycle:
    """测试模拟任务的完整生命周期"""

    def test_complete_job_lifecycle(self, tmp_path):
        """测试从创建到完成的完整任务流程"""
        # 1. 创建建筑
        idf_file = tmp_path / "test.idf"
        idf_file.touch()

        building = Building(
            name="Test Building",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=idf_file,
            floor_area=1000.0,
        )

        # 2. 创建天气文件
        epw_file = tmp_path / "test.epw"
        epw_file.touch()

        weather = WeatherFile(
            file_path=epw_file,
            location="Chicago",
            scenario="TMY",
        )

        # 3. 创建ECM参数
        ecm_params = ECMParameters(
            window_u_value=1.5,
            cooling_cop=4.0,
        )

        # 4. 创建模拟任务
        job = SimulationJob(
            building=building,
            weather_file=weather,
            simulation_type=SimulationType.BASELINE,
            output_directory=tmp_path / "output",
            output_prefix="test",
            ecm_parameters=ecm_params,
        )

        # 5. 验证初始状态
        from backend.domain.models.enums import SimulationStatus
        assert job.status == SimulationStatus.PENDING

        # 6. 开始任务
        job.start()
        assert job.status == SimulationStatus.RUNNING
        assert job.started_at is not None

        # 7. 创建结果（模拟完成）
        from backend.domain.models import SimulationResult

        result = SimulationResult(
            job_id=job.id,
            output_directory=job.output_directory,
            source_eui=150.0,
            success=True,
        )

        # 8. 完成任务
        job.complete(result)
        assert job.status == SimulationStatus.COMPLETED
        assert job.completed_at is not None
        assert job.result == result

        # 9. 验证执行时间
        duration = job.get_duration()
        assert duration is not None
        assert duration >= 0


class TestBuildingOptimizationWorkflow:
    """测试建筑优化完整工作流"""

    def test_ecm_and_pv_optimization(self, tmp_path, mocker):
        """测试ECM和PV组合优化"""
        # 1. 创建建筑
        idf_file = tmp_path / "test.idf"
        idf_file.touch()

        building = Building(
            name="Office Building",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=idf_file,
            floor_area=5000.0,
            num_floors=10,
        )

        # 2. 创建位置
        location = Location(
            city="Chicago",
            country="USA",
            latitude=41.88,
            longitude=-87.63,
        )

        # 3. 创建ECM参数
        ecm_params = ECMParameters(
            window_u_value=1.5,
            window_shgc=0.4,
            cooling_cop=4.5,
            lighting_reduction_factor=0.3,
        )

        # 4. Mock IDF操作
        mock_idf = mocker.MagicMock()
        mock_idf.idfobjects.get.return_value = []

        # 5. 应用ECM
        ecm_applicator = ECMApplicator()
        assert ecm_applicator.validate(ecm_params)
        # 实际应用需要真实的IDF对象，这里跳过

        # 6. 设计PV系统
        pv_designer = PVSystemDesigner()
        pv_design = pv_designer.design_pv_system(
            building=building,
            location=location,
            coverage_ratio=0.7,
        )

        # 7. 验证优化结果
        assert pv_design['system_capacity_kw'] > 0
        assert pv_design['estimated_annual_generation_kwh'] > 0

        # 8. 验证ECM参数数量
        assert len(ecm_params.to_dict()) == 4


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 总结

### 使用Pydantic的领域层优势

1. **自动验证**：减少手动验证代码
2. **类型安全**：编译时 + 运行时双重检查
3. **JSON支持**：内置序列化/反序列化
4. **性能优异**：Pydantic V2使用Rust实现
5. **开发体验**：代码更简洁易读

### 设计原则重申

1. **类型安全**：100% 类型提示 + Pydantic运行时验证
2. **职责明确**：领域模型不包含硬编码的计算逻辑
3. **数据完整性**：在构造时自动验证数据
4. **不可变性**：值对象使用 `frozen=True`
5. **可测试性**：Pydantic模型易于测试

### 重要提醒

- ❌ 不要在领域模型中硬编码EUI目标值
- ✅ EUI应该由模拟服务（ResultParser）计算后存储
- ❌ 避免在领域模型中包含复杂的计算逻辑
- ✅ 领域模型专注于数据完整性和简单的业务规则

下一步：实现服务层和基础设施层。

---

**文档版本**: 2.0（Pydantic版本）
**最后更新**: 2025-10-27
**下一篇**: [02_SERVICE_LAYER.md](02_SERVICE_LAYER.md)