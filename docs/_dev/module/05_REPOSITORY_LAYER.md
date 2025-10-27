# 仓储层实现指南

> Repository Layer Implementation Guide
>
> 版本：1.0
> 更新日期：2025-10-27

---

## 目录

1. [概述](#概述)
2. [仓储模式](#仓储模式)
3. [目录结构](#目录结构)
4. [接口定义](#接口定义)
5. [文件系统实现](#文件系统实现)
6. [数据库实现](#数据库实现)
7. [缓存集成](#缓存集成)
8. [迁移策略](#迁移策略)
9. [使用示例](#使用示例)
10. [测试策略](#测试策略)

---

## 概述

仓储层（Repository Layer）实现了领域模型与数据存储之间的抽象，提供了统一的数据访问接口。

### 核心职责

1. **数据访问抽象**：隐藏数据存储的具体实现
2. **CRUD操作**：提供标准的增删改查接口
3. **查询封装**：将复杂查询逻辑封装在仓储中
4. **数据映射**：在领域对象和存储格式之间转换
5. **缓存管理**：集成缓存以提高性能

### 设计特点

- ✅ 接口驱动，支持多种存储后端
- ✅ 领域对象与数据库解耦
- ✅ 集成缓存层
- ✅ 完整的类型提示
- ✅ 易于测试和替换

---

## 仓储模式

### 什么是仓储模式

仓储模式（Repository Pattern）是一种数据访问模式，它提供了一个类似集合的接口来访问领域对象，而不需要关心底层存储的细节。

```python
# ✅ 使用仓储模式
class SomeService:
    def __init__(self, building_repo: IBuildingRepository):
        self._repo = building_repo

    def process_building(self, building_id: UUID) -> None:
        # 通过仓储接口访问数据，不关心存储方式
        building = self._repo.find_by_id(building_id)
        # 处理业务逻辑...
        self._repo.save(building)

# ❌ 不使用仓储模式
class SomeService:
    def process_building(self, building_id: UUID) -> None:
        # 直接操作数据库，违反单一职责原则
        connection = connect_to_database()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM buildings WHERE id = ?", (building_id,))
        # ...
```

### 仓储模式的优势

1. **关注点分离**：业务逻辑与数据访问分离
2. **可测试性**：易于创建Mock实现用于测试
3. **可替换性**：可以轻松切换存储后端
4. **集中管理**：数据访问逻辑集中在一处
5. **类型安全**：返回领域对象而非原始数据

---

## 目录结构

```
backend/infrastructure/repositories/
├── __init__.py
├── interfaces/                              # 仓储接口（领域层定义）
│   ├── __init__.py
│   ├── i_building_repository.py
│   ├── i_weather_repository.py
│   └── i_result_repository.py
│
├── filesystem/                              # 文件系统实现
│   ├── __init__.py
│   ├── filesystem_building_repository.py
│   ├── filesystem_weather_repository.py
│   └── filesystem_result_repository.py
│
├── database/                                # 数据库实现（可选）
│   ├── __init__.py
│   ├── models.py                           # SQLAlchemy模型
│   ├── database_building_repository.py
│   ├── database_weather_repository.py
│   └── database_result_repository.py
│
└── cache/                                   # 缓存装饰器
    ├── __init__.py
    └── cached_repository.py
```

---

## 接口定义

### IBuildingRepository

```python
"""
建筑仓储接口

定义建筑数据访问的标准接口。
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from backend.domain.models import Building, BuildingType


class IBuildingRepository(ABC):
    """
    建筑仓储接口

    提供建筑实体的CRUD操作和查询方法。

    Example:
        >>> repo = FileSystemBuildingRepository(...)
        >>> building = repo.find_by_id(building_id)
        >>> if building:
        ...     print(building.name)
    """

    @abstractmethod
    def save(self, building: Building) -> None:
        """
        保存建筑

        Args:
            building: 建筑对象

        Raises:
            RepositoryError: 保存失败
        """
        pass

    @abstractmethod
    def find_by_id(self, building_id: UUID) -> Optional[Building]:
        """
        根据ID查找建筑

        Args:
            building_id: 建筑ID

        Returns:
            建筑对象，如果不存在则返回None
        """
        pass

    @abstractmethod
    def find_by_name(self, name: str) -> List[Building]:
        """
        根据名称查找建筑（模糊匹配）

        Args:
            name: 建筑名称

        Returns:
            建筑对象列表
        """
        pass

    @abstractmethod
    def find_by_type(self, building_type: BuildingType) -> List[Building]:
        """
        根据类型查找建筑

        Args:
            building_type: 建筑类型

        Returns:
            建筑对象列表
        """
        pass

    @abstractmethod
    def find_by_location(self, location: str) -> List[Building]:
        """
        根据位置查找建筑

        Args:
            location: 位置名称

        Returns:
            建筑对象列表
        """
        pass

    @abstractmethod
    def find_all(self) -> List[Building]:
        """
        查找所有建筑

        Returns:
            所有建筑对象列表
        """
        pass

    @abstractmethod
    def delete(self, building_id: UUID) -> bool:
        """
        删除建筑

        Args:
            building_id: 建筑ID

        Returns:
            如果删除成功则返回True
        """
        pass

    @abstractmethod
    def exists(self, building_id: UUID) -> bool:
        """
        检查建筑是否存在

        Args:
            building_id: 建筑ID

        Returns:
            如果存在则返回True
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        统计建筑总数

        Returns:
            建筑总数
        """
        pass
```

### IWeatherRepository

```python
"""
天气文件仓储接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from backend.domain.models import WeatherFile


class IWeatherRepository(ABC):
    """
    天气文件仓储接口

    提供天气文件的CRUD操作和查询方法。
    """

    @abstractmethod
    def save(self, weather: WeatherFile) -> None:
        """保存天气文件"""
        pass

    @abstractmethod
    def find_by_id(self, weather_id: UUID) -> Optional[WeatherFile]:
        """根据ID查找天气文件"""
        pass

    @abstractmethod
    def find_by_location(self, location: str) -> List[WeatherFile]:
        """根据位置查找天气文件"""
        pass

    @abstractmethod
    def find_by_scenario(self, scenario: str) -> List[WeatherFile]:
        """
        根据场景查找天气文件

        Args:
            scenario: 场景名称（如 TMY, FTMY）

        Returns:
            天气文件列表
        """
        pass

    @abstractmethod
    def find_all(self) -> List[WeatherFile]:
        """查找所有天气文件"""
        pass

    @abstractmethod
    def delete(self, weather_id: UUID) -> bool:
        """删除天气文件"""
        pass

    @abstractmethod
    def exists(self, weather_id: UUID) -> bool:
        """检查天气文件是否存在"""
        pass
```

### IResultRepository

```python
"""
结果仓储接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from backend.domain.models import SimulationResult


class IResultRepository(ABC):
    """
    结果仓储接口

    提供模拟结果的存储和查询方法。
    """

    @abstractmethod
    def save(self, result: SimulationResult) -> None:
        """
        保存模拟结果

        Args:
            result: 模拟结果对象
        """
        pass

    @abstractmethod
    def find_by_id(self, result_id: UUID) -> Optional[SimulationResult]:
        """根据ID查找结果"""
        pass

    @abstractmethod
    def find_by_job_id(self, job_id: UUID) -> Optional[SimulationResult]:
        """
        根据任务ID查找结果

        Args:
            job_id: 模拟任务ID

        Returns:
            模拟结果对象
        """
        pass

    @abstractmethod
    def find_by_building(self, building_id: UUID) -> List[SimulationResult]:
        """
        根据建筑ID查找所有结果

        Args:
            building_id: 建筑ID

        Returns:
            结果列表
        """
        pass

    @abstractmethod
    def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[SimulationResult]:
        """
        根据日期范围查找结果

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            结果列表
        """
        pass

    @abstractmethod
    def find_successful(self) -> List[SimulationResult]:
        """查找所有成功的结果"""
        pass

    @abstractmethod
    def find_failed(self) -> List[SimulationResult]:
        """查找所有失败的结果"""
        pass

    @abstractmethod
    def find_all(self) -> List[SimulationResult]:
        """查找所有结果"""
        pass

    @abstractmethod
    def delete(self, result_id: UUID) -> bool:
        """删除结果"""
        pass

    @abstractmethod
    def delete_old_results(self, days: int) -> int:
        """
        删除旧结果

        Args:
            days: 保留天数（删除超过N天的结果）

        Returns:
            删除的结果数量
        """
        pass
```

---

## 文件系统实现

### FileSystemBuildingRepository

```python
"""
基于文件系统的建筑仓储实现

从文件系统加载IDF文件并创建Building对象。
"""

from pathlib import Path
from typing import List, Optional
from uuid import UUID

from loguru import logger

from backend.domain.repositories import IBuildingRepository
from backend.domain.models import Building, BuildingType
from backend.factories import BuildingFactory
from backend.utils.exceptions import RepositoryError


class FileSystemBuildingRepository(IBuildingRepository):
    """
    文件系统建筑仓储

    从指定目录加载IDF文件，并提供查询功能。

    Attributes:
        _base_directory: IDF文件基础目录
        _building_factory: 建筑工厂
        _cache: 内存缓存 {UUID: Building}

    Example:
        >>> repo = FileSystemBuildingRepository(
        ...     base_directory=Path("data/prototypes"),
        ...     building_factory=BuildingFactory(),
        ... )
        >>> offices = repo.find_by_type(BuildingType.OFFICE_LARGE)
        >>> print(f"Found {len(offices)} office buildings")
    """

    def __init__(
        self,
        base_directory: Path,
        building_factory: BuildingFactory,
    ):
        """
        初始化仓储

        Args:
            base_directory: IDF文件基础目录
            building_factory: 建筑工厂
        """
        self._base_directory = base_directory
        self._building_factory = building_factory
        self._cache: dict[UUID, Building] = {}
        self._logger = logger

        # 验证目录存在
        if not self._base_directory.exists():
            raise RepositoryError(
                f"Base directory does not exist: {self._base_directory}"
            )

    def save(self, building: Building) -> None:
        """
        保存建筑（仅保存到缓存）

        注意：文件系统仓储是只读的，save只更新缓存。

        Args:
            building: 建筑对象
        """
        self._cache[building.id] = building
        self._logger.debug(f"Cached building {building.id}: {building.name}")

    def find_by_id(self, building_id: UUID) -> Optional[Building]:
        """
        根据ID查找建筑

        Args:
            building_id: 建筑ID

        Returns:
            建筑对象，如果不存在则返回None
        """
        return self._cache.get(building_id)

    def find_by_name(self, name: str) -> List[Building]:
        """
        根据名称查找建筑（模糊匹配）

        Args:
            name: 建筑名称

        Returns:
            建筑对象列表
        """
        self._logger.info(f"Finding buildings by name: {name}")

        # 使用glob查找匹配的IDF文件
        pattern = f"*{name}*.idf"
        idf_files = list(self._base_directory.glob(pattern))

        self._logger.info(f"Found {len(idf_files)} IDF files")

        buildings = []
        for idf_file in idf_files:
            building = self._load_building_from_file(idf_file)
            if building:
                buildings.append(building)

        return buildings

    def find_by_type(self, building_type: BuildingType) -> List[Building]:
        """
        根据类型查找建筑

        Args:
            building_type: 建筑类型

        Returns:
            建筑对象列表
        """
        self._logger.info(f"Finding buildings of type {building_type.value}")

        # 使用glob查找匹配的IDF文件
        # 假设文件名包含建筑类型，如 "Chicago_OfficeLarge.idf"
        pattern = f"*{building_type.value}*.idf"
        idf_files = list(self._base_directory.glob(pattern))

        # 也搜索递归目录
        pattern_recursive = f"**/*{building_type.value}*.idf"
        idf_files.extend(self._base_directory.glob(pattern_recursive))

        # 去重
        idf_files = list(set(idf_files))

        self._logger.info(f"Found {len(idf_files)} IDF files")

        buildings = []
        for idf_file in idf_files:
            building = self._load_building_from_file(idf_file, building_type)
            if building:
                buildings.append(building)

        return buildings

    def find_by_location(self, location: str) -> List[Building]:
        """
        根据位置查找建筑

        Args:
            location: 位置名称

        Returns:
            建筑对象列表
        """
        pattern = f"*{location}*.idf"
        idf_files = list(self._base_directory.glob(pattern))

        buildings = []
        for idf_file in idf_files:
            building = self._load_building_from_file(idf_file, location=location)
            if building:
                buildings.append(building)

        return buildings

    def find_all(self) -> List[Building]:
        """
        查找所有建筑

        Returns:
            所有建筑对象列表
        """
        idf_files = list(self._base_directory.glob("*.idf"))
        idf_files.extend(self._base_directory.glob("**/*.idf"))

        # 去重
        idf_files = list(set(idf_files))

        buildings = []
        for idf_file in idf_files:
            building = self._load_building_from_file(idf_file)
            if building:
                buildings.append(building)

        return buildings

    def delete(self, building_id: UUID) -> bool:
        """
        删除建筑（从缓存）

        Args:
            building_id: 建筑ID

        Returns:
            如果删除成功则返回True
        """
        if building_id in self._cache:
            del self._cache[building_id]
            return True
        return False

    def exists(self, building_id: UUID) -> bool:
        """
        检查建筑是否存在

        Args:
            building_id: 建筑ID

        Returns:
            如果存在则返回True
        """
        return building_id in self._cache

    def count(self) -> int:
        """
        统计建筑总数

        Returns:
            建筑总数
        """
        return len(self.find_all())

    def _load_building_from_file(
        self,
        idf_file: Path,
        building_type: Optional[BuildingType] = None,
        location: Optional[str] = None,
    ) -> Optional[Building]:
        """
        从IDF文件加载建筑

        Args:
            idf_file: IDF文件路径
            building_type: 建筑类型（如果已知）
            location: 位置（如果已知）

        Returns:
            建筑对象或None
        """
        try:
            # 从文件名提取信息
            if not building_type:
                building_type = self._extract_building_type_from_filename(idf_file.stem)

            if not location:
                location = self._extract_location_from_filename(idf_file.stem)

            # 使用工厂创建建筑对象
            building = self._building_factory.create_from_idf(
                idf_path=idf_file,
                building_type=building_type,
                location=location,
            )

            # 保存到缓存
            self._cache[building.id] = building

            return building

        except Exception as e:
            self._logger.warning(f"Failed to load building from {idf_file}: {e}")
            return None

    def _extract_location_from_filename(self, filename: str) -> str:
        """
        从文件名提取位置

        假设文件名格式：Location_BuildingType.idf
        例如：Chicago_OfficeLarge.idf

        Args:
            filename: 文件名（不含扩展名）

        Returns:
            位置名称
        """
        parts = filename.split('_')
        return parts[0] if parts else "Unknown"

    def _extract_building_type_from_filename(self, filename: str) -> BuildingType:
        """
        从文件名提取建筑类型

        Args:
            filename: 文件名（不含扩展名）

        Returns:
            建筑类型
        """
        # 尝试匹配所有建筑类型
        for building_type in BuildingType:
            if building_type.value in filename:
                return building_type

        # 默认值
        return BuildingType.OFFICE_LARGE
```

### FileSystemWeatherRepository

```python
"""
基于文件系统的天气文件仓储实现
"""

from pathlib import Path
from typing import List, Optional
from uuid import UUID

from loguru import logger

from backend.domain.repositories import IWeatherRepository
from backend.domain.models import WeatherFile
from backend.utils.exceptions import RepositoryError


class FileSystemWeatherRepository(IWeatherRepository):
    """
    文件系统天气文件仓储

    从指定目录加载EPW文件。

    Attributes:
        _base_directories: EPW文件基础目录列表（如 TMY, FTMY）
        _cache: 内存缓存

    Example:
        >>> repo = FileSystemWeatherRepository(
        ...     base_directories={
        ...         "TMY": Path("data/tmys"),
        ...         "FTMY": Path("data/ftmys"),
        ...     }
        ... )
        >>> tmy_files = repo.find_by_scenario("TMY")
    """

    def __init__(self, base_directories: dict[str, Path]):
        """
        初始化仓储

        Args:
            base_directories: 场景名到目录路径的映射
                例如: {"TMY": Path("data/tmys"), "FTMY": Path("data/ftmys")}
        """
        self._base_directories = base_directories
        self._cache: dict[UUID, WeatherFile] = {}
        self._logger = logger

        # 验证目录存在
        for scenario, directory in self._base_directories.items():
            if not directory.exists():
                self._logger.warning(
                    f"Weather directory for scenario '{scenario}' does not exist: {directory}"
                )

    def save(self, weather: WeatherFile) -> None:
        """保存天气文件（仅缓存）"""
        self._cache[weather.id] = weather

    def find_by_id(self, weather_id: UUID) -> Optional[WeatherFile]:
        """根据ID查找天气文件"""
        return self._cache.get(weather_id)

    def find_by_location(self, location: str) -> List[WeatherFile]:
        """
        根据位置查找天气文件

        Args:
            location: 位置名称

        Returns:
            天气文件列表
        """
        weather_files = []

        for scenario, directory in self._base_directories.items():
            if not directory.exists():
                continue

            # 查找匹配的EPW文件
            pattern = f"*{location}*.epw"
            epw_files = list(directory.glob(pattern))

            for epw_file in epw_files:
                weather = self._load_weather_from_file(epw_file, scenario)
                if weather:
                    weather_files.append(weather)

        return weather_files

    def find_by_scenario(self, scenario: str) -> List[WeatherFile]:
        """
        根据场景查找天气文件

        Args:
            scenario: 场景名称（如 TMY, FTMY）

        Returns:
            天气文件列表
        """
        if scenario not in self._base_directories:
            self._logger.warning(f"Unknown weather scenario: {scenario}")
            return []

        directory = self._base_directories[scenario]
        if not directory.exists():
            return []

        # 查找所有EPW文件
        epw_files = list(directory.glob("*.epw"))

        weather_files = []
        for epw_file in epw_files:
            weather = self._load_weather_from_file(epw_file, scenario)
            if weather:
                weather_files.append(weather)

        return weather_files

    def find_all(self) -> List[WeatherFile]:
        """查找所有天气文件"""
        weather_files = []

        for scenario in self._base_directories.keys():
            weather_files.extend(self.find_by_scenario(scenario))

        return weather_files

    def delete(self, weather_id: UUID) -> bool:
        """删除天气文件（从缓存）"""
        if weather_id in self._cache:
            del self._cache[weather_id]
            return True
        return False

    def exists(self, weather_id: UUID) -> bool:
        """检查天气文件是否存在"""
        return weather_id in self._cache

    def _load_weather_from_file(
        self,
        epw_file: Path,
        scenario: str,
    ) -> Optional[WeatherFile]:
        """
        从EPW文件加载天气对象

        Args:
            epw_file: EPW文件路径
            scenario: 场景名称

        Returns:
            天气文件对象或None
        """
        try:
            # 从文件名提取位置
            # 假设文件名格式：Location_Scenario.epw
            # 例如：Chicago_TMY.epw
            location = self._extract_location_from_filename(epw_file.stem)

            weather = WeatherFile(
                file_path=epw_file,
                location=location,
                scenario=scenario,
            )

            # 保存到缓存
            self._cache[weather.id] = weather

            return weather

        except Exception as e:
            self._logger.warning(f"Failed to load weather from {epw_file}: {e}")
            return None

    def _extract_location_from_filename(self, filename: str) -> str:
        """
        从文件名提取位置

        Args:
            filename: 文件名（不含扩展名）

        Returns:
            位置名称
        """
        # 移除场景后缀（如 _TMY, _FTMY）
        for suffix in ["_TMY", "_FTMY", "_2040", "_2060", "_2080"]:
            if filename.endswith(suffix):
                return filename[:-len(suffix)]

        # 否则使用下划线分割，取第一部分
        parts = filename.split('_')
        return parts[0] if parts else filename
```

### FileSystemResultRepository

```python
"""
基于文件系统的结果仓储实现
"""

from pathlib import Path
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta
import json

from loguru import logger

from backend.domain.repositories import IResultRepository
from backend.domain.models import SimulationResult


class FileSystemResultRepository(IResultRepository):
    """
    文件系统结果仓储

    将结果保存为JSON文件。

    Attributes:
        _base_directory: 结果存储基础目录
        _cache: 内存缓存

    Example:
        >>> repo = FileSystemResultRepository(
        ...     base_directory=Path("output/results")
        ... )
        >>> repo.save(result)
        >>> found = repo.find_by_job_id(job.id)
    """

    def __init__(self, base_directory: Path):
        """
        初始化仓储

        Args:
            base_directory: 结果存储基础目录
        """
        self._base_directory = base_directory
        self._cache: dict[UUID, SimulationResult] = {}
        self._logger = logger

        # 确保目录存在
        self._base_directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: SimulationResult) -> None:
        """
        保存模拟结果

        Args:
            result: 模拟结果对象
        """
        # 保存到缓存
        self._cache[result.job_id] = result

        # 保存到文件
        result_file = self._get_result_file(result.job_id)
        result_data = self._serialize_result(result)

        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            self._logger.debug(f"Saved result to {result_file}")

        except Exception as e:
            self._logger.error(f"Failed to save result: {e}")

    def find_by_id(self, result_id: UUID) -> Optional[SimulationResult]:
        """根据ID查找结果"""
        # 暂时使用 job_id 作为 result_id
        return self.find_by_job_id(result_id)

    def find_by_job_id(self, job_id: UUID) -> Optional[SimulationResult]:
        """
        根据任务ID查找结果

        Args:
            job_id: 模拟任务ID

        Returns:
            模拟结果对象
        """
        # 先查缓存
        if job_id in self._cache:
            return self._cache[job_id]

        # 从文件加载
        result_file = self._get_result_file(job_id)
        if not result_file.exists():
            return None

        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)

            result = self._deserialize_result(result_data)
            self._cache[job_id] = result

            return result

        except Exception as e:
            self._logger.error(f"Failed to load result: {e}")
            return None

    def find_by_building(self, building_id: UUID) -> List[SimulationResult]:
        """
        根据建筑ID查找所有结果

        Args:
            building_id: 建筑ID

        Returns:
            结果列表
        """
        # 遍历所有结果文件
        results = []
        for result_file in self._base_directory.glob("*.json"):
            result = self._load_result_from_file(result_file)
            if result and str(building_id) in result_file.stem:
                results.append(result)

        return results

    def find_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[SimulationResult]:
        """
        根据日期范围查找结果

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            结果列表
        """
        results = []
        for result_file in self._base_directory.glob("*.json"):
            # 检查文件修改时间
            mtime = datetime.fromtimestamp(result_file.stat().st_mtime)
            if start_date <= mtime <= end_date:
                result = self._load_result_from_file(result_file)
                if result:
                    results.append(result)

        return results

    def find_successful(self) -> List[SimulationResult]:
        """查找所有成功的结果"""
        results = []
        for result_file in self._base_directory.glob("*.json"):
            result = self._load_result_from_file(result_file)
            if result and result.success:
                results.append(result)

        return results

    def find_failed(self) -> List[SimulationResult]:
        """查找所有失败的结果"""
        results = []
        for result_file in self._base_directory.glob("*.json"):
            result = self._load_result_from_file(result_file)
            if result and not result.success:
                results.append(result)

        return results

    def find_all(self) -> List[SimulationResult]:
        """查找所有结果"""
        results = []
        for result_file in self._base_directory.glob("*.json"):
            result = self._load_result_from_file(result_file)
            if result:
                results.append(result)

        return results

    def delete(self, result_id: UUID) -> bool:
        """删除结果"""
        result_file = self._get_result_file(result_id)
        if result_file.exists():
            try:
                result_file.unlink()
                if result_id in self._cache:
                    del self._cache[result_id]
                return True
            except Exception as e:
                self._logger.error(f"Failed to delete result: {e}")

        return False

    def delete_old_results(self, days: int) -> int:
        """
        删除旧结果

        Args:
            days: 保留天数（删除超过N天的结果）

        Returns:
            删除的结果数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for result_file in self._base_directory.glob("*.json"):
            mtime = datetime.fromtimestamp(result_file.stat().st_mtime)
            if mtime < cutoff_date:
                try:
                    result_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    self._logger.error(f"Failed to delete {result_file}: {e}")

        self._logger.info(f"Deleted {deleted_count} old results")
        return deleted_count

    def _get_result_file(self, job_id: UUID) -> Path:
        """获取结果文件路径"""
        return self._base_directory / f"result_{job_id}.json"

    def _serialize_result(self, result: SimulationResult) -> dict:
        """序列化结果对象"""
        return {
            'job_id': str(result.job_id),
            'success': result.success,
            'source_eui': result.source_eui,
            'site_eui': result.site_eui,
            'total_energy_kwh': result.total_energy_kwh,
            'execution_time': result.execution_time,
            'output_directory': str(result.output_directory),
            'table_csv_path': str(result.table_csv_path) if result.table_csv_path else None,
            'meter_csv_path': str(result.meter_csv_path) if result.meter_csv_path else None,
            'sql_path': str(result.sql_path) if result.sql_path else None,
            'error_messages': result.error_messages,
            'warning_messages': result.warning_messages,
        }

    def _deserialize_result(self, data: dict) -> SimulationResult:
        """反序列化结果对象"""
        result = SimulationResult(
            job_id=UUID(data['job_id']),
            output_directory=Path(data['output_directory']),
        )

        result.success = data['success']
        result.source_eui = data.get('source_eui')
        result.site_eui = data.get('site_eui')
        result.total_energy_kwh = data.get('total_energy_kwh')
        result.execution_time = data.get('execution_time')

        if data.get('table_csv_path'):
            result.table_csv_path = Path(data['table_csv_path'])
        if data.get('meter_csv_path'):
            result.meter_csv_path = Path(data['meter_csv_path'])
        if data.get('sql_path'):
            result.sql_path = Path(data['sql_path'])

        result.error_messages = data.get('error_messages', [])
        result.warning_messages = data.get('warning_messages', [])

        return result

    def _load_result_from_file(self, result_file: Path) -> Optional[SimulationResult]:
        """从文件加载结果"""
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)

            return self._deserialize_result(result_data)

        except Exception as e:
            self._logger.warning(f"Failed to load result from {result_file}: {e}")
            return None
```

---

## 数据库实现

### SQLAlchemy模型

```python
"""
SQLAlchemy数据库模型

用于数据库仓储实现（可选）。
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class BuildingModel(Base):
    """建筑数据库模型"""

    __tablename__ = "buildings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    building_type = Column(String(50), nullable=False, index=True)
    location = Column(String(100), nullable=False, index=True)
    idf_file_path = Column(String(500), nullable=False)
    floor_area = Column(Float, nullable=True)
    num_floors = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    results = relationship("ResultModel", back_populates="building")


class WeatherModel(Base):
    """天气文件数据库模型"""

    __tablename__ = "weather_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_path = Column(String(500), nullable=False)
    location = Column(String(100), nullable=False, index=True)
    scenario = Column(String(50), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ResultModel(Base):
    """结果数据库模型"""

    __tablename__ = "simulation_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)
    building_id = Column(UUID(as_uuid=True), ForeignKey("buildings.id"), nullable=True)
    success = Column(Boolean, default=False)
    source_eui = Column(Float, nullable=True)
    site_eui = Column(Float, nullable=True)
    total_energy_kwh = Column(Float, nullable=True)
    execution_time = Column(Float, nullable=True)
    output_directory = Column(String(500), nullable=False)
    error_messages = Column(Text, nullable=True)  # JSON array as text
    warning_messages = Column(Text, nullable=True)  # JSON array as text
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # 关系
    building = relationship("BuildingModel", back_populates="results")
```

### 数据库仓储实现示例

```python
"""
基于数据库的建筑仓储实现（可选）

使用SQLAlchemy。
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from loguru import logger

from backend.domain.repositories import IBuildingRepository
from backend.domain.models import Building, BuildingType
from backend.infrastructure.repositories.database.models import BuildingModel


class DatabaseBuildingRepository(IBuildingRepository):
    """
    数据库建筑仓储

    使用SQLAlchemy与关系数据库交互。
    """

    def __init__(self, session: Session):
        """
        初始化仓储

        Args:
            session: SQLAlchemy会话
        """
        self._session = session
        self._logger = logger

    def save(self, building: Building) -> None:
        """保存建筑"""
        # 检查是否存在
        existing = self._session.query(BuildingModel).filter_by(id=building.id).first()

        if existing:
            # 更新
            existing.name = building.name
            existing.building_type = building.building_type.value
            existing.location = building.location
            existing.idf_file_path = str(building.idf_file_path)
            existing.floor_area = building.floor_area
            existing.num_floors = building.num_floors
        else:
            # 插入
            model = BuildingModel(
                id=building.id,
                name=building.name,
                building_type=building.building_type.value,
                location=building.location,
                idf_file_path=str(building.idf_file_path),
                floor_area=building.floor_area,
                num_floors=building.num_floors,
            )
            self._session.add(model)

        self._session.commit()

    def find_by_id(self, building_id: UUID) -> Optional[Building]:
        """根据ID查找建筑"""
        model = self._session.query(BuildingModel).filter_by(id=building_id).first()
        return self._model_to_entity(model) if model else None

    def find_by_type(self, building_type: BuildingType) -> List[Building]:
        """根据类型查找建筑"""
        models = self._session.query(BuildingModel).filter_by(
            building_type=building_type.value
        ).all()

        return [self._model_to_entity(model) for model in models]

    # ... 其他方法类似实现

    def _model_to_entity(self, model: BuildingModel) -> Building:
        """将数据库模型转换为领域实体"""
        from pathlib import Path

        return Building(
            id=model.id,
            name=model.name,
            building_type=BuildingType(model.building_type),
            location=model.location,
            idf_file_path=Path(model.idf_file_path),
            floor_area=model.floor_area,
            num_floors=model.num_floors,
        )
```

---

## 缓存集成

### 缓存装饰器

```python
"""
仓储缓存装饰器

为仓储添加缓存层。
"""

from typing import List, Optional
from uuid import UUID
from functools import wraps

from backend.domain.repositories import IBuildingRepository
from backend.domain.models import Building, BuildingType
from backend.infrastructure.cache import SmartCache


class CachedBuildingRepository(IBuildingRepository):
    """
    缓存建筑仓储装饰器

    在仓储前添加缓存层。

    Example:
        >>> base_repo = FileSystemBuildingRepository(...)
        >>> cached_repo = CachedBuildingRepository(
        ...     repository=base_repo,
        ...     cache=SmartCache(...),
        ... )
        >>> # 第一次查询从base_repo获取
        >>> building = cached_repo.find_by_id(building_id)
        >>> # 第二次查询从缓存获取
        >>> building = cached_repo.find_by_id(building_id)
    """

    def __init__(
        self,
        repository: IBuildingRepository,
        cache: SmartCache,
        ttl: int = 3600,
    ):
        """
        初始化缓存仓储

        Args:
            repository: 底层仓储
            cache: 缓存服务
            ttl: 缓存过期时间（秒）
        """
        self._repository = repository
        self._cache = cache
        self._ttl = ttl

    def find_by_id(self, building_id: UUID) -> Optional[Building]:
        """根据ID查找建筑（带缓存）"""
        cache_key = f"building:{building_id}"

        # 先查缓存
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # 查询底层仓储
        building = self._repository.find_by_id(building_id)

        # 缓存结果
        if building:
            self._cache.set(cache_key, building, ttl=self._ttl)

        return building

    def save(self, building: Building) -> None:
        """保存建筑（更新缓存）"""
        self._repository.save(building)

        # 更新缓存
        cache_key = f"building:{building.id}"
        self._cache.set(cache_key, building, ttl=self._ttl)

    def delete(self, building_id: UUID) -> bool:
        """删除建筑（清除缓存）"""
        result = self._repository.delete(building_id)

        # 清除缓存
        cache_key = f"building:{building_id}"
        self._cache.delete(cache_key)

        return result

    # 其他方法委托给底层仓储
    def find_by_name(self, name: str) -> List[Building]:
        return self._repository.find_by_name(name)

    def find_by_type(self, building_type: BuildingType) -> List[Building]:
        return self._repository.find_by_type(building_type)

    def find_by_location(self, location: str) -> List[Building]:
        return self._repository.find_by_location(location)

    def find_all(self) -> List[Building]:
        return self._repository.find_all()

    def exists(self, building_id: UUID) -> bool:
        return self._repository.exists(building_id)

    def count(self) -> int:
        return self._repository.count()
```

---

## 迁移策略

### 从文件系统迁移到数据库

```python
"""
仓储迁移工具
"""

from pathlib import Path
from typing import Type

from loguru import logger

from backend.domain.repositories import IBuildingRepository, IWeatherRepository
from backend.infrastructure.repositories.filesystem import (
    FileSystemBuildingRepository,
    FileSystemWeatherRepository,
)
from backend.infrastructure.repositories.database import (
    DatabaseBuildingRepository,
    DatabaseWeatherRepository,
)


class RepositoryMigrator:
    """
    仓储迁移器

    从一种存储后端迁移到另一种。

    Example:
        >>> source_repo = FileSystemBuildingRepository(...)
        >>> target_repo = DatabaseBuildingRepository(...)
        >>>
        >>> migrator = RepositoryMigrator()
        >>> migrator.migrate_buildings(source_repo, target_repo)
    """

    def __init__(self):
        self._logger = logger

    def migrate_buildings(
        self,
        source: IBuildingRepository,
        target: IBuildingRepository,
    ) -> int:
        """
        迁移建筑数据

        Args:
            source: 源仓储
            target: 目标仓储

        Returns:
            迁移的建筑数量
        """
        self._logger.info("Starting building migration...")

        buildings = source.find_all()
        self._logger.info(f"Found {len(buildings)} buildings to migrate")

        migrated = 0
        for building in buildings:
            try:
                target.save(building)
                migrated += 1
            except Exception as e:
                self._logger.error(f"Failed to migrate building {building.id}: {e}")

        self._logger.info(f"Migration completed. Migrated {migrated}/{len(buildings)} buildings")
        return migrated

    def migrate_weather(
        self,
        source: IWeatherRepository,
        target: IWeatherRepository,
    ) -> int:
        """迁移天气文件数据"""
        self._logger.info("Starting weather file migration...")

        weather_files = source.find_all()
        self._logger.info(f"Found {len(weather_files)} weather files to migrate")

        migrated = 0
        for weather in weather_files:
            try:
                target.save(weather)
                migrated += 1
            except Exception as e:
                self._logger.error(f"Failed to migrate weather {weather.id}: {e}")

        self._logger.info(f"Migration completed. Migrated {migrated}/{len(weather_files)} files")
        return migrated
```

---

## 使用示例

### 基本使用

```python
"""
仓储使用示例
"""

from pathlib import Path

from backend.infrastructure.repositories.filesystem import (
    FileSystemBuildingRepository,
    FileSystemWeatherRepository,
    FileSystemResultRepository,
)
from backend.factories import BuildingFactory


def example_usage():
    # 1. 创建建筑仓储
    building_factory = BuildingFactory()
    building_repo = FileSystemBuildingRepository(
        base_directory=Path("data/prototypes"),
        building_factory=building_factory,
    )

    # 2. 查询建筑
    offices = building_repo.find_by_type(BuildingType.OFFICE_LARGE)
    print(f"Found {len(offices)} office buildings")

    chicago_buildings = building_repo.find_by_location("Chicago")
    print(f"Found {len(chicago_buildings)} buildings in Chicago")

    # 3. 创建天气仓储
    weather_repo = FileSystemWeatherRepository(
        base_directories={
            "TMY": Path("data/tmys"),
            "FTMY": Path("data/ftmys"),
        }
    )

    # 4. 查询天气文件
    tmy_files = weather_repo.find_by_scenario("TMY")
    print(f"Found {len(tmy_files)} TMY weather files")

    # 5. 创建结果仓储
    result_repo = FileSystemResultRepository(
        base_directory=Path("output/results"),
    )

    # 6. 保存结果
    result_repo.save(simulation_result)

    # 7. 查询结果
    successful_results = result_repo.find_successful()
    print(f"Found {len(successful_results)} successful simulations")


if __name__ == "__main__":
    example_usage()
```

### 使用缓存

```python
"""
使用缓存的仓储示例
"""

from backend.infrastructure.repositories.cache import CachedBuildingRepository
from backend.infrastructure.cache import SmartCache


def example_with_cache():
    # 创建基础仓储
    base_repo = FileSystemBuildingRepository(...)

    # 创建缓存
    cache = SmartCache(
        cache_dir=Path(".cache"),
        max_memory_items=100,
        default_ttl=3600,
    )

    # 创建缓存仓储
    cached_repo = CachedBuildingRepository(
        repository=base_repo,
        cache=cache,
        ttl=3600,
    )

    # 使用缓存仓储
    # 第一次查询：从文件系统加载
    building = cached_repo.find_by_id(building_id)

    # 第二次查询：从缓存获取（快速）
    building = cached_repo.find_by_id(building_id)
```

---

## 测试策略

### 仓储单元测试

```python
"""
仓储层测试
"""

import pytest
from pathlib import Path

from backend.infrastructure.repositories.filesystem import FileSystemBuildingRepository
from backend.domain.models import BuildingType
from backend.factories import BuildingFactory


class TestFileSystemBuildingRepository:
    """文件系统建筑仓储测试"""

    @pytest.fixture
    def temp_idf_files(self, tmp_path):
        """创建临时IDF文件"""
        idf_dir = tmp_path / "idfs"
        idf_dir.mkdir()

        # 创建测试IDF文件
        (idf_dir / "Chicago_OfficeLarge.idf").write_text("VERSION,23.1;")
        (idf_dir / "Chicago_OfficeSmall.idf").write_text("VERSION,23.1;")

        return idf_dir

    @pytest.fixture
    def repository(self, temp_idf_files):
        """创建仓储实例"""
        factory = BuildingFactory()
        return FileSystemBuildingRepository(
            base_directory=temp_idf_files,
            building_factory=factory,
        )

    def test_find_by_type(self, repository):
        """测试按类型查找"""
        offices = repository.find_by_type(BuildingType.OFFICE_LARGE)
        assert len(offices) > 0
        assert all(b.building_type == BuildingType.OFFICE_LARGE for b in offices)

    def test_find_by_location(self, repository):
        """测试按位置查找"""
        chicago_buildings = repository.find_by_location("Chicago")
        assert len(chicago_buildings) > 0
        assert all("Chicago" in b.location for b in chicago_buildings)

    def test_count(self, repository):
        """测试统计"""
        count = repository.count()
        assert count >= 2  # 至少有2个测试文件


class TestCachedRepository:
    """缓存仓储测试"""

    def test_cache_hit(self):
        """测试缓存命中"""
        base_repo = Mock(spec=IBuildingRepository)
        cache = SmartCache(Path(".cache"))

        cached_repo = CachedBuildingRepository(
            repository=base_repo,
            cache=cache,
        )

        building_id = uuid4()
        test_building = Building(...)

        base_repo.find_by_id.return_value = test_building

        # 第一次查询：调用base_repo
        result1 = cached_repo.find_by_id(building_id)
        assert base_repo.find_by_id.call_count == 1

        # 第二次查询：从缓存获取，不调用base_repo
        result2 = cached_repo.find_by_id(building_id)
        assert base_repo.find_by_id.call_count == 1
        assert result1 == result2
```

---

## 总结

仓储层实现了：

### 核心特性

1. **接口驱动**：统一的数据访问接口
2. **多种实现**：文件系统、数据库等
3. **缓存集成**：提高查询性能
4. **类型安全**：完整的类型提示
5. **易于测试**：Mock友好的设计

### 扩展性

添加新仓储实现：

```python
# 1. 实现接口
class CustomBuildingRepository(IBuildingRepository):
    def save(self, building: Building) -> None:
        # 自定义实现
        pass

    # 实现其他方法...

# 2. 注册到容器
container.register_singleton('IBuildingRepository', CustomBuildingRepository(...))
```

### 下一步

继续阅读：
- [03_INFRASTRUCTURE_LAYER.md](03_INFRASTRUCTURE_LAYER.md) - 基础设施层
- [07_TESTING_STRATEGY.md](07_TESTING_STRATEGY.md) - 测试策略

---

**文档版本**: 1.0
**最后更新**: 2025-10-27
**上一篇**: [04_APPLICATION_LAYER.md](04_APPLICATION_LAYER.md)
