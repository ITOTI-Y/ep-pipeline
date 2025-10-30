# 基础设施层实现指南

> Infrastructure Layer Implementation Guide
>
> 版本：1.0
> 更新日期：2025-10-27

---

## 目录

1. [概述](#概述)
2. [设计原则](#设计原则)
3. [目录结构](#目录结构)
4. [EnergyPlus执行器](#energyplus执行器)
5. [结果解析器](#结果解析器)
6. [仓储实现](#仓储实现)
7. [缓存系统](#缓存系统)
8. [日志系统](#日志系统)
9. [工厂实现](#工厂实现)
10. [使用示例](#使用示例)
11. [测试策略](#测试策略)

---

## 概述

基础设施层（Infrastructure Layer）实现领域层和服务层定义的接口，负责与外部系统（文件系统、EnergyPlus、数据库等）的交互。

### 核心职责

1. **外部系统集成**：与EnergyPlus、文件系统等交互
2. **技术实现**：实现抽象接口的具体功能
3. **资源管理**：管理文件、连接、缓存等资源
4. **性能优化**：缓存、连接池等性能相关实现

### 设计特点

- ✅ 实现领域层定义的仓储接口
- ✅ 实现服务层定义的执行器接口
- ✅ 隔离外部依赖，便于测试和替换
- ✅ 错误处理和资源清理
- ✅ 性能优化（缓存、连接池）

---

## 设计原则

### 依赖倒置原则

```python
# ✅ 基础设施层实现领域层的接口
class FileSystemBuildingRepository(IBuildingRepository):
    """实现接口，而不是被依赖"""
    ...

# 服务层依赖抽象，而不是具体实现
class SomeService:
    def __init__(self, repo: IBuildingRepository):  # 依赖抽象
        self._repo = repo
```

### 单一职责

```python
# ✅ 每个类只负责一个外部系统
class EnergyPlusExecutor:
    """只负责执行EnergyPlus"""
    ...

class ResultParser:
    """只负责解析结果文件"""
    ...
```

---

## 目录结构

```
backend/infrastructure/
├── __init__.py
├── energyplus/                   # EnergyPlus相关
│   ├── __init__.py
│   ├── executor.py              # EnergyPlus执行器
│   ├── result_parser.py         # 结果解析器
│   └── idf_loader.py            # IDF加载器
│
├── repositories/                 # 仓储实现
│   ├── __init__.py
│   ├── filesystem_building_repo.py
│   ├── filesystem_weather_repo.py
│   └── filesystem_result_repo.py
│
├── cache/                        # 缓存实现
│   ├── __init__.py
│   ├── memory_cache.py          # 内存缓存
│   ├── disk_cache.py            # 磁盘缓存
│   └── smart_cache.py           # 智能缓存
│
└── logging/                      # 日志实现
    ├── __init__.py
    └── loguru_logger.py         # Loguru日志器
```

---

## EnergyPlus执行器

### 实现

```python
"""
EnergyPlus执行器实现

使用eppy库执行EnergyPlus模拟。
"""

from pathlib import Path
from typing import Optional
import io
import time

from eppy.modeleditor import IDF
from loguru import logger

from backend.services.interfaces import IEnergyPlusExecutor, ExecutionResult
from backend.utils.exceptions import ExecutionError


class EnergyPlusExecutor(IEnergyPlusExecutor):
    """
    EnergyPlus执行器

    使用eppy库执行EnergyPlus模拟。

    Attributes:
        _idd_path: IDD文件路径
        _energyplus_path: EnergyPlus可执行文件路径（可选，eppy会自动查找）
    """

    def __init__(
        self,
        idd_path: Path,
        energyplus_path: Optional[Path] = None,
    ):
        """
        初始化执行器

        Args:
            idd_path: IDD文件路径
            energyplus_path: EnergyPlus可执行文件路径（可选）
        """
        self._idd_path = idd_path
        self._energyplus_path = energyplus_path
        self._logger = logger

        # 设置IDD文件（eppy要求）
        IDF.setiddname(str(idd_path))

        # 验证安装
        if not self.validate_installation():
            raise ExecutionError("EnergyPlus installation validation failed")

    def run(
        self,
        idf: IDF,
        weather_file: Path,
        output_directory: Path,
        output_prefix: str,
        read_variables: bool = True,
    ) -> ExecutionResult:
        """
        运行EnergyPlus模拟

        Args:
            idf: IDF对象
            weather_file: 天气文件路径
            output_directory: 输出目录
            output_prefix: 输出文件前缀
            read_variables: 是否读取输出变量

        Returns:
            ExecutionResult: 执行结果

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
        self._logger.info(f"Starting EnergyPlus execution: {output_prefix}")

        # 创建输出目录
        output_directory.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            # 执行模拟
            idf.run(
                weather=str(weather_file),
                output_directory=str(output_directory),
                output_prefix=output_prefix,
                readvars=read_variables,
                verbose='q',  # 安静模式
            )

            execution_time = time.time() - start_time

            # 创建结果对象
            result = ExecutionResult(
                success=True,
                return_code=0,
                stdout="",
                stderr="",
                output_directory=output_directory,
            )

            # 检查错误文件
            err_file = output_directory / f"{output_prefix}out.err"
            if err_file.exists():
                self._parse_error_file(err_file, result)

            self._logger.info(
                f"EnergyPlus execution completed in {execution_time:.2f}s"
            )

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

    def validate_installation(self) -> bool:
        """
        验证EnergyPlus安装

        Returns:
            如果EnergyPlus正确安装则返回True
        """
        try:
            # 检查IDD文件
            if not self._idd_path.exists():
                self._logger.error(f"IDD file not found: {self._idd_path}")
                return False

            # 尝试创建一个空IDF来测试eppy设置
            test_idf = IDF(io.StringIO("VERSION,23.1;"))

            self._logger.info("EnergyPlus installation validated")
            return True

        except Exception as e:
            self._logger.error(f"EnergyPlus validation failed: {e}")
            return False

    def _parse_error_file(self, err_file: Path, result: ExecutionResult) -> None:
        """
        解析错误文件

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

                # 查找警告
                if "** Warning **" in content:
                    warnings = [
                        line for line in content.split('\n')
                        if "** Warning **" in line
                    ]
                    for warning in warnings[:10]:  # 限制警告数量
                        result.add_warning(warning.strip())

        except Exception as e:
            self._logger.warning(f"Failed to parse error file: {e}")
```

---

## 结果解析器

### 实现

```python
"""
结果解析器实现

解析EnergyPlus输出文件（CSV、SQL等）。
"""

from pathlib import Path
from uuid import UUID
from typing import Optional

import pandas as pd
from loguru import logger

from backend.services.interfaces import IResultParser
from backend.domain.models import SimulationResult
from backend.utils.exceptions import ParsingError


class ResultParser(IResultParser):
    """
    结果解析器

    解析EnergyPlus输出文件并提取关键指标。
    """

    def __init__(self):
        self._logger = logger

    def parse(
        self,
        job_id: UUID,
        output_directory: Path,
        output_prefix: str,
    ) -> SimulationResult:
        """
        解析模拟结果

        Args:
            job_id: 任务ID
            output_directory: 输出目录
            output_prefix: 输出文件前缀

        Returns:
            SimulationResult: 解析后的结果对象
        """
        self._logger.info(f"Parsing results for job {job_id}")

        result = SimulationResult(
            job_id=job_id,
            output_directory=output_directory,
        )

        try:
            # 解析Table CSV
            table_csv = output_directory / f"{output_prefix}Table.csv"
            if table_csv.exists():
                result.table_csv_path = table_csv
                eui_data = self.parse_eui(table_csv)

                result.source_eui = eui_data.get('source_eui')
                result.site_eui = eui_data.get('site_eui')
                result.total_energy_kwh = eui_data.get('total_energy_kwh')

            # 解析Meter CSV（如果需要）
            meter_csv = output_directory / f"{output_prefix}Meter.csv"
            if meter_csv.exists():
                result.meter_csv_path = meter_csv

            # 检查SQL文件
            sql_file = output_directory / f"{output_prefix}out.sql"
            if sql_file.exists():
                result.sql_path = sql_file

            # 标记成功
            result.success = True

            self._logger.info(
                f"Parsing completed. Source EUI: {result.source_eui} kWh/m²/yr"
            )

        except Exception as e:
            self._logger.error(f"Parsing failed: {e}")
            result.add_error(f"Parsing failed: {e}")

        return result

    def parse_eui(self, table_csv_path: Path) -> dict[str, float]:
        """
        从Table CSV解析EUI

        Args:
            table_csv_path: Table CSV文件路径

        Returns:
            包含EUI指标的字典

        Example:
            >>> parser = ResultParser()
            >>> eui = parser.parse_eui(Path("output/baselineTable.csv"))
            >>> print(eui['source_eui'])
            150.5
        """
        try:
            # 读取CSV（跳过前几行的元数据）
            df = pd.read_csv(table_csv_path, skiprows=1)

            # 查找EUI行（EnergyPlus输出格式特定）
            eui_data = {}

            # 查找"Site and Source Energy"部分
            for idx, row in df.iterrows():
                if pd.notna(row.iloc[0]):
                    row_name = str(row.iloc[0]).strip()

                    # Source EUI
                    if "Total Source Energy" in row_name:
                        # EnergyPlus输出格式：值通常在第二列
                        if len(row) > 1 and pd.notna(row.iloc[1]):
                            try:
                                eui_data['source_eui'] = float(row.iloc[1])
                            except (ValueError, IndexError):
                                pass

                    # Site EUI
                    if "Total Site Energy" in row_name:
                        if len(row) > 1 and pd.notna(row.iloc[1]):
                            try:
                                eui_data['site_eui'] = float(row.iloc[1])
                            except (ValueError, IndexError):
                                pass

            return eui_data

        except Exception as e:
            self._logger.error(f"Failed to parse EUI from {table_csv_path}: {e}")
            raise ParsingError(f"Failed to parse EUI: {e}") from e
```

---

## 仓储实现

### FileSystemBuildingRepository

```python
"""
文件系统建筑仓储实现
"""

from pathlib import Path
from typing import Optional
from uuid import UUID

from loguru import logger

from backend.domain.repositories import IBuildingRepository
from backend.domain.models import Building, BuildingType
from backend.factories import BuildingFactory


class FileSystemBuildingRepository(IBuildingRepository):
    """
    基于文件系统的建筑仓储

    从文件系统加载IDF文件并创建Building对象。

    Attributes:
        _base_directory: IDF文件基础目录
        _building_factory: 建筑工厂
        _cache: 内存缓存
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

    def save(self, building: Building) -> None:
        """
        保存建筑（缓存）

        Args:
            building: 建筑对象
        """
        self._cache[building.id] = building
        self._logger.debug(f"Cached building {building.id}")

    def find_by_id(self, building_id: UUID) -> Optional[Building]:
        """
        根据ID查找建筑

        Args:
            building_id: 建筑ID

        Returns:
            建筑对象，如果不存在则返回None
        """
        return self._cache.get(building_id)

    def find_by_type(self, building_type: BuildingType) -> list[Building]:
        """
        根据类型查找建筑

        Args:
            building_type: 建筑类型

        Returns:
            建筑对象列表
        """
        self._logger.info(f"Finding buildings of type {building_type.value}")

        # 使用glob查找匹配的IDF文件
        pattern = f"*{building_type.value}*.idf"
        idf_files = list(self._base_directory.glob(pattern))

        self._logger.info(f"Found {len(idf_files)} IDF files")

        buildings = []
        for idf_file in idf_files:
            # 从文件名提取位置
            location = self._extract_location_from_filename(idf_file.stem)

            # 使用工厂创建建筑对象
            building = self._building_factory.create_from_idf(
                idf_path=idf_file,
                building_type=building_type,
                location=location,
            )

            buildings.append(building)
            self._cache[building.id] = building

        return buildings

    def find_by_location(self, location: str) -> list[Building]:
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
            # 从文件名提取建筑类型
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
        """
        查找所有建筑

        Returns:
            所有建筑对象列表
        """
        idf_files = list(self._base_directory.glob("*.idf"))

        buildings = []
        for idf_file in idf_files:
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

    def _extract_location_from_filename(self, filename: str) -> str:
        """从文件名提取位置"""
        # 假设文件名格式：Location_BuildingType.idf
        # 例如：Chicago_OfficeLarge.idf
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

## 缓存系统

### SmartCache实现

```python
"""
智能缓存系统

结合内存缓存和磁盘缓存。
"""

import pickle
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class SmartCache:
    """
    智能缓存

    - 热数据在内存中
    - 冷数据在磁盘上
    - 自动过期管理

    Attributes:
        _cache_dir: 磁盘缓存目录
        _memory_cache: 内存缓存字典
        _max_memory_items: 最大内存缓存项数
        _default_ttl: 默认过期时间（秒）
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_items: int = 100,
        default_ttl: int = 3600,
    ):
        """
        初始化缓存

        Args:
            cache_dir: 磁盘缓存目录
            max_memory_items: 最大内存缓存项数
            default_ttl: 默认过期时间（秒）
        """
        self._cache_dir = cache_dir or Path(".cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory_cache: dict[str, tuple[Any, float]] = {}
        self._max_memory_items = max_memory_items
        self._default_ttl = default_ttl
        self._logger = logger

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        先查内存，再查磁盘。

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在或已过期则返回None
        """
        # 检查内存缓存
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
                    self._set_memory(key, value, expires_at)
                    return value
                else:
                    cache_file.unlink()

            except Exception as e:
                self._logger.warning(f"Failed to load cache: {e}")

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值

        同时保存到内存和磁盘。

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        ttl = ttl or self._default_ttl
        expires_at = time.time() + ttl

        # 设置内存缓存
        self._set_memory(key, value, expires_at)

        # 设置磁盘缓存
        cache_file = self._get_cache_file(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'expires_at': expires_at,
                }, f)
        except Exception as e:
            self._logger.warning(f"Failed to save cache: {e}")

    def delete(self, key: str) -> None:
        """删除缓存"""
        # 删除内存缓存
        if key in self._memory_cache:
            del self._memory_cache[key]

        # 删除磁盘缓存
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            cache_file.unlink()

    def clear(self) -> None:
        """清空所有缓存"""
        self._memory_cache.clear()

        for cache_file in self._cache_dir.glob("*.cache"):
            cache_file.unlink()

    def _set_memory(self, key: str, value: Any, expires_at: float) -> None:
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
        import hashlib
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"
```

---

## 日志系统

### LoguruLogger实现

```python
"""
Loguru日志实现
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


class LoguruLogger:
    """
    Loguru日志器

    配置和管理loguru日志系统。
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "7 days",
    ):
        """
        初始化日志器

        Args:
            log_dir: 日志目录
            level: 日志级别
            rotation: 日志轮转大小
            retention: 日志保留时间
        """
        self._log_dir = log_dir or Path("logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # 移除默认handler
        logger.remove()

        # 添加控制台handler（彩色输出）
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=level,
            colorize=True,
        )

        # 添加文件handler
        logger.add(
            self._log_dir / "app.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

        # 添加错误专用handler
        logger.add(
            self._log_dir / "error.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    @staticmethod
    def get_logger():
        """获取logger实例"""
        return logger
```

---

## 工厂实现

### BuildingFactory

```python
"""
建筑工厂实现
"""

from pathlib import Path
from typing import Optional

from eppy.modeleditor import IDF
from loguru import logger

from backend.domain.models import Building, BuildingType


class BuildingFactory:
    """
    建筑工厂

    从IDF文件创建Building对象。
    """

    def __init__(self):
        self._logger = logger

    def create_from_idf(
        self,
        idf_path: Path,
        building_type: BuildingType,
        location: str,
    ) -> Building:
        """
        从IDF文件创建建筑对象

        Args:
            idf_path: IDF文件路径
            building_type: 建筑类型
            location: 位置

        Returns:
            Building: 建筑对象
        """
        # 加载IDF并提取信息
        idf = IDF(str(idf_path))

        # 提取建筑面积
        floor_area = self._extract_floor_area(idf)

        # 提取楼层数
        num_floors = self._extract_num_floors(idf)

        return Building(
            name=idf_path.stem,
            building_type=building_type,
            location=location,
            idf_file_path=idf_path,
            floor_area=floor_area,
            num_floors=num_floors,
        )

    def _extract_floor_area(self, idf: IDF) -> Optional[float]:
        """从IDF提取建筑面积"""
        try:
            zones = idf.idfobjects.get('ZONE', [])
            if zones:
                # 简化：使用Zone对象的Floor Area
                total_area = sum(
                    getattr(zone, 'Floor_Area', 0.0)
                    for zone in zones
                )
                return total_area if total_area > 0 else None
        except Exception as e:
            self._logger.debug(f"Could not extract floor area: {e}")

        return None

    def _extract_num_floors(self, idf: IDF) -> Optional[int]:
        """从IDF提取楼层数"""
        try:
            # 简化：使用Zone数量作为楼层数的近似
            zones = idf.idfobjects.get('ZONE', [])
            return len(zones) if zones else None
        except Exception:
            return None
```

---

## 使用示例

```python
"""
基础设施层使用示例
"""

from pathlib import Path

from backend.infrastructure.energyplus import EnergyPlusExecutor, ResultParser
from backend.infrastructure.repositories import FileSystemBuildingRepository
from backend.infrastructure.cache import SmartCache
from backend.factories import BuildingFactory


def example_usage():
    # 1. EnergyPlus执行器
    executor = EnergyPlusExecutor(
        idd_path=Path("data/Energy+.idd")
    )

    # 2. 结果解析器
    parser = ResultParser()

    # 3. 建筑仓储
    building_factory = BuildingFactory()
    building_repo = FileSystemBuildingRepository(
        base_directory=Path("data/prototypes"),
        building_factory=building_factory,
    )

    # 查找所有办公建筑
    offices = building_repo.find_by_type(BuildingType.OFFICE_LARGE)
    print(f"Found {len(offices)} office buildings")

    # 4. 缓存
    cache = SmartCache(
        cache_dir=Path(".cache"),
        max_memory_items=50,
        default_ttl=3600,
    )

    # 使用缓存
    cache_key = "test_key"
    cache.set(cache_key, {"data": "value"}, ttl=300)
    cached_value = cache.get(cache_key)
    print(f"Cached value: {cached_value}")


if __name__ == "__main__":
    example_usage()
```

---

## 测试策略

### 单元测试

```python
"""
基础设施层单元测试
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.infrastructure.energyplus import EnergyPlusExecutor


class TestEnergyPlusExecutor:
    """EnergyPlus执行器测试"""

    @pytest.fixture
    def executor(self, tmp_path):
        idd_path = tmp_path / "Energy+.idd"
        idd_path.write_text("VERSION,23.1;")

        return EnergyPlusExecutor(idd_path=idd_path)

    def test_validate_installation(self, executor):
        """测试安装验证"""
        assert executor.validate_installation()

    @patch('eppy.modeleditor.IDF.run')
    def test_run_success(self, mock_run, executor, tmp_path):
        """测试成功执行"""
        # Arrange
        mock_idf = Mock()
        weather_file = tmp_path / "test.epw"
        weather_file.touch()
        output_dir = tmp_path / "output"

        # Act
        result = executor.run(
            idf=mock_idf,
            weather_file=weather_file,
            output_directory=output_dir,
            output_prefix="test",
        )

        # Assert
        assert result.success
        assert mock_run.called
```

---

## 总结

基础设施层实现了：

### 核心功能

1. **EnergyPlus集成**：可靠的模拟执行
2. **结果解析**：准确提取EUI等指标
3. **仓储实现**：文件系统数据访问
4. **缓存系统**：提升性能
5. **日志系统**：完整的日志记录

### 下一步

继续阅读：
- [04_APPLICATION_LAYER.md](04_APPLICATION_LAYER.md) - 应用层实现
- [05_REPOSITORY_LAYER.md](05_REPOSITORY_LAYER.md) - 仓储层详解
- [06_UTILITIES_LAYER.md](06_UTILITIES_LAYER.md) - 工具层实现

---

**文档版本**: 1.0
**最后更新**: 2025-10-27
**下一篇**: [04_APPLICATION_LAYER.md](04_APPLICATION_LAYER.md)
