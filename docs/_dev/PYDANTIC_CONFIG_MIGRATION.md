# Pydantic配置类迁移指南

> 从dataclass迁移到Pydantic BaseModel
>
> 更新日期：2025-11-02

---

## 概述

本文档说明了配置管理类从Python dataclass迁移到Pydantic BaseModel的原因、过程和最佳实践。

---

## 为什么迁移到Pydantic

### 1. 自动验证

**dataclass（需要手动验证）：**
```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathsConfig:
    prototype_idf: Path
    
    def __post_init__(self):
        if not self.prototype_idf.exists():
            raise ValueError(f"文件不存在: {self.prototype_idf}")
        if self.prototype_idf.suffix != '.idf':
            raise ValueError(f"必须是IDF文件")
```

**Pydantic（自动验证）：**
```python
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

class PathsConfig(BaseModel):
    prototype_idf: Path = Field(..., description="原型IDF文件路径")
    
    @field_validator('prototype_idf')
    @classmethod
    def validate_idf_file(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"文件不存在: {v}")
        if v.suffix.lower() != '.idf':
            raise ValueError(f"必须是IDF文件，当前为: {v.suffix}")
        return v
```

### 2. 架构一致性

项目中的领域模型已经使用Pydantic：
- `Building` (backend/domain/models/building.py)
- `SimulationContext` (backend/services/simulation/simulation_context.py)
- `Location` (backend/domain/value_objects/location.py)
- `ECMParameters` (backend/domain/value_objects/ecm_parameters.py)

配置类使用Pydantic保持架构一致性。

### 3. 更好的错误信息

**dataclass错误：**
```
ValueError: 文件不存在: /path/to/file.idf
```

**Pydantic错误：**
```json
{
  "type": "value_error",
  "loc": ["prototype_idf"],
  "msg": "文件不存在: /path/to/file.idf",
  "input": "/path/to/file.idf",
  "ctx": {"error": "文件不存在"}
}
```

### 4. 自动类型转换

```python
from pydantic import BaseModel
from pathlib import Path

class Config(BaseModel):
    path: Path
    count: int

# 自动将字符串转换为Path和int
config = Config(path="/tmp/data", count="42")
assert isinstance(config.path, Path)  # True
assert config.count == 42  # True
```

### 5. 跨字段验证

```python
from pydantic import BaseModel, model_validator

class SimulationConfig(BaseModel):
    start_year: int
    end_year: int
    
    @model_validator(mode='after')
    def validate_year_range(self) -> 'SimulationConfig':
        if self.start_year > self.end_year:
            raise ValueError(
                f"开始年份({self.start_year})不能大于结束年份({self.end_year})"
            )
        return self
```

---

## 迁移对比

### PathsConfig

**之前（dataclass）：**
```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathsConfig:
    prototype_idf: Path
    tmy_dir: Path
    output_dir: Path
    eplus_executable: Path
    idd_file: Path
```

**之后（Pydantic）：**
```python
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

class PathsConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        frozen=False,
    )
    
    prototype_idf: Path = Field(..., description="原型IDF文件路径")
    tmy_dir: Path = Field(..., description="TMY天气文件目录")
    output_dir: Path = Field(..., description="输出根目录")
    eplus_executable: Path = Field(..., description="EnergyPlus可执行文件路径")
    idd_file: Path = Field(..., description="IDD文件路径")
    
    @field_validator('eplus_executable', 'idd_file')
    @classmethod
    def validate_file_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"文件不存在: {v}")
        return v
    
    @field_validator('prototype_idf')
    @classmethod
    def validate_idf_file(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"IDF文件不存在: {v}")
        if v.suffix.lower() != '.idf':
            raise ValueError(f"文件必须是IDF格式")
        return v
```

### SimulationConfig

**之前（dataclass）：**
```python
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    start_year: int
    end_year: int
    default_output_suffix: str
    cleanup_files: list[str]
```

**之后（Pydantic）：**
```python
from pydantic import BaseModel, Field, model_validator

class SimulationConfig(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=False,
    )
    
    start_year: int = Field(..., ge=1900, le=2100, description="模拟开始年份")
    end_year: int = Field(..., ge=1900, le=2100, description="模拟结束年份")
    default_output_suffix: str = Field(
        default="L",
        min_length=1,
        max_length=10,
        description="默认输出文件后缀"
    )
    cleanup_files: list[str] = Field(
        default_factory=lambda: ['.audit', '.bnd', '.eio'],
        description="需要清理的文件扩展名列表"
    )
    
    @model_validator(mode='after')
    def validate_year_range(self) -> 'SimulationConfig':
        if self.start_year > self.end_year:
            raise ValueError("开始年份不能大于结束年份")
        return self
```

---

## 与OmegaConf集成

Pydantic和OmegaConf完美配合：

```python
from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, ValidationError

# 1. 使用OmegaConf加载和合并配置
config_files = ["base.yaml", "override.yaml"]
configs = [OmegaConf.load(f) for f in config_files]
merged = OmegaConf.merge(*configs)

# 2. 转换为字典（解析变量插值）
config_dict = OmegaConf.to_container(merged, resolve=True)

# 3. 使用Pydantic验证
try:
    paths_config = PathsConfig(**config_dict['paths'])
    sim_config = SimulationConfig(**config_dict['simulation'])
except ValidationError as e:
    print("配置验证失败:")
    for error in e.errors():
        print(f"  {error['loc']}: {error['msg']}")
```

**优势：**
- OmegaConf: 配置合并、变量插值、YAML支持
- Pydantic: 自动验证、类型安全、详细错误

---

## 最佳实践

### 1. 使用Field()添加元数据

```python
from pydantic import BaseModel, Field

class Config(BaseModel):
    port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="服务器端口号"
    )
```

### 2. 使用field_validator进行复杂验证

```python
from pydantic import BaseModel, field_validator
from pathlib import Path

class Config(BaseModel):
    data_dir: Path
    
    @field_validator('data_dir')
    @classmethod
    def validate_and_create_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v
```

### 3. 使用model_validator进行跨字段验证

```python
from pydantic import BaseModel, model_validator

class Config(BaseModel):
    min_value: int
    max_value: int
    
    @model_validator(mode='after')
    def validate_range(self) -> 'Config':
        if self.min_value >= self.max_value:
            raise ValueError("min_value必须小于max_value")
        return self
```

### 4. 配置model_config

```python
from pydantic import BaseModel, ConfigDict

class Config(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,  # 赋值时也验证
        arbitrary_types_allowed=True,  # 允许Path等类型
        frozen=False,  # 是否不可变
        str_strip_whitespace=True,  # 自动去除空格
    )
```

---

## 迁移检查清单

- [x] 将`@dataclass`改为`class XXX(BaseModel)`
- [x] 添加`model_config = ConfigDict(...)`
- [x] 为字段添加`Field()`描述
- [x] 添加`@field_validator`进行字段验证
- [x] 添加`@model_validator`进行跨字段验证
- [x] 更新ConfigManager的解析方法
- [x] 添加配置文件示例
- [x] 更新文档说明

---

## 参考资料

- [Pydantic官方文档](https://docs.pydantic.dev/)
- [OmegaConf官方文档](https://omegaconf.readthedocs.io/)
- `docs/_dev/REFACTORING_GUIDE.md` - 完整的配置管理实现
- `docs/_dev/module/01_DOMAIN_LAYER.md` - Pydantic在领域层的使用
- `docs/_dev/module/02_SERVICE_LAYER.md` - Pydantic在服务层的使用

