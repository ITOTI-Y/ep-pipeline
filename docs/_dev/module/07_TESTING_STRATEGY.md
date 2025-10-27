# 测试策略实施指南

> Testing Strategy Implementation Guide
>
> 版本：1.0
> 更新日期：2025-10-27

---

## 目录

1. [概述](#概述)
2. [测试金字塔](#测试金字塔)
3. [单元测试](#单元测试)
4. [集成测试](#集成测试)
5. [性能测试](#性能测试)
6. [测试覆盖率](#测试覆盖率)
7. [测试工具](#测试工具)
8. [持续集成](#持续集成)

---

## 概述

测试策略旨在确保代码质量、可靠性和可维护性。目标是达到**80%以上**的测试覆盖率。

### 测试目标

- ✅ **单元测试覆盖率**: >80%
- ✅ **集成测试**: 覆盖核心流程
- ✅ **性能测试**: 验证性能指标
- ✅ **回归测试**: 防止功能退化

---

## 测试金字塔

```
        /\
       /  \      E2E Tests (5%)
      /____\
     /      \    Integration Tests (15%)
    /________\
   /          \
  /____________\ Unit Tests (80%)
```

### 比例分配

- **单元测试 (80%)**: 快速、隔离、大量
- **集成测试 (15%)**: 组件交互、中等数量
- **端到端测试 (5%)**: 完整流程、少量

---

## 单元测试

### 测试框架

使用**pytest**进行单元测试。

### 基本示例

```python
"""
领域层单元测试示例
"""

import pytest
from pathlib import Path

from backend.domain.models import Building, BuildingType
from backend.domain.value_objects import ECMParameters


class TestBuilding:
    """建筑实体测试"""

    @pytest.fixture
    def temp_idf(self, tmp_path):
        """创建临时IDF文件"""
        idf_file = tmp_path / "test.idf"
        idf_file.write_text("VERSION,23.1;")
        return idf_file

    def test_create_building_success(self, temp_idf):
        """测试成功创建建筑"""
        # Arrange & Act
        building = Building(
            name="Test Building",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=temp_idf,
            floor_area=1000.0
        )

        # Assert
        assert building.name == "Test Building"
        assert building.floor_area == 1000.0
        assert building.get_identifier() == "Chicago_OfficeLarge"

    def test_create_building_invalid_file(self):
        """测试无效文件路径"""
        with pytest.raises(ValueError, match="does not exist or is not a file"):
            Building(
                name="Test",
                building_type=BuildingType.OFFICE_LARGE,
                location="Chicago",
                idf_file_path=Path("/nonexistent/file.idf"),
            )

    def test_building_negative_area(self, temp_idf):  
        """测试负面积"""  
        with pytest.raises(ValueError, match="Floor area must be positive"):  
            Building(  
                name="Test",  
                building_type=BuildingType.OFFICE_LARGE,  
                location="Chicago",  
                idf_file_path=temp_idf,  
                floor_area=-100.0,  
            ) 


class TestECMParameters:
    """ECM参数测试"""

    def test_valid_parameters(self):
        """测试有效参数"""
        params = ECMParameters(
            window_u_value=1.5,
            window_shgc=0.4,
            cooling_cop=4.0,
        )

        assert params.window_u_value == 1.5
        assert params.to_dict() == {
            'window_u_value': 1.5,
            'window_shgc': 0.4,
            'cooling_cop': 4.0,
        }

    def test_invalid_u_value(self):
        """测试无效U值"""
        with pytest.raises(ValueError, match="Invalid window U-value"):
            ECMParameters(window_u_value=15.0)

    def test_hashable(self):
        """测试可哈希性"""
        params1 = ECMParameters(window_u_value=1.5)
        params2 = ECMParameters(window_u_value=1.5)
        params3 = ECMParameters(window_u_value=2.0)

        assert hash(params1) == hash(params2)
        assert hash(params1) != hash(params3)
```

### 服务层测试

```python
"""
服务层单元测试示例
"""

from unittest.mock import Mock, patch

from backend.services.simulation import BaselineSimulationService


class TestBaselineSimulationService:
    """基准模拟服务测试"""

    @pytest.fixture
    def mock_executor(self):
        """Mock EnergyPlus执行器"""
        executor = Mock()
        executor.run.return_value = Mock(
            success=True,
            return_code=0,
            errors=[],
        )
        return executor

    @pytest.fixture
    def mock_parser(self):
        """Mock结果解析器"""
        parser = Mock()
        parser.parse.return_value = Mock(
            success=True,
            source_eui=150.0,
        )
        return parser

    @pytest.fixture
    def service(self, mock_executor, mock_parser):
        """创建服务实例"""
        return BaselineSimulationService(
            executor=mock_executor,
            parser=mock_parser,
        )

    def test_run_success(self, service, mock_context):
        """测试成功运行"""
        # Act
        result = service.run(mock_context)

        # Assert
        assert result is not None
        assert result.success
        assert result.source_eui == 150.0

    def test_prepare_validates_files(self, service, mock_context):
        """测试文件验证"""
        # Arrange - 删除IDF文件
        mock_context.job.building.idf_file_path.unlink()

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            service.prepare(mock_context)
```

---

## 集成测试

### 端到端模拟测试

```python
"""
集成测试示例
"""

import pytest
from pathlib import Path

from backend.utils.config import get_settings, setup_container


@pytest.mark.integration
class TestSimulationIntegration:
    """模拟集成测试"""

    @pytest.fixture
    def integration_setup(self):
        """集成测试设置"""
        settings = get_settings()
        container = setup_container(settings)
        return container

    def test_full_baseline_simulation(
        self,
        integration_setup,
        real_idf_file,
        real_epw_file,
        tmp_path,
    ):
        """测试完整的基准模拟流程"""
        # Arrange
        container = integration_setup
        orchestrator = container.resolve(SimulationOrchestrator)

        building = Building(
            name="Integration Test",
            building_type=BuildingType.OFFICE_LARGE,
            location="Chicago",
            idf_file_path=real_idf_file,
        )

        weather = WeatherFile(
            file_path=real_epw_file,
            location="Chicago",
            scenario="TMY",
        )

        job = SimulationJob(
            building=building,
            weather_file=weather,
            simulation_type="baseline",
            output_directory=tmp_path / "output",
            output_prefix="integration_test",
        )

        # Act
        results = orchestrator.execute_batch([job], use_cache=False)

        # Assert
        assert len(results) == 1
        result = results[0]
        assert result.success
        assert result.source_eui > 0
        assert result.execution_time > 0

        # 验证输出文件
        assert (result.output_directory / f"{job.output_prefix}Table.csv").exists()

    def test_parallel_execution(
        self,
        integration_setup,
        create_multiple_jobs,
    ):
        """测试并行执行"""
        # Arrange
        orchestrator = integration_setup.resolve(SimulationOrchestrator)
        jobs = create_multiple_jobs(count=5)

        # Act
        import time
        start = time.time()
        results = orchestrator.execute_batch(jobs)
        elapsed = time.time() - start

        # Assert
        assert len(results) == 5
        assert all(r.success for r in results)

        # 验证并行加速
        # 假设单个模拟需要60秒，5个并行应该<150秒
        assert elapsed < 150
```

---

## 性能测试

### 基准测试

```python
"""
性能测试示例
"""

import pytest
import time


@pytest.mark.performance
class TestPerformance:
    """性能测试"""

    def test_cache_hit_performance(self, orchestrator, cached_jobs):
        """测试缓存命中性能"""
        # 第一次运行（填充缓存）
        _ = orchestrator.execute_batch(cached_jobs, use_cache=True)

        # 第二次运行（应该使用缓存）
        start = time.time()
        results = orchestrator.execute_batch(cached_jobs, use_cache=True)
        elapsed = time.time() - start

        # 缓存命中应该非常快（<1秒）
        assert elapsed < 1.0
        assert all(r.success for r in results)

    def test_parallel_speedup(self, orchestrator):
        """测试并行加速比"""
        jobs = create_benchmark_jobs(count=10)

        # 顺序执行
        start = time.time()
        sequential_results = execute_sequential(jobs)
        sequential_time = time.time() - start

        # 并行执行
        start = time.time()
        parallel_results = orchestrator.execute_batch(jobs, max_workers=4)
        parallel_time = time.time() - start

        # 计算加速比
        speedup = sequential_time / parallel_time

        # 至少应该有3倍加速（理想是4倍）
        assert speedup >= 3.0
```

---

## 测试覆盖率

### 配置pytest-cov

```toml
# pyproject.toml

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=backend",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80",
    "-v",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "performance: Performance tests",
]
```

### 运行覆盖率测试

```bash
# 运行所有测试并生成覆盖率报告
pytest --cov=backend --cov-report=html --cov-report=term

# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration

# 生成详细报告
pytest --cov=backend --cov-report=html
open htmlcov/index.html
```

---

## 测试工具

### 推荐工具栈

```toml
[tool.poetry.group.test.dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.21.0"
pytest-mock = "^3.12.0"
pytest-xdist = "^3.5.0"  # 并行测试
hypothesis = "^6.92.0"    # 基于属性的测试
faker = "^20.1.0"         # 测试数据生成
```

### Hypothesis（基于属性的测试）

```python
"""
使用Hypothesis进行属性测试
"""

from hypothesis import given, strategies as st

from backend.domain.value_objects import ECMParameters


class TestECMParametersProperties:
    """ECM参数属性测试"""

    @given(
        u_value=st.floats(min_value=0.1, max_value=10.0),
        shgc=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_valid_parameters_always_create(self, u_value, shgc):
        """有效参数总是能创建对象"""
        params = ECMParameters(
            window_u_value=u_value,
            window_shgc=shgc,
        )

        assert params.window_u_value == u_value
        assert params.window_shgc == shgc

    @given(
        u_value=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_to_dict_roundtrip(self, u_value):
        """to_dict和重新创建应该保持一致"""
        original = ECMParameters(window_u_value=u_value)
        params_dict = original.to_dict()
        recreated = ECMParameters(**params_dict)

        assert recreated.window_u_value == original.window_u_value
```

---

## 持续集成

### GitHub Actions配置

```yaml
# .github/workflows/test.yml

name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install dependencies
        run: |
          poetry install --with test

      - name: Run linting
        run: |
          poetry run ruff check backend/
          poetry run mypy backend/

      - name: Run tests
        run: |
          poetry run pytest --cov=backend --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

---

## 测试最佳实践

### 1. AAA模式

```python
def test_something():
    # Arrange - 准备测试数据
    input_data = create_test_data()

    # Act - 执行被测代码
    result = function_under_test(input_data)

    # Assert - 验证结果
    assert result == expected_value
```

### 2. 使用Fixtures

```python
@pytest.fixture
def sample_building(tmp_path):
    """可重用的建筑fixture"""
    idf_file = tmp_path / "test.idf"
    idf_file.write_text("VERSION,23.1;")

    return Building(
        name="Test Building",
        building_type=BuildingType.OFFICE_LARGE,
        location="Chicago",
        idf_file_path=idf_file,
    )
```

### 3. 参数化测试

```python
@pytest.mark.parametrize("u_value,expected_valid", [
    (0.5, True),
    (1.5, True),
    (5.0, True),
    (15.0, False),  # 超出范围
])
def test_window_u_value_validation(u_value, expected_valid):
    """参数化测试U值验证"""
    if expected_valid:
        params = ECMParameters(window_u_value=u_value)
        assert params.window_u_value == u_value
    else:
        with pytest.raises(ValueError):
            ECMParameters(window_u_value=u_value)
```

---

## 测试检查清单

### 每个功能的测试

- [ ] **正常情况测试**: 验证正常输入的行为
- [ ] **边界情况测试**: 测试边界值
- [ ] **异常情况测试**: 测试错误处理
- [ ] **集成测试**: 测试与其他组件的交互

### 代码覆盖率

- [ ] 行覆盖率 > 80%
- [ ] 分支覆盖率 > 70%
- [ ] 核心业务逻辑 100%

### 性能测试

- [ ] 基准测试
- [ ] 并行加速验证
- [ ] 内存使用测试

---

## 总结

测试策略确保：

1. **高覆盖率**: >80%测试覆盖
2. **快速反馈**: 单元测试快速执行
3. **可靠性**: 集成测试验证交互
4. **性能保证**: 性能测试验证指标
5. **持续集成**: CI/CD自动化

---

**文档版本**: 1.0
**最后更新**: 2025-10-27
**下一篇**: [08_PERFORMANCE_OPTIMIZATION.md](08_PERFORMANCE_OPTIMIZATION.md)
