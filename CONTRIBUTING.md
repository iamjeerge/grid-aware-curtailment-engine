# Contributing Guide

Thank you for your interest in contributing to the Grid-Aware Curtailment Engine! This guide will help you get started.

---

## 🎯 Ways to Contribute

### 1. Code Contributions
- **Bug fixes**: Identify and fix issues
- **New features**: Extend functionality
- **Optimization**: Improve performance or accuracy
- **Testing**: Add more test coverage

### 2. Documentation
- **Examples**: Add use case examples
- **Tutorials**: Create learning materials
- **API docs**: Improve reference documentation
- **Translations**: Translate to other languages

### 3. Community
- **Help others**: Answer questions in discussions
- **Share use cases**: Post about how you're using GACE
- **Report bugs**: File detailed issue reports
- **Request features**: Suggest improvements

---

## 🚀 Getting Started as a Contributor

### Step 1: Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/iamjeerge/grid-aware-curtailment-engine.git
cd grid-aware-curtailment-engine

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
poetry install

# Set up Git hooks (for code quality)
pre-commit install

# Verify setup
poetry run pytest tests/test_domain.py -v
```

### Step 2: Understand the Codebase

1. **Read the docs**: Start with [Documentation Index](INDEX.md)
2. **Explore architecture**: Review [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Understand the problem**: Read [README Problem Statement](../README.md#-problem-statement)
4. **Run the demo**: Execute `docker-compose up` and try the UI

### Step 3: Create a Branch

```bash
# Create feature branch with descriptive name
git checkout -b feature/add-rl-curriculum-learning
# or
git checkout -b bugfix/fix-battery-soc-calculation
# or
git checkout -b docs/add-sensitivity-analysis-tutorial
```

**Branch naming convention**:
- `feature/description` - New functionality
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation
- `test/description` - Test improvements
- `perf/description` - Performance optimizations

---

## 📝 Code Contribution Guidelines

### Code Quality Standards

All code must pass:

```bash
# Format with Black (100 char lines)
poetry run black src/ tests/

# Lint with Ruff
poetry run ruff check src/ tests/

# Type check with MyPy (strict mode)
poetry run mypy . --strict

# Run tests with coverage
poetry run pytest tests/ --cov=src --cov-threshold=85
```

### Adding a New Feature

#### 1. Create the Feature

Example: Adding a new optimization strategy

```python
# src/controllers/new_strategy.py

"""
New Optimization Strategy

Implements [strategy name] for [specific use case].
Based on [reference paper if applicable].

Key features:
- [Feature 1]
- [Feature 2]
- [Feature 3]
"""

from pydantic import BaseModel
from src.domain import TimeStep, OptimizationDecision

class NewStrategyController:
    """
    Implements [strategy name].
    
    This strategy [brief description of approach].
    
    Usage:
        controller = NewStrategyController(param1=value1)
        decision = controller.decide(state, forecast)
    
    Pros:
        - [Pro 1]
        - [Pro 2]
    
    Cons:
        - [Con 1]
        - [Con 2]
    
    References:
        - [Paper 1]
        - [Paper 2]
    """
    
    def __init__(self, param1: float, param2: int):
        """Initialize with hyperparameters."""
        self.param1 = param1
        self.param2 = param2
    
    def decide(self, state: TimeStep, forecast: list[float]) -> OptimizationDecision:
        """Make optimization decision for current timestep.
        
        Args:
            state: Current system state
            forecast: 24-hour ahead forecast
        
        Returns:
            Decision with sell, store, and curtail amounts
        
        Raises:
            ValueError: If forecast invalid or constraints violated
        """
        # Implementation...
        pass
```

#### 2. Write Tests

```python
# tests/test_new_strategy.py

import pytest
from src.controllers import NewStrategyController
from src.domain import TimeStep

class TestNewStrategyController:
    """Tests for new strategy."""
    
    @pytest.fixture
    def controller(self):
        return NewStrategyController(param1=1.0, param2=100)
    
    @pytest.fixture
    def sunny_day(self):
        """Standard sunny day scenario."""
        return {
            "generation": [100, 200, 400, 600, 500, 300, 100, 50],
            "grid_capacity": 300,
            "prices": [50, 40, -20, -25, 60, 100, 80, 60],
        }
    
    def test_respects_grid_constraints(self, controller, sunny_day):
        """Verify strategy never exceeds grid capacity."""
        for t in range(len(sunny_day["generation"])):
            decision = controller.decide(
                state=TimeStep(hour=t, generation_mw=sunny_day["generation"][t]),
                forecast=sunny_day["generation"],
            )
            assert decision.sell_mw <= sunny_day["grid_capacity"]
    
    def test_curtailment_when_necessary(self, controller):
        """Verify strategy curtails when generation > capacity + storage."""
        decision = controller.decide(...)
        assert decision.curtail_mw > 0  # Some curtailment
        assert decision.sell_mw == 300  # At grid limit
    
    def test_improves_on_naive(self, controller, sunny_day):
        """Verify strategy beats naive baseline."""
        optimized_revenue = controller.run(sunny_day)
        naive_revenue = naive_strategy.run(sunny_day)
        assert optimized_revenue > naive_revenue
```

**Testing Requirements**:
- Minimum 85% code coverage
- Test happy path and edge cases
- Test constraint violations
- Compare against baselines
- Include docstrings explaining test purpose

#### 3. Document the Feature

Add to [FEATURES.md](FEATURES.md) under appropriate section:

```markdown
### 5. Your New Strategy

**What it does**: Brief description

**Algorithm**:
```python
Your algorithm in pseudocode
```

**When to use**: Use cases

**Advantages**:
- Advantage 1
- Advantage 2

**Disadvantages**:
- Disadvantage 1
- Disadvantage 2

**Example**:
```python
from src.controllers import NewStrategyController

controller = NewStrategyController(param1=1.0)
decision = controller.decide(state, forecast)
```
```

#### 4. Add to API (if user-facing)

If your feature should be accessible via the web UI or API:

```python
# src/api/services.py

class OptimizationService:
    async def run_optimization(self, scenario, strategies):
        if "new_strategy" in strategies:
            controller = NewStrategyController()
            result = controller.decide(...)
            # Store result
```

#### 5. Submit for Review

```bash
# Commit your changes
git add -A
git commit -m "Add new strategy: [strategy name]

- Brief description of changes
- Reference issue #123 if applicable
- Mention any new dependencies

Testing:
- All tests pass: 95% coverage
- Code quality: black, ruff, mypy all pass
- Compared against baselines
"

# Push to GitHub
git push origin feature/add-rl-curriculum-learning
```

---

### Modifying Core Components

#### Modifying Battery Physics

If you improve the battery model:

1. **Update domain model** (`src/domain/battery.py`)
2. **Update constraints** (MILP formulation + RL environment)
3. **Add tests** for new physics
4. **Update documentation**:
   - [FEATURES.md](FEATURES.md)
   - [README configuration](../README.md#-configuration)
   - Code docstrings

Example:

```python
# src/domain/battery.py

class BatteryState(BaseModel):
    """
    State of Battery Energy Storage System.
    
    Attributes:
        soc_percent: State of charge (10-90%)
        charge_efficiency: Charging efficiency (default: 0.95)
        discharge_efficiency: Discharging efficiency (default: 0.95)
        degradation_cost_per_mwh: Wear cost per MWh cycled (default: $8)
        
    Note:
        This model assumes ideal electrochemistry. Real systems experience:
        - Temperature-dependent efficiency
        - Cycle-dependent degradation curves
        - Frequency response losses (if applicable)
        
        See src/validation/assumptions.py for complete assumption list.
    """
```

#### Modifying Optimization Formulation

If you improve the MILP formulation:

1. **Update formulation** (`src/optimization/formulation.py`)
2. **Document changes** with equations
3. **Validate with tests**: Must be constraint-feasible
4. **Benchmark**: Compare solution quality and solve time
5. **Update assumptions**: Document any new assumptions

---

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/test_feature.py

import pytest
from src.feature import MyClass

class TestMyClass:
    """Tests for MyClass."""
    
    @pytest.fixture
    def instance(self):
        """Fixture: standard instance."""
        return MyClass(param1=1.0, param2=100)
    
    @pytest.fixture
    def sample_data(self):
        """Fixture: sample input data."""
        return {...}
    
    # Happy path
    def test_normal_operation(self, instance, sample_data):
        """Should work with valid inputs."""
        result = instance.process(sample_data)
        assert result is not None
        assert result.is_valid
    
    # Edge cases
    def test_empty_input(self, instance):
        """Should handle empty input gracefully."""
        with pytest.raises(ValueError):
            instance.process([])
    
    # Constraints
    def test_respects_grid_limits(self, instance):
        """Should never exceed grid capacity."""
        result = instance.process(...)
        assert result.power_mw <= MAX_CAPACITY
    
    # Baseline comparison
    def test_beats_naive_baseline(self, instance, sample_data):
        """Should outperform naive strategy."""
        optimized = instance.process(sample_data)
        naive = naive_strategy.process(sample_data)
        assert optimized.profit > naive.profit
```

### Running Tests

```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_battery.py -v

# Run specific test
poetry run pytest tests/test_battery.py::TestBatteryState::test_soc_bounds -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html

# Run only fast tests (skip integration tests)
poetry run pytest tests/ -m "not integration"
```

---

## 📚 Documentation Guidelines

### Code Docstrings

Use Google-style docstrings:

```python
def optimize(
    generation: list[float],
    prices: list[float],
    grid_capacity: float,
) -> OptimizationDecision:
    """
    Compute optimal dispatch decision.
    
    This function solves a mixed-integer linear program to find the
    optimal balance between selling to grid, charging battery, and
    curtailing excess generation.
    
    Args:
        generation: Hourly solar forecast (MW), length 24
        prices: Hourly market prices ($/MWh), length 24
        grid_capacity: Maximum export capacity (MW)
    
    Returns:
        OptimizationDecision with hourly sell, store, curtail amounts
    
    Raises:
        ValueError: If inputs invalid (negative values, wrong length)
        SolverError: If solver fails to find solution
    
    Note:
        - Assumes price-taker (doesn't affect market prices)
        - Requires accurate forecast for optimal results
        - Computation time ~5-10 seconds for 24-hour horizon
    
    Example:
        >>> decision = optimize(
        ...     generation=[600, 550, 500],
        ...     prices=[50, -25, 60],
        ...     grid_capacity=300,
        ... )
        >>> decision.total_revenue
        8750
    
    References:
        - CAISO Operations: https://www.caiso.com/market/Pages/default.aspx
        - MILP Tutorial: https://www.gurobi.com/resource/mip-basics/
    """
```

### Feature Documentation

Add features to [FEATURES.md](FEATURES.md):

1. **"What it does"**: One sentence summary
2. **"Algorithm"**: Pseudocode or mathematical formulation
3. **"When to use"**: Specific use cases
4. **"How it works"**: Step-by-step explanation
5. **"Advantages/Disadvantages"**: Pros and cons
6. **"Example"**: Working code example
7. **"References"**: Related papers or docs

---

## 🔄 Pull Request Process

### Before Submitting

```bash
# 1. Format code
poetry run black .

# 2. Lint
poetry run ruff check .

# 3. Type check
poetry run mypy . --strict

# 4. Run tests
poetry run pytest tests/ -v

# 5. Check coverage
poetry run pytest --cov=src --cov-report=term-missing
```

### Submission Checklist

- [ ] Feature is complete and tested
- [ ] All tests pass (85%+ coverage)
- [ ] Code follows style guide (black, ruff, mypy)
- [ ] Documentation updated
- [ ] Docstrings added to new functions
- [ ] Example added to docs/FEATURES.md
- [ ] No breaking changes to public API
- [ ] Commit messages are clear
- [ ] Branch is rebased on latest main

### Pull Request Description

```markdown
## Description
Brief description of what this PR does

## Problem
What problem does this solve?

## Solution
How does this PR solve it?

## Testing
- [ ] Unit tests added
- [ ] Integration tests passed
- [ ] Manual testing completed
- [ ] Coverage: 85%+

## Documentation
- [ ] Docstrings added
- [ ] Feature guide updated
- [ ] API docs updated (if applicable)
- [ ] Example added

## Breaking Changes
None / Description of breaking changes

## Benchmark (if applicable)
Performance before: X sec
Performance after: Y sec
Improvement: Z%

Fixes #123
References #456
```

---

## 🐛 Bug Report Guidelines

### Report a Bug

When filing a bug, include:

1. **Description**: What's the bug?
2. **Steps to reproduce**: How to trigger it?
3. **Expected behavior**: What should happen?
4. **Actual behavior**: What actually happens?
5. **Environment**: OS, Python version, poetry version
6. **Logs**: Error messages and stack traces
7. **Reproducible example**:

```python
# Minimal code to reproduce the bug
from src.feature import MyClass

obj = MyClass()
result = obj.broken_method()  # This fails!
# Error: ...
```

---

## 🏆 Code Review Expectations

All PRs require:
- ✅ At least 1 approval from core maintainers
- ✅ All CI checks passing (tests, linting, types)
- ✅ 85%+ code coverage
- ✅ Clear documentation

---

## 📦 Adding Dependencies

Before adding a new dependency:

1. **Check if it's already used**: Search `pyproject.toml`
2. **Justify the addition**: Why is it needed?
3. **Check alternatives**: Are there lighter alternatives?
4. **Update documentation**: Add to architecture docs

```bash
# Add to pyproject.toml via poetry
poetry add new-package

# Then commit
git add poetry.lock pyproject.toml
git commit -m "Add new-package for [feature]"
```

---

## 🚀 Release Process

The maintainers follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Contributors don't need to handle releases, but should note breaking changes in PRs.

---

## 💬 Questions & Discussions

- **Ask questions**: Use GitHub Discussions
- **Share ideas**: Suggest features before implementing
- **Give feedback**: Review others' PRs
- **Help others**: Answer questions in issues

---

## 📊 Contribution Statistics

Track your contributions:
- Check [Contributors](https://github.com/iamjeerge/grid-aware-curtailment-engine/graphs/contributors)
- View your commits: `git log --author="Your Name"`
- See impact: Check which PRs got merged

---

## 🎓 Learning Resources

### Optimization
- [Pyomo Documentation](http://www.pyomo.org/)
- [GLPK Manual](https://www.gnu.org/software/glpk/)
- [MILP Tutorial](https://www.gurobi.com/resource/mip-basics/)

### Reinforcement Learning
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [OpenAI PPO Paper](https://arxiv.org/abs/1707.06347)
- [Deep RL Course](http://rail.eecs.berkeley.edu/deeprlcourse/)

### Energy Systems
- [CAISO Operations](https://www.caiso.com/)
- [Smart Grid Handbook](https://smartgridcc.org/)

---

## 🙏 Thank You!

Contributing to GACE helps advance renewable energy optimization. Your efforts are appreciated!

---

**Happy coding! 🚀⚡📊**

