# Contributing

感谢参与本项目！

## 开发流程

1. Fork 并克隆本仓库。
2. 创建特性分支：`git checkout -b feat/your-feature`。
3. 安装开发依赖：`pip install -e ".[dev]"`。
4. 编写代码并补充测试。
5. 确保本地通过：

```bash
pytest -q
ruff check engine
ruff format --check engine
```

6. 提交并发起 Pull Request，描述改动与验证方式。

## 提交规范

建议使用约定式提交：

- `feat:` 新功能
- `fix:` 缺陷修复
- `perf:` 性能优化
- `refactor:` 重构
- `docs:` 文档
- `test:` 测试
- `chore:` 工程/构建

## 报告问题

提 issue 时请附上：环境（OS、GPU、CUDA、vLLM/TensorRT-LLM 版本）、复现步骤、
期望与实际表现。
