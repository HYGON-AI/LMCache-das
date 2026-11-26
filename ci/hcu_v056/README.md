# v0.5.6_dev 的 PR CI

目标分支是 `v0.5.6_dev`。新建、更新、重开 PR 或从草稿转为就绪时触发。不需要打 PR 标签；这里的 label **只决定把 HCU job 派到哪个 K8s runner**。不改 main 的旧版 CI，也不发布 wheel、不跑 nightly。

## 接入 K8s

在仓库 **Settings → Secrets and variables → Actions → Variables** 配置以下非敏感变量：

| 变量 | 取值 |
| --- | --- |
| `LMCACHE_HCU_RUNNER_LABEL`（必填） | 平台已为本仓分配的、隔离的 HCU ARC scale set 的准确名称；不要填节点名、`self-hosted` 或仅打包的 `hygon-build`。填错且无匹配 runner 时 GitHub 会排队，不会自动换池。 |
| `LMCACHE_V056_CI_IMAGE`（必填） | 平台批准的 HCU 开发镜像，填完整 `registry/project/image@sha256:<64位十六进制>`；不能用旧版镜像或只填浮动 tag。 |
| `LMCACHE_V056_QWEN_MODEL`（选填） | 默认 `/models/Qwen/Qwen3-8B`。只读挂载本地可信模型；不下载权重或运行远程模型代码。 |
| `LMCACHE_V056_TP`（选填） | 默认 `8`，与本分支 README 示例一致。可填 `1/2/4/8`，必须等于该 Pod 实际获配的卡数，且显存足以加载模型。 |

镜像需能运行 Actions 的 Node、bash、git、Python 3.10–3.13 / venv / pip，包括 vLLM **0.28.1**、HIP PyTorch、DTK/hipcc、pytest、packaging，以及固定版上游 LMCache 所需的编译和运行依赖。Pod 必须有可写临时盘、足够编译空间、共享内存和至少容得下 20 GiB MP 缓存的主存。私有镜像由 K8s 拉取，不给 PR 传 registry 密码。

让平台先确认这是一次性、非特权的 PR Pod：没有用户家目录、宿主 Docker socket、K8s 令牌或生产凭据；只挂模型和任务临时盘，限制网络和设备。**runner label 不是权限控制**。保护版本分支和 CI 文件；runner group 仅允许本仓下列已审核工作流使用该资源（若平台不能限制到 workflow，则不要把共享的内网 runner 开给公开 PR）：

```text
HYGON-AI/LMCache-das/.github/workflows/hcu-v056.yml@refs/heads/v0.5.6_dev
```

按组织策略为外部 fork 开启 GitHub 原生运行审批。审批、分支保护和 runner group 是平台设置，不由本提交修改。

## 到底测什么

1. **CPU（GitHub 托管）**：核对本次合并提交的两个父 SHA，在该提交上只编一次 `lmcache-hcu==0.5.6+hcu` wheel。安装 wheel 后在源码目录外验证导入、原始补丁 CLI 和临时文件的幂等/还原；保存 JUnit、SHA256 和测试的提交 SHA。当前分支的 `PATCH_TASKS` 是空的，CLI 退出成功不代表改写过任何业务源码。
2. **HCU（K8s）**：按分支 README 的 HIP 参数从 `LMCache/LMCache@3e11e10812b448142c6cdb177a3058652bd826fa` 安装上游；在隔离 venv 安装并核验上一步的同一个 wheel，确认没有导入镜像里的旧包。检查每张分配卡能实际计算、MP connector 能导入，同时收集 PR 的 `tests/`。初始分支只有 `tests/__init__.py`，不会把它说成已有一套业务用例。
3. **Qwen MP**：在同一个 Pod 启动 LMCache MP server，冷请求完成后重启 vLLM，保留 MP server，再发同一请求。只有第二个 vLLM 进程的外部 KV 命中计数上升才通过；收到 HTTP 200、本地前缀缓存命中或“发起过查询”均不算。HCU 镜像须提供 vLLM 的 `vllm:external_prefix_cache_hits` Prometheus 计数器。失败时查看 `lmcache-v056-hcu-reports-*` 中的服务日志、JUnit、版本和 wheel 信息。
4. **hcu-tests** 汇总 runner 配置、wheel 和 HCU job；任一失败、取消或跳过都不报通过。没有照搬 0.4.2 的旧 native wheel 规则、旧用例数或 LocalDisk/POSIX 模型配置。

## 首次上线

可复用工作流固定从受信的 `@v0.5.6_dev` 读取；测试代码取事件固定的合并 SHA，控制脚本取事件固定的基线 SHA。第一次引入 CI 时，尚未合入的工作流不能自我授权调卡。可先合入 `.github/workflows/hcu-v056.yml` 和 `ci/hcu_v056/`，再用正常 PR 加入口 `.github/workflows/pr-test-hcu-v056.yml` 做首次端到端验证；若一次性合入，则用**新的**目标分支 PR 验证。无需修改默认分支 main。等首次真实运行完成后，再按 GitHub 实际显示的检查名设置必需检查。
