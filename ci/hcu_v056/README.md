# LMCache 0.5.6：PR 审批与 HCU 测试

仅验证目标分支 `v0.5.6_dev`。有贴标签权限的维护者审核当前提交后，手动添加 **`runci`**。打开、追加提交、重开 PR 和其他标签不启动本流水线。测新提交或重跑，需要摘掉标签、审核后重新贴。不要允许机器人或使用个人令牌的自动化代贴这个标签。

| Job | ARC runner（单个池名） | 内容 |
| --- | --- | --- |
| 审批 / 汇总 | `hygon-cpu` | 不检出或运行 PR 代码。审批同时核对当前标签、PR 状态、打标事件的 base/head 和 GitHub 测试合并提交的两个父 SHA。 |
| 构建 | `hygon-build` | 在固定镜像中为获批合并提交构建一次 overlay wheel；在源码目录外安装、自测，保存 SHA256、源 SHA 和 JUnit。 |
| 框架和 Qwen | `hygon-bw1100-8card` | 沿用本分支的 TP=8。按 README 的 HIP 编译参数安装固定上游 `3e11e10812b448142c6cdb177a3058652bd826fa`，安装同一次构建的 wheel，核对版本、8 卡计算、MP connector 和 PR 自带 pytest。 |

模型使用本地 Qwen3-8B：冷请求后只重启 vLLM，再发相同请求，检查外部 LMCache 命中计数确有增加。单纯服务启动、HTTP 200 或本地 prefix cache 命中都不能代替此项。初始分支没有业务 pytest，补丁任务表也为空；CLI 自测不是生产补丁已生效的证明。无发布、nightly 或自动重试。

## 上线前

1. 将受信的可复用工作流 `.github/workflows/hcu-v056.yml` 及 `ci/hcu_v056/` 合入 **`v0.5.6_dev`**。将本分支的薄入口 `.github/workflows/pr-test-hcu-v056.yml` 单独审查、同步至受保护的**默认分支 main**；它只匹配 `v0.5.6_dev`。仅推送提案分支不会启用此入口。参考 [GitHub 的 pull_request_target 事件说明](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target)。
2. 仓库 Actions Variables 必填 `LMCACHE_V056_CI_IMAGE`（平台批准的完整镜像地址及 sha256 digest）和 `LMCACHE_V056_QWEN_MODEL`（平台只读 `/public` 或 `/public4` 下的模型实际目录）。无 runner 标签变量。镜像需包含 vLLM 0.28.1、HIP PyTorch、DTK/hipcc、Python/venv、符合 pyproject 要求的 setuptools、wheel、pytest、packaging，以及固定上游编译和运行依赖。构建与测试均不在 job 中启动 Docker。
3. 只有在组织确认这几个池对公开 PR 隔离后才启用。所有容器均不得暴露用户家目录（即使只读）、生产凭据、宿主套接字或他人可写的任务缓存。挂载由平台配置；若 Pod 能看到共享 home，流水线会拒绝执行 PR 代码，请平台调整隔离，不要关闭检查。构建产物在 jobs 间用 Actions artifact 传递。
4. 保护默认分支、版本分支和 CI 代码；runner group 限制到受信的 `HYGON-AI/LMCache-das/.github/workflows/hcu-v056.yml@refs/heads/v0.5.6_dev`，不能只靠 label 控制谁能使用集群。若仓库其他 workflows 仍自动监听 PR，它们不受本入口控制。确认没有需要关闭的旁路后，用新的目标分支 PR 验证“不打标不运行、贴 runci 仅测本次 SHA、新 push 不继承批准”。不要把标签审批或默认分支的 check 当作 PR head 已成功通过硬件测试。
