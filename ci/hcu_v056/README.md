# LMCache 0.5.6：PR 审批与 HCU 测试

仅验证目标分支 `v0.5.6_dev`。有贴标签权限的维护者审核当前提交后，手动添加 **`runci`**。打开、追加提交、重开 PR 和其他标签不启动本流水线。测新提交或重跑，需要摘掉标签、审核后重新贴。不要允许机器人或使用个人令牌的自动化代贴这个标签。

| Job | ARC runner（单个池名） | 内容 |
| --- | --- | --- |
| 审批 / 汇总 | `hygon-cpu` | 不检出或运行 PR 代码。审批同时核对当前标签、PR 状态、打标事件的 base/head 和 GitHub 测试合并提交的两个父 SHA。 |
| 构建 | `hygon-build` | 在固定镜像中为获批合并提交构建一次 overlay wheel；在源码目录外安装、自测，保存 SHA256、源 SHA 和 JUnit。 |
| HCU 单测 | `hygon-bw1100-1card` | 只分配 1 卡，检查 HIP 计算、MP connector 和 PR 自带 pytest，不启动模型。 |
| Qwen 模型 | `hygon-bw1100-2card` | Qwen3-8B、TP=2；取与单测相同的 PR wheel 和固定上游包。只测缓存复用，不测精度。 |

模型使用本地 Qwen3-8B：冷请求后只重启 vLLM，再发相同请求，检查外部 LMCache 命中计数确有增加。单纯服务启动、HTTP 200 或本地 prefix cache 命中都不能代替此项。初始分支没有业务 pytest，补丁任务表也为空；CLI 自测不是生产补丁已生效的证明。无发布、nightly 或自动重试。

## 上线前

Qwen 默认只读目录是 `/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-8B`；平台挂载路径不同时才需设置 `LMCACHE_V056_QWEN_MODEL` 覆盖。

1. 将受信的可复用工作流 `.github/workflows/hcu-v056.yml` 及 `ci/hcu_v056/` 合入 **`v0.5.6_dev`**。将本分支的薄入口 `.github/workflows/pr-test-hcu-v056.yml` 单独审查、同步至受保护的**默认分支 main**；它只匹配 `v0.5.6_dev`。仅推送提案分支不会启用此入口。参考 [GitHub 的 pull_request_target 事件说明](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request_target)。
2. 固定镜像 `42.228.13.241:5000/hcu/base_env/vllm:0.28.1-ubuntu22.04-dtk2610-py3.12-sf_b075-20261114-1013@sha256:dddc90925d5562c930ee875b526ff51b39ac1c4a41e7374b85d70ec1d7b7bc7c` 自带匹配的 vLLM、HIP PyTorch 和已编译的 LMCache。由 kubelet/CRI 拉取，不在 job 内启动 Docker；若需要认证，使用平台预配的 imagePullSecrets。模型目录使用只读挂载。
3. 只有在组织确认这几个池对公开 PR 隔离后才启用。所有容器均不得暴露用户家目录（即使只读）、生产凭据、宿主套接字或他人可写的任务缓存。挂载由平台配置；若 Pod 能看到共享 home，流水线会拒绝执行 PR 代码，请平台调整隔离，不要关闭检查。构建产物在 jobs 间用 Actions artifact 传递。
4. 保护默认分支、版本分支和 CI 代码；runner group 限制到受信的 `HYGON-AI/LMCache-das/.github/workflows/hcu-v056.yml@refs/heads/v0.5.6_dev`，不能只靠 label 控制谁能使用集群。若仓库其他 workflows 仍自动监听 PR，它们不受本入口控制。确认没有需要关闭的旁路后，用新的目标分支 PR 验证“不打标不运行、贴 runci 仅测本次 SHA、新 push 不继承批准”。不要把标签审批或默认分支的 check 当作 PR head 已成功通过硬件测试。

## 上游包（只发布一次，不是容器镜像）

固定 [LMCache v0.5.6 的 commit 源码包](https://codeload.github.com/LMCache/LMCache/tar.gz/3e11e10812b448142c6cdb177a3058652bd826fa)，SHA256 为 `6cbc61020b30609edec743a6c81c0a7ed91f1b5c081bbc513d75d0a01ee5b75e`。在维护节点执行 `bash ci/hcu_v056/stage-upstream.sh /path/to/lmcache-0.5.6-3e11e108.tar.gz`。脚本校验后原子发布，不覆盖旧文件；同内容可重复执行。目标为 `/ci_public/lmcache-das/assets/upstream/lmcache-0.5.6-3e11e108.tar.gz`。

单卡业务测试需要只读挂载上述源码包，校验后只导入其测试夹具；不重新编上游、不安装 grpcio-tools。运行时和模型均使用锁定镜像里已编译好的 `lmcache 0.5.6+dtk2610.torch2110.2610221358.g3e11e1`。先核对版本和原生扩展，再把已安装的模块复制到作业私有 venv，避免补丁 CLI 写到镜像基础环境。HCU 适配 wheel 每轮编一次。运行用户须能在容器内解析到用户名；不需要 root，也不应挂宿主机的 /etc/passwd。

## 业务回归与 weekly

业务测试来自 main 的 `39ee5c23`，以前的 159 个参数化 node ID 列在 `business-nodeids.json`。要求全部收集，只排除历史批准的 4 个精确 ID，其余全部实际运行；允许新增测试，不允许少收集、跳过或用 smoke 代替。单独产出 `business-inventory.json`、`business-selection.json` 和 `business.xml`；收集错误同样使 CI 失败。

Weekly 单独使用 `weekly-hcu-v056.yml`，不监听 PR。仅在受保护的 `v0.5.6_dev` 上手动运行；定时或手动入口需在默认分支注册，定时始终检出版本分支并将解析到的 SHA 固定给全部 job。只允许受信 workflow 使用相应 ARC 池。六组测试沿用 main 的工作量：Qwen3-8B（2 卡）、DeepSeek-R1-W4A8（8 卡），各自测 CPU、LocalDisk、POSIX；每组测长文与 CMMLU，磁盘两组另测 OpenCompass（HumanEval、GSM8K）。不使用双卡 MP smoke 代替精度测试。

构建池需要只读的固定工具包；单卡池需要上游 fixtures 包；weekly 模型池还需要按 `ci/hcu/model-test-manifest.json` 的容器路径只读挂载相应模型、评测数据和审核过的 OpenCompass 环境，提供独占的 `/sandbox`、大盘 `/local_disk` 及评测依赖。预检失败会列出每个场景缺少什么，不启动模型。只有单测与预检均通过才调度模型池。
