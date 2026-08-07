// SPDX-License-Identifier: Apache-2.0
# LMCache-HCU vLLM Example

## Recommended Image

Use the following container image for this example:

## Install LMCache-HCU

Before running this example, install baseline LMCache, install LMCache-HCU, apply the source patches, and verify runtime patch activation by following [`../README.md`](../README.md).

## Start vLLM

Set the runtime environment variables first:

```bash
export VLLM_LOGGING_LEVEL=INFO
export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export HIP_VISIBLE_DEVICES=1,2
export LMCACHE_CONFIG_FILE=lmcache_config.yaml
```

Then start vLLM:

```bash
nohup vllm serve /model/Qwen3-8B \
    --disable-log-requests \
    --max-log-len 64 \
    --tensor-parallel-size 2 \
    --port 10009 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_connector":"LMCacheHcuConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_hcu.integration.vllm.lmcache_hcu_connector_v1"}' \
    --trust-remote-code \
    --pipeline-parallel-size 1 \
    --served-model-name "Qwen3-8B" \
    > vllm_running.log 2>&1 &
```

Notes:

- `LMCACHE_CONFIG_FILE=lmcache_config.yaml` tells LMCache which configuration file to load.
- `HIP_VISIBLE_DEVICES=1,2` selects the two HCU devices used by this example.
- `--tensor-parallel-size 2` should match the number of visible devices.
- `--kv-transfer-config` enables the LMCache connector for both prefill and decode roles.
- Adjust the model path, port, device list, and memory settings for your environment.

## Example `lmcache_config.yaml`

Create `lmcache_config.yaml` in the same directory where vLLM is started:

```yaml
# Number of tokens stored in each KV cache chunk.
chunk_size: 256

# Enable the XDS backend and set the XDS mount path.
xds_path: /mnt/volume1

# XDS capacity, in GiB, for one device.
max_xds_size: 1024

# GPU memory used by the XDS buffer, in MiB.
# If this value is omitted, LMCache uses the POSIX protocol.
xds_buffer_size: 6144

# Disable local CPU memory storage.
local_cpu: false

# Timeout, in seconds, for blocking remote backend operations.
blocking_timeout_secs: 10

# Extra XDS backend options.
extra_config:
  xds_io_threads: 8
```

## Check the logs

After vLLM starts, inspect the log file:

```bash
tail -f vllm_running.log
```

The log should show that LMCache loads `lmcache_config.yaml` and initializes the configured storage backend.
