# SGLang Engine Monitor（社区风格负载监控面板）

单文件监控面板，为本地部署的 **SGLang** 推理引擎设计：一个 `sglang_monitor.py` 起一个 FastAPI
页面（默认 `:7100`），后台线程每秒抓取引擎 `/metrics` 与 `/get_load`，在浏览器里展示
吞吐、队列、能耗、请求时延与硬件状态，并支持可选的引擎控制 / 崩溃自动拉起。

代码里没有任何默认密钥：引擎 `--api-key` 通过 **环境变量** 或 **启动器脚本** 在运行时读取，
不会写进代码或仓库。

## 功能一览

- **服务指标**：实时/聚合输出 tok/s、TTFT/端到端时延分位（P50/P90/P99）、
  起草接受率（EAGLE/MTP）、并发与排队、KV/Token 用量、L2 host 缓存、错误率、调度空闲率
- **硬件监视**：CPU/内存/swap、磁盘读写（累计+速率）、网络上下行、GPU 卡片（利用率/显存/温度/功耗/频率）
- **能耗卡**：GPU+CPU（RAPL，需要 `sudo -n` 可读）功率积分、逐日 kWh、电费估算（电价可改）
- **价值卡**：按可编辑单价估算今日/累计 token 成本（输入/缓存/输出分列）
- **队列图表**：当日 24h 队列曲线（60s 快照）+ 单行实时队列表
- **已完成请求表**：输入/缓存/输出、prefill 速度、输出速度、排队/前向耗时
- **引擎控制（可选）**：启动 / 停止 / 重启按钮、启动脚本在线编辑（`bash -n` 校验 + 自动备份）、
  崩溃自动拉起（防抖 + 每小时熔断）
- **日志管理**：引擎日志跟随（含稀疏日志/写洞检测）、容量上限、过期日志清理

## 快速开始

```bash
pip install -r requirements.txt          # fastapi + uvicorn
python3 sglang_monitor.py                # 面板 :7100，引擎 :7007
# 浏览器打开 http://<host>:7100/
```

最常用参数：

```bash
python3 sglang_monitor.py \
  --port 7100 \                      # 面板端口
  --engine-port 7007 \               # 被监控的 SGLang 端口
  --engine-host 127.0.0.1 \
  --api-key "$SGLANG_API_KEY" \      # 引擎 --api-key（/get_load 需要；/metrics 免鉴权）
  --launcher /path/to/serve.sh \     # 可选：开启 启动/停止/重启/自动拉起
  --engine-log /tmp/sglang.log \     # 引擎日志（默认 /tmp/sglang.log）
  --data-dir /var/lib/sglang-monitor # 数据目录（默认 ./data）
```

全部参数都可用环境变量替代（`SGLANG_MONITOR_PORT` / `SGLANG_ENGINE_HOST` /
`SGLANG_ENGINE_PORT` / `SGLANG_API_KEY` / `SGLANG_LAUNCHER` / `SGLANG_ENGINE_LOG` /
`SGLANG_MONITOR_DATA_DIR` / `SGLANG_MONITOR_AUTO_RESTART` /
`SGLANG_MONITOR_AUTO_DOWN_TICKS` / `SGLANG_MONITOR_AUTO_MAX_PER_HOUR` /
`SGLANG_MONITOR_LOG_CAP_MB` / `SGLANG_MONITOR_ELECTRICITY_PRICE` /
`SGLANG_MONITOR_COMPENSATE_W`）。

也可以用脚本管理：

```bash
./monitor_ctl.sh start      # start|stop|status [port]
```

## 引擎控制与自动拉起（默认关闭）

- 未配置 `--launcher` 时，控制按钮返回"未配置启动脚本"，面板仍可只读监控。
- `SGLANG_MONITOR_AUTO_RESTART=1` 开启崩溃自动拉起：引擎连续 3 次巡检不可达才动作，
  10 分钟最小间隔、每小时最多 3 次熔断；手动"停止"会自动关闭自动拉起。
- 它只补"没人拉起"的空档，不修复底层 OOM —— 请与真正的修复配对使用。

## 能耗与计费

- 功率 = GPU（nvidia-smi）+ CPU（Intel/AMD RAPL `energy_uj`，需 `sudo -n`）+
  未被传感器覆盖的固定补偿（默认 45W，可改）。
- RAPL 不可读时 CPU 功耗显示"不可读"，面板其余功能不受影响。
- 电价默认 1.0 元/kWh（示例），可在界面修改，写入 `data/energy.json`。
- token 单价默认为示例价，可在界面修改，写入 `data/pricing.json`。

## systemd 示例

```ini
[Unit]
Description=SGLang engine monitor
After=network.target

[Service]
User=YOUR_USER
Environment=SGLANG_API_KEY=
Environment=SGLANG_LAUNCHER=
WorkingDirectory=/opt/sglang-monitor
ExecStart=/usr/bin/python3 sglang_monitor.py --port 7100 --engine-port 7007
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## 安全设计

- 出网仅限两个精确 allowlist 的本地端点（`/metrics`、`/get_load`），
  每次请求都校验 scheme/host/解析后 IP，防 DNS rebinding；重定向关闭。
- 引擎 API key 只从环境变量或你本机的启动脚本读取，仓库代码不含任何密钥。
- 启动脚本编辑接口要求内容以 `#!` 开头且包含 "sglang"，写入前 `bash -n` 语法校验，
  保存前自动备份 `.bak-monitor`。

## FAQ

- **面板能打开但全是 "-"**：检查 `--engine-port` 与引擎端口一致；引擎挂了也会显示"不可达/已停止"。
- **队列/时延卡一直 "-"**：`/get_load` 需要 API key，配置 `--api-key` 或让监控读启动器。
- **能耗卡 CPU 不可读**：RAPL 需要 root；确认 `sudo -n cat /sys/class/powercap/.../energy_uj` 可用。
- **想改电价/单价**：界面直接改，或编辑 `data/energy.json` / `data/pricing.json`。

## English quickstart

```bash
pip install -r requirements.txt
python3 sglang_monitor.py --port 7100 --engine-port 7007 --api-key <key> --launcher <serve.sh>
# open http://<host>:7100/
```

A single-file FastAPI dashboard for a local SGLang engine: live tok/s, queue charts,
TTFT/E2E percentiles, energy (RAPL) & cost cards, hardware widgets, optional
engine start/stop and crash auto-restart (opt-in via `--launcher`).
All configuration is via CLI flags or environment variables; no secrets are
stored in the repo.

License: MIT.
