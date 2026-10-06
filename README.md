# 近海坐底式海流计 · 潮流调和分析服务

把“每来一批就从头跑 MATLAB 老脚本”的流程替换成一个可重复、可审计的后端：
FastAPI + SQLite，数据与结果落在挂载卷；调和分析的分潮判别、最小二乘与椭圆换算
均为手写实现（见 [`docs/ALGORITHM.md`](docs/ALGORITHM.md)），不使用任何潮汐库。

* Python 固定 **3.11**，容器基础镜像 `python:3.11-slim`
* 对外只有 HTTP（8000 端口），容器内不含测试代码与其他入口
* 每批数据产生一个不可变结果版本，任何一版可回查
* 同一批重复上传结果不变；乱序到达与按时间顺序送达的最终结果一致；重启不丢档

## 运行

```bash
docker compose up --build          # 数据在命名卷 tide-data -> /data
# 或直接容器运行
docker build -t tide-harmonic .
docker run -p 8000:8000 -v "$PWD/data:/data" tide-harmonic
```

健康检查：`GET /health`。Swagger UI：`http://localhost:8000/docs`。
数据库路径由环境变量 `TIDE_DB_PATH` 覆盖（默认 `/data/tide.db`）。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/stations` | 建测站 |
| GET  | `/stations/{id}` | 查测站配置 |
| POST | `/stations/{id}/batches` | 上传一个批次（触发重新拟合、存档） |
| GET  | `/stations/{id}/latest` | 最新结果 |
| GET  | `/stations/{id}/versions` | 版本清单 |
| GET  | `/stations/{id}/versions/{n}` | 回查第 n 版完整结果 |

### 建测站

```json
POST /stations
{
  "station_id": "ST01",
  "longitude": 122.0,
  "latitude": 30.0,
  "sample_interval_seconds": 3600.0,
  "deployed_at": "2020-01-01T00:00:00Z"
}
```

### 上传批次

```json
POST /stations/ST01/batches
{
  "batch_id": "2020-01-05-A",
  "samples": [
    {"time": "2020-01-01T00:00:00Z", "u": 12.3, "v": -4.1},
    {"time": "2020-01-01T01:00:00Z", "u": null, "v": null}
  ]
}
```

* `time`：ISO-8601（建议 UTC，`Z` 结尾；裸时间按 UTC 处理），必须不早于布放、
  且落在测站采样网格上；
* `u/v`：东、北分量 cm/s，缺测用 `null`（也可只缺一个分量）；`NaN/Infinity`、
  非数值、类型错误一律 422 并给出 `samples[i].字段`；
* 批内同一时刻重复、与历史批次时刻重叠：422 并指出具体下标；
* `(station_id, batch_id)` 相同但内容不同：409；内容完全相同：返回当前最新版、
  `replayed=true`，不产生新版本；
* 测站不存在：422，字段 `station_id`。

### 结果结构

```json
{
  "station_id": "ST01",
  "version": 2,
  "triggered_by_batch": "2020-01-05-A",
  "replayed": false,
  "batches": ["2020-01-05-A"],
  "record": {
    "n_samples": 96, "n_valid": 95, "n_missing": 1,
    "start_time": "2020-01-01T00:00:00.000Z",
    "end_time": "2020-01-04T23:00:00.000Z",
    "span_hours": 95.0
  },
  "mean": {"u": 3.1, "v": -0.8, "speed": 3.2,
           "direction_rad": -0.253, "direction_deg": -14.5},
  "constituents": [
    {"name": "M2", "frequency_rad_per_hour": 0.50586805,
     "u": {"amplitude": 98.2, "phase": 1.72},
     "v": {"amplitude": 77.4, "phase": 2.05},
     "ellipse": {"semi_major": 113.1, "semi_minor": 12.4,
                 "inclination": 0.61, "inclination_deg": 34.9,
                 "phase": 0.21}}
  ],
  "excluded": [
    {"name": "S2", "masked_by": "M2",
     "required_hours": 354.367, "missing_hours": 259.367}
  ]
}
```

* `semi_minor` 带符号：**正 = 逆时针**，负 = 顺时针；
* `inclination/phase` 为弧度，范围分别为 `[0,π)`、`[0,2π)`；
* 未纳取的分潮在 `excluded` 中注明被谁遮住（`masked_by`）、所需总时长与
  `missing_hours`（还差多长记录）。

## 分潮纳取规则摘要

经典 Rayleigh 判据 R=1：候选分潮与每个已纳取分潮的有效记录跨度都达到
`1/|Δf|` 小时（f 取周/小时，整拍周期）才纳入。候选按
`M₂, S₂, K₁, O₁, N₂, K₂` 固定优先级评估。由此 S₂ 与 K₂ 在约 29 天记录前不可能
同时出现。详见算法文档。

## 测试

```bash
python -m pip install -r requirements.txt pytest httpx
python -m pytest -q
```

覆盖：单一 M₂ 无噪声恢复（振幅/相位/椭圆 1e-6 内）；逆/顺时针圆流短半轴符号；
坐标系旋转后长短轴不变、方向角正好随转角；加常数只改平均流；短记录 S₂/K₂ 不
同时进拟合并给出遮罩信息；乱序到达等价；增量=一次性（相对振幅 1e-9、相位
1e-7 rad）；重复上传幂等、同号异容 409；版本存档与重启存活；以及全部字段级
报错（离格/不等间隔、重复时刻、非有限值、早于布放、测站不存在）。

## 目录

```
app/
  harmonics.py   频率表、Rayleigh 判别、正规方程最小二乘、椭圆换算（手写）
  service.py     字段校验、幂等、全量重拟合与版本编排
  db.py          SQLite schema 与存取
  main.py        FastAPI 路由（唯一对外接口）
docs/ALGORITHM.md
tests/
Dockerfile  docker-compose.yml  requirements.txt
```
