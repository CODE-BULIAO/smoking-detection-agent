# 建筑工地吸烟检测智能体

本项目完整代码：基于四级漏斗（人物检测 → 香烟检测+时序确认 → 姿态门控 → 多模态大模型终审）的工地吸烟实时检测系统，含实时 Web 服务（`server.py`）与离线批处理（`pipeline.py`）。

- Stage1/Stage3 使用 Ultralytics 官方 YOLO 权重（首次运行自动下载）；Stage2 香烟检测权重与 OpenVINO IR 导出件体积较大，未随仓库提供，可联系作者获取。
- 终审大模型：任意 OpenAI 兼容视觉模型（默认 Qwen-VL）。复制 `llm_config.json.example` 为 `llm_config.json` 并填写 `base_url` / `model` / `api_key`。

## 运行

```bash
pip install -r requirements.txt
python server.py --port 8000   # 浏览器打开 http://localhost:8000
```

参数与边缘部署方式见 `server.py` / `pipeline.py` 内注释。
