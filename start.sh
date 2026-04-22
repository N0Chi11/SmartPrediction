#!/bin/bash
echo "=========================================="
echo "  慧估后端 v3 — 多模型 · K折CV · 舆情 · 对话"
echo "=========================================="
echo ""
echo "检查依赖..."
python -c "import lightgbm, flask, flask_cors, pandas, numpy, openpyxl, sklearn, requests, bs4" 2>/dev/null || {
    echo "安装依赖..."
    pip install lightgbm flask flask-cors openpyxl pandas numpy scikit-learn requests beautifulsoup4
}
echo "依赖就绪"
echo ""
echo "候选模型: LightGBM · GBM · RandomForest · Ridge · ElasticNet"
echo "工作流:"
echo "  Stage A  行业选优 (K折CV)"
echo "  Stage B  公司预测"
echo "  Stage C  舆情分析 (股吧爬取 + LLM 情感分析)"
echo "  对话框   AI 财务助手 (携带预测/舆情上下文)"
echo ""
echo "启动后端服务..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo "=========================================="
python app.py
