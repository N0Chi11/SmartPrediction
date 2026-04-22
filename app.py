"""
慧估后端 v2 — 多模型选优 + K折交叉验证 + AI报告生成
=======================================
工作流：
  Stage 1  /api/evaluate  上传行业数据 → 对每个指标跑所有模型
                         → K折时间序列交叉验证 → 输出每指标的最优模型
  Stage 2  /api/predict   上传公司数据 + 最优模型映射 → 用最优模型预测
  Stage 3  /api/generate_report  根据预测结果生成AI报告
"""

import os, io, json, traceback, time, re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 调试信息
print("开始导入模块...")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

print("模块导入完成")

app = Flask(__name__, static_folder=".")
CORS(app)

# ═══════════════════════════════════════════════════════════
# 指标配置
# ═══════════════════════════════════════════════════════════
IND_KEYS = [
    "营业收入（同比增长率）",
    "销售毛利率",
    "净资产收益率ROE",
    "销售净利率",
    "总资产净利率ROA",
    "研发投入总额占营业收入比例",
    "总资产周转率",
    "权益乘数（杜邦分析）",
]

IND_SHORT = {
    "营业收入（同比增长率）": "营收增长率",
    "销售毛利率": "毛利率",
    "净资产收益率ROE": "ROE",
    "销售净利率": "净利率",
    "总资产净利率ROA": "ROA",
    "研发投入总额占营业收入比例": "研发投入占比",
    "总资产周转率": "资产周转率",
    "权益乘数（杜邦分析）": "权益乘数",
}

QMAP = {"一季": "Q1", "中报": "Q2", "三季": "Q3", "年报": "Q4"}

# 大语言模型API配置（DeepSeek）
LLM_API_KEY = "sk-b28f9f72cec34dde853b40345b7274cf"
LLM_API_URL = "https://api.deepseek.com/v1/chat/completions"


def call_llm_api(prompt, model="deepseek-chat", max_tokens=2000, temperature=0.7):
    """
    调用大语言模型API生成报告
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的财务分析师，擅长分析企业的盈利能力和财务状况。请根据提供的财务预测数据，生成一份详细、专业的资产盈利能力预测报告。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM API调用失败: {e}")
        # 失败时回退到规则生成
        return None


# ═══════════════════════════════════════════════════════════
# 1. 模型映射表 (Model Registry)
# ═══════════════════════════════════════════════════════════
# 每个 factory 接收 (training_size) 返回一个 sklearn 风格的模型实例
# training_size 用来自适应一些模型的超参数(比如树的深度、叶子数)

def _lightgbm_factory(n):
    return lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=max(2, n // 8),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        n_jobs=-1,
    )

def _gbm_factory(n):
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=max(2, n // 10),
        random_state=42,
    )

def _rf_factory(n):
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=max(2, n // 10),
        n_jobs=-1,
        random_state=42,
    )

def _ridge_factory(n):
    return Ridge(alpha=1.0, random_state=42)

def _elasticnet_factory(n):
    return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)


MODEL_REGISTRY = {
    "LightGBM":    {"factory": _lightgbm_factory,   "label": "LightGBM",     "family": "boosting"},
    "GBM":         {"factory": _gbm_factory,        "label": "梯度提升树",    "family": "boosting"},
    "RandomForest":{"factory": _rf_factory,         "label": "随机森林",      "family": "bagging"},
    "Ridge":       {"factory": _ridge_factory,      "label": "岭回归",        "family": "linear"},
    "ElasticNet":  {"factory": _elasticnet_factory, "label": "弹性网络",      "family": "linear"},
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════
# 2. 解析同花顺宽格式 Excel
# ═══════════════════════════════════════════════════════════
def parse_excel(file_bytes: bytes, company: str = "", mode: str = "median") -> pd.DataFrame:
    """
    mode:
      "median"  - 返回行业中位数 (用于 /api/evaluate)
      "company" - 返回指定公司的时序 (用于 /api/predict)
      "all"     - 返回每个公司×季度×指标的长表 (用于 /api/evaluate 的整体训练)
    """
    xl = pd.read_excel(io.BytesIO(file_bytes), header=0)

    if "证券名称" not in xl.columns:
        for col in xl.columns:
            if "名称" in str(col):
                xl.rename(columns={col: "证券名称"}, inplace=True)
                break

    if mode == "company":
        if not company:
            raise ValueError("company 模式必须指定公司名")
        company_data = xl[xl["证券名称"] == company]
        if company_data.empty:
            raise ValueError(f"未找到公司: {company}")
        rows = [company_data]
    else:
        rows = [xl]

    # 识别并重塑长格式
    records = []
    for df in rows:
        data = df.iloc[:, 2:]
        names = df["证券名称"].values if "证券名称" in df.columns else [None] * len(df)
        for col in data.columns:
            col_str = str(col).replace("\n", "").replace(" ", "")
            matched_key = None
            for k in IND_KEYS:
                if col_str.startswith(k.replace(" ", "")):
                    matched_key = k
                    break
            if matched_key is None:
                continue
            year_q = None
            for y in range(2014, 2027):
                for cn, en in QMAP.items():
                    if f"{y}{cn}" in str(col):
                        year_q = (y, en)
                        break
                if year_q:
                    break
            if year_q is None:
                continue
            nums = pd.to_numeric(data[col], errors="coerce")
            for i, v in enumerate(nums):
                if pd.isna(v):
                    continue
                records.append({
                    "company": names[i] if i < len(names) else None,
                    "year": year_q[0],
                    "quarter": year_q[1],
                    "indicator": matched_key,
                    "value": float(v),
                })

    if not records:
        raise ValueError("未识别到有效指标列，请检查文件格式")

    long_df = pd.DataFrame(records)
    long_df["t"] = long_df["year"].astype(str) + long_df["quarter"]

    def t_key(s):
        return int(s[:4]) * 4 + int(s[5])

    if mode == "company":
        pivot = long_df.pivot_table(index="t", columns="indicator", values="value", aggfunc="mean")
    elif mode == "median":
        pivot = long_df.pivot_table(index="t", columns="indicator", values="value", aggfunc="median")
    elif mode == "all":
        return long_df
    else:
        raise ValueError(f"未知 mode: {mode}")

    pivot = pivot.iloc[sorted(range(len(pivot)), key=lambda i: t_key(pivot.index[i]))]
    keep = [c for c in IND_KEYS if c in pivot.columns and pivot[c].notna().mean() >= 0.7]
    pivot = pivot[keep]
    pivot = pivot.interpolate(method="linear", limit=2, limit_direction="both")
    return pivot


# ═══════════════════════════════════════════════════════════
# 3. 滑窗特征构造 (与原版保持一致)
# ═══════════════════════════════════════════════════════════
def make_features(series: np.ndarray, seq_len: int = 8):
    """给单个指标时序构建监督学习样本。"""
    X, y = [], []
    n = len(series)
    for i in range(seq_len, n):
        window = series[i - seq_len:i]
        diff1 = np.diff(window)
        diff2 = np.diff(diff1)
        seasonal = series[i - 4] if i >= 4 else window[0]
        trend = np.polyfit(np.arange(seq_len), window, 1)[0]
        feats = np.concatenate([window, diff1, diff2, [seasonal, trend]])
        # 处理NaN值，用0替换
        feats = np.nan_to_num(feats, nan=0.0)
        X.append(feats)
        y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_one_feature(history: list, seq_len: int = 8):
    """给一段 history 构造单步特征 (用于滚动预测)。"""
    window = np.array(history[-seq_len:], dtype=np.float32)
    diff1 = np.diff(window)
    diff2 = np.diff(diff1)
    seasonal_idx = len(history) - 4
    seasonal = history[seasonal_idx] if seasonal_idx >= 0 else window[0]
    trend = float(np.polyfit(np.arange(seq_len), window, 1)[0])
    feats = np.concatenate([window, diff1, diff2, [seasonal, trend]])
    # 处理NaN值，用0替换
    feats = np.nan_to_num(feats, nan=0.0)
    return feats.reshape(1, -1)


# ═══════════════════════════════════════════════════════════
# 4. K折时间序列交叉验证
# ═══════════════════════════════════════════════════════════
def cv_evaluate(series: np.ndarray, model_name: str, seq_len: int = 8, n_splits: int = 5) -> dict:
    """
    对单个指标时序用指定模型做 TimeSeriesSplit K 折交叉验证。
    返回 {mean_mse, std_mse, fold_mses}。
    """
    if len(series) < seq_len + n_splits + 2:
        return {"mean_mse": None, "std_mse": None, "fold_mses": [], "error": "数据不足"}

    scaler = StandardScaler()
    s_scaled = scaler.fit_transform(series.reshape(-1, 1)).ravel()
    X, y = make_features(s_scaled, seq_len)

    if len(X) < n_splits + 1:
        return {"mean_mse": None, "std_mse": None, "fold_mses": [], "error": "样本不足以做 K 折"}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    factory = MODEL_REGISTRY[model_name]["factory"]

    fold_mses = []
    for tr_idx, va_idx in tscv.split(X):
        if len(tr_idx) < 3 or len(va_idx) < 1:
            continue
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = factory(len(X_tr))
        try:
            model.fit(X_tr, y_tr)
            pred_scaled = model.predict(X_va)
            # 反归一化到原始量级再算 MSE，结果才有可比性
            pred_orig = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            true_orig = scaler.inverse_transform(y_va.reshape(-1, 1)).ravel()
            fold_mses.append(float(mean_squared_error(true_orig, pred_orig)))
        except Exception as e:
            print(f"    [warn] {model_name} fold fit failed: {e}")
            continue

    if not fold_mses:
        return {"mean_mse": None, "std_mse": None, "fold_mses": [], "error": "所有 fold 都失败"}

    return {
        "mean_mse": round(float(np.mean(fold_mses)), 6),
        "std_mse": round(float(np.std(fold_mses)), 6),
        "fold_mses": [round(m, 6) for m in fold_mses],
    }


# ═══════════════════════════════════════════════════════════
# 5. 单指标训练 + 滚动预测 (指定模型)
# ═══════════════════════════════════════════════════════════
def train_predict_one(series: np.ndarray, n_steps: int, model_name: str,
                      seq_len: int = 8) -> dict:
    """用指定模型预测 n_steps 步。"""
    # 必须提供模型名称
    if not model_name:
        raise ValueError("必须提供模型名称")
    
    # 检查模型是否注册
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"模型 {model_name} 未注册")
    
    scaler = StandardScaler()
    s_scaled = scaler.fit_transform(series.reshape(-1, 1)).ravel()
    X, y = make_features(s_scaled, seq_len)

    if len(X) < 4:
        # 数据太短 → 线性外推兜底
        slope = np.polyfit(np.arange(len(series)), series, 1)[0]
        last = series[-1]
        preds = [float(last + slope * (i + 1)) for i in range(n_steps)]
        return {"preds": preds, "train_mse": None, "method": "linear_fallback", "model": "Linear"}

    # 训练集 / 验证集 (末尾 15% 做验证，最少 2 期)
    n_val = max(2, int(len(X) * 0.15))
    X_tr, X_val = X[:-n_val], X[-n_val:]
    y_tr, y_val = y[:-n_val], y[-n_val:]

    factory = MODEL_REGISTRY[model_name]["factory"]
    model = factory(len(X_tr))

    # LightGBM 支持 early stopping，其他模型直接 fit
    if model_name == "LightGBM":
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
        except Exception:
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)

    val_pred = model.predict(X_val)
    # 反归一化后再算 MSE，与 CV 保持同一量纲
    val_pred_orig = scaler.inverse_transform(val_pred.reshape(-1, 1)).ravel()
    y_val_orig = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    train_mse = float(mean_squared_error(y_val_orig, val_pred_orig))

    # 滚动预测
    history = s_scaled.tolist()
    preds_scaled = []
    for _ in range(n_steps):
        feat = make_one_feature(history, seq_len)
        p = float(model.predict(feat)[0])
        preds_scaled.append(p)
        history.append(p)
    preds_orig = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).ravel().tolist()

    # 特征重要性 (仅对树模型有意义)
    fi_norm = None
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_.tolist()
        fi_sum = sum(fi) or 1
        fi_norm = [round(v / fi_sum, 4) for v in fi]

    return {
        "preds": [round(v, 4) for v in preds_orig],
        "train_mse": round(train_mse, 6),
        "method": model_name.lower(),
        "model": model_name,
        "model_label": MODEL_REGISTRY.get(model_name, {}).get("label", model_name),
        "feature_importance": fi_norm,
    }


# ═══════════════════════════════════════════════════════════
# 6. 预测季度标签生成
# ═══════════════════════════════════════════════════════════
def next_quarter_labels(last_label: str, n: int) -> list:
    y = int(last_label[:4])
    q = int(last_label[5])
    labels = []
    for _ in range(n):
        q += 1
        if q > 4:
            q = 1; y += 1
        labels.append(f"{y}Q{q}")
    return labels


# ═══════════════════════════════════════════════════════════
# 7. AI报告生成函数
# ═══════════════════════════════════════════════════════════
def generate_ai_report(result_data, eval_data=None):
    """
    根据预测结果生成AI报告
    result_data: /api/predict 的返回数据
    eval_data: /api/evaluate 的返回数据（可选）
    """
    company = result_data.get("company", "目标公司")
    indicators = result_data.get("indicators", {})
    years = result_data.get("years", 3)
    
    # 分析各项指标的趋势
    analysis_results = {}
    for key, data in indicators.items():
        short_name = data.get("short", key)
        hist_values = data.get("hist", [])
        fore_values = data.get("fore", [])
        
        if not hist_values or not fore_values:
            continue
            
        # 获取历史和预测的末期值
        hist_last = hist_values[-1] if hist_values else None
        fore_last = fore_values[-1] if fore_values else None
        
        # 计算变化趋势
        change_pct = None
        if hist_last is not None and fore_last is not None and hist_last != 0:
            change_pct = (fore_last - hist_last) / abs(hist_last) * 100
        
        analysis_results[key] = {
            "short_name": short_name,
            "hist_last": hist_last,
            "fore_last": fore_last,
            "change_pct": change_pct,
            "mse": data.get("train_mse"),
            "model_used": data.get("model_label", data.get("model", "Unknown"))
        }
    
    # 构建大语言模型提示词
    prompt = f"# 资产盈利能力预测报告生成任务\n\n"
    prompt += f"## 公司信息\n"
    prompt += f"- 公司名称: {company}\n"
    prompt += f"- 预测周期: {years}年\n\n"
    
    prompt += "## 预测数据\n"
    for key, data in analysis_results.items():
        short_name = data["short_name"]
        hist_last = data["hist_last"]
        fore_last = data["fore_last"]
        change_pct = data["change_pct"]
        model_used = data["model_used"]
        
        prompt += f"- {short_name}: 当前值 {hist_last:.2f}%，预测值 {fore_last:.2f}%，变化 {change_pct:.2f}%，使用模型 {model_used}\n"
    
    if eval_data:
        prompt += "\n## 模型选优信息\n"
        prompt += "本次预测采用多模型K折交叉验证选优策略，为不同指标选用最适合的预测模型：\n"
        for indicator, model_name in eval_data.get("best_models", {}).items():
            if indicator in analysis_results:
                short_name = analysis_results[indicator]["short_name"]
                mse = eval_data['indicators'][indicator]['best_mse']
                prompt += f"- {short_name}：采用{model_name}模型，CV验证MSE为{mse:.6f}\n"
    
    prompt += "\n## 报告要求\n"
    prompt += "请生成一份详细、专业的资产盈利能力预测报告，包括以下内容：\n"
    prompt += "1. 总体概况：对公司未来盈利能力的整体评估\n"
    prompt += "2. 详细指标分析：对每个指标的变化趋势进行分析\n"
    prompt += "3. 模型精度评估：基于MSE值评估预测的可靠性\n"
    prompt += "4. 风险提示：识别可能存在的风险因素\n"
    prompt += "5. 总结与建议：给出针对性的建议\n"
    prompt += "6. 模型选优说明：如果有选优数据，说明模型选择的依据\n"
    prompt += "\n请确保报告内容专业、客观、全面，避免使用过于技术性的术语，保持语言通俗易懂。"
    
    # 调用大语言模型API
    llm_report = call_llm_api(prompt)
    
    # 如果LLM调用失败，回退到规则生成
    if not llm_report:
        # 生成报告内容
        report_parts = []
        
        # 1. 总体概况
        overall_assessment = "总体来看，" + company + "在未来几年的盈利能力表现：\n\n"
        
        # 分析关键指标
        key_metrics = ["净资产收益率ROE", "销售毛利率", "总资产净利率ROA", "销售净利率"]
        positive_count = 0
        total_count = 0
        
        for metric in key_metrics:
            if metric in analysis_results:
                result = analysis_results[metric]
                total_count += 1
                if result["change_pct"] is not None and result["change_pct"] > 0:
                    positive_count += 1
        
        if total_count > 0:
            improvement_rate = positive_count / total_count * 100
            if improvement_rate >= 75:
                overall_assessment += "表现优异，大部分关键指标呈现上升趋势。"
            elif improvement_rate >= 50:
                overall_assessment += "表现稳定，多数关键指标保持平稳或略有提升。"
            elif improvement_rate >= 25:
                overall_assessment += "面临一定挑战，部分关键指标有所下滑。"
            else:
                overall_assessment += "存在较大压力，多数关键指标呈下降趋势。"
        
        report_parts.append(overall_assessment)
        
        # 2. 详细指标分析
        report_parts.append("\n## 详细指标分析\n")
        
        # ROE分析
        if "净资产收益率ROE" in analysis_results:
            roe_data = analysis_results["净资产收益率ROE"]
            roe_text = f"- **净资产收益率(ROE)**: 当前为{roe_data['hist_last']:.2f}%，预测未来将变为{roe_data['fore_last']:.2f}%"
            if roe_data["change_pct"] is not None:
                if roe_data["change_pct"] > 0:
                    roe_text += f"，提升{roe_data['change_pct']:.2f}%，显示股东回报能力增强。"
                else:
                    roe_text += f"，下降{abs(roe_data['change_pct']):.2f}%，需关注股东回报能力变化。"
            else:
                roe_text += "，变化趋势待观察。"
            report_parts.append(roe_text)
        
        # 毛利率分析
        if "销售毛利率" in analysis_results:
            gross_margin_data = analysis_results["销售毛利率"]
            margin_text = f"- **销售毛利率**: 当前为{gross_margin_data['hist_last']:.2f}%，预测未来将变为{gross_margin_data['fore_last']:.2f}%"
            if gross_margin_data["change_pct"] is not None:
                if gross_margin_data["change_pct"] > 0:
                    margin_text += f"，提升{gross_margin_data['change_pct']:.2f}%，显示成本控制能力增强。"
                else:
                    margin_text += f"，下降{abs(gross_margin_data['change_pct']):.2f}%，需关注成本控制。"
            else:
                margin_text += "，变化趋势待观察。"
            report_parts.append(margin_text)
        
        # 营收增长率分析
        if "营业收入（同比增长率）" in analysis_results:
            revenue_data = analysis_results["营业收入（同比增长率）"]
            revenue_text = f"- **营业收入增长率**: 当前为{revenue_data['hist_last']:.2f}%，预测未来将变为{revenue_data['fore_last']:.2f}%"
            if revenue_data["change_pct"] is not None:
                if revenue_data["change_pct"] > 0:
                    revenue_text += f"，提升{revenue_data['change_pct']:.2f}%，显示业务增长势头良好。"
                else:
                    revenue_text += f"，下降{abs(revenue_data['change_pct']):.2f}%，需关注市场拓展情况。"
            else:
                revenue_text += "，变化趋势待观察。"
            report_parts.append(revenue_text)
        
        # 3. 模型精度评估
        report_parts.append("\n## 模型精度评估\n")
        avg_mse = 0
        valid_mse_count = 0
        for key, data in analysis_results.items():
            if data["mse"] is not None:
                avg_mse += data["mse"]
                valid_mse_count += 1
        
        if valid_mse_count > 0:
            avg_mse /= valid_mse_count
            if avg_mse < 0.01:
                accuracy_text = "模型预测精度非常高，预测结果可信度高。"
            elif avg_mse < 0.05:
                accuracy_text = "模型预测精度较高，预测结果较为可靠。"
            elif avg_mse < 0.1:
                accuracy_text = "模型预测精度中等，预测结果可供参考。"
            else:
                accuracy_text = "模型预测精度有待提升，建议结合其他信息综合判断。"
            report_parts.append(f"- 平均MSE: {avg_mse:.4f}，{accuracy_text}")
        
        # 4. 风险提示
        report_parts.append("\n## 风险提示\n")
        risks = []
        for key, data in analysis_results.items():
            if data["change_pct"] is not None and data["change_pct"] < -10:
                short_name = data["short_name"]
                risks.append(f"- {short_name}指标预测下降超过10%，需重点关注")
        
        if risks:
            for risk in risks:
                report_parts.append(f"- {risk}")
        else:
            report_parts.append("- 当前预测显示各项指标相对稳定，风险可控")
        
        # 5. 总结与建议
        report_parts.append("\n## 总结与建议\n")
        report_parts.append(f"基于以上分析，{company}在未来{result_data.get('years', 3)}年的盈利能力表现预计为：")
        
        # 根据指标变化趋势给出总结
        strong_points = []
        improvement_areas = []
        
        if "净资产收益率ROE" in analysis_results:
            roe_data = analysis_results["净资产收益率ROE"]
            if roe_data["change_pct"] is not None and roe_data["change_pct"] > 5:
                strong_points.append("股东回报能力显著提升")
            elif roe_data["change_pct"] is not None and roe_data["change_pct"] < -5:
                improvement_areas.append("股东回报能力需加强")
        
        if "销售毛利率" in analysis_results:
            margin_data = analysis_results["销售毛利率"]
            if margin_data["change_pct"] is not None and margin_data["change_pct"] > 5:
                strong_points.append("成本控制能力改善")
            elif margin_data["change_pct"] is not None and margin_data["change_pct"] < -5:
                improvement_areas.append("成本控制需优化")
        
        if strong_points:
            report_parts.append(f"- 优势方面：{'、'.join(strong_points)}")
        
        if improvement_areas:
            report_parts.append(f"- 改善方面：{'、'.join(improvement_areas)}")
        
        report_parts.append(f"\n建议持续关注上述指标变化，并结合市场环境、行业政策等因素进行综合决策。")
        
        # 如果有选优数据，添加模型选优信息
        if eval_data:
            report_parts.append("\n## 模型选优说明\n")
            report_parts.append("本次预测采用多模型K折交叉验证选优策略，为不同指标选用最适合的预测模型：")
            for indicator, model_name in eval_data.get("best_models", {}).items():
                if indicator in analysis_results:
                    short_name = analysis_results[indicator]["short_name"]
                    report_parts.append(f"- {short_name}：采用{model_name}模型，CV验证MSE为{eval_data['indicators'][indicator]['best_mse']:.6f}")
        
        return "\n".join(report_parts)
    
    return llm_report


# ═══════════════════════════════════════════════════════════
# 8. API 端点：/api/evaluate  —— 行业数据 → K折CV → 最优模型
# ═══════════════════════════════════════════════════════════
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    输入：整个行业的 xlsx
    输出：每个指标的各模型 K 折 MSE + 最优模型名
    """
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "缺少文件"}), 400

        f = request.files["file"]
        n_splits = int(request.form.get("n_splits", 5))
        n_splits = min(max(n_splits, 3), 8)

        file_bytes = f.read()
        print(f"[EVAL] 收到文件: {f.filename}, K折={n_splits}")

        t0 = time.time()
        panel = parse_excel(file_bytes, mode="median")
        print(f"[EVAL] 行业中位数时序: {len(panel)} 期, 指标: {list(panel.columns)}")

        indicators_result = {}
        best_models = {}  # indicator -> best model name
        leaderboard = []  # 整体排名统计

        for col in panel.columns:
            series = panel[col].values.astype(float)
            short = IND_SHORT.get(col, col)
            per_model = {}
            best = None
            for mname in MODEL_NAMES:
                r = cv_evaluate(series, mname, seq_len=8, n_splits=n_splits)
                per_model[mname] = r
                if r["mean_mse"] is not None:
                    if best is None or r["mean_mse"] < per_model[best]["mean_mse"]:
                        best = mname

            best_models[col] = best or "LightGBM"
            indicators_result[col] = {
                "short": short,
                "per_model": per_model,
                "best_model": best,
                "best_mse": per_model[best]["mean_mse"] if best else None,
                "best_std": per_model[best]["std_mse"] if best else None,
            }
            print(f"  [{short}] 最优={best}  MSE={indicators_result[col]['best_mse']}")

        # 计算各模型全局胜率
        wins = {m: 0 for m in MODEL_NAMES}
        for col, info in indicators_result.items():
            if info["best_model"]:
                wins[info["best_model"]] += 1

        # 每个模型的平均 MSE
        avg_mse = {}
        for m in MODEL_NAMES:
            vals = [info["per_model"][m]["mean_mse"] for info in indicators_result.values()
                    if info["per_model"][m]["mean_mse"] is not None]
            avg_mse[m] = round(float(np.mean(vals)), 6) if vals else None

        for m in MODEL_NAMES:
            leaderboard.append({
                "model": m,
                "label": MODEL_REGISTRY[m]["label"],
                "family": MODEL_REGISTRY[m]["family"],
                "wins": wins[m],
                "avg_mse": avg_mse[m],
            })
        # 按胜场数降序、平均MSE升序
        leaderboard.sort(key=lambda x: (-x["wins"], x["avg_mse"] if x["avg_mse"] is not None else 1e18))

        # 解析公司列表 (供下游预测用)
        raw = pd.read_excel(io.BytesIO(file_bytes), header=0)
        if "证券名称" not in raw.columns:
            for c in raw.columns:
                if "名称" in str(c):
                    raw.rename(columns={c: "证券名称"}, inplace=True)
                    break
        companies = []
        if "证券名称" in raw.columns:
            companies = [str(x) for x in raw["证券名称"].dropna().unique().tolist()]

        elapsed = round(time.time() - t0, 2)

        return jsonify({
            "ok": True,
            "data": {
                "hist_labels": list(panel.index),
                "indicators": indicators_result,
                "best_models": best_models,
                "leaderboard": leaderboard,
                "model_names": MODEL_NAMES,
                "n_splits": n_splits,
                "n_companies": len(companies),
                "companies": companies,
                "elapsed": elapsed,
                "model_labels": {m: MODEL_REGISTRY[m]["label"] for m in MODEL_NAMES},
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# 9. API 端点：/api/predict  —— 用最优模型预测单公司
# ═══════════════════════════════════════════════════════════
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "缺少文件"}), 400

        f = request.files["file"]
        years = int(request.form.get("years", 3))
        years = min(max(years, 1), 5)
        n_steps = years * 4
        company = request.form.get("company", "")

        # 最优模型映射 (从 /api/evaluate 回传)
        best_models_json = request.form.get("best_models", "{}")
        try:
            best_models = json.loads(best_models_json) if best_models_json else {}
        except Exception:
            best_models = {}

        # 必须提供最优模型映射
        if not best_models:
            return jsonify({
                "ok": False,
                "error": "请先完成 Stage A（行业选优），为每个指标选择最优模型"
            })

        file_bytes = f.read()
        print(f"[PRED] 文件={f.filename} 公司={company} 年数={years}")
        print(f"[PRED] 使用指标→模型映射: {best_models}")

        # 解析
        if company:
            panel = parse_excel(file_bytes, company=company, mode="company")
        else:
            panel = parse_excel(file_bytes, mode="median")
        print(f"[PRED] 解析完成: {len(panel)} 期, 指标: {list(panel.columns)}")

        hist_labels = list(panel.index)
        fore_labels = next_quarter_labels(hist_labels[-1], n_steps)

        results = {}
        for col in panel.columns:
            series = panel[col].values.astype(float)
            mname = best_models.get(col)
            if not mname:
                return jsonify({
                    "ok": False,
                    "error": f"指标 {IND_SHORT.get(col, col)} 缺少最优模型配置"
                })
            if mname not in MODEL_REGISTRY:
                return jsonify({
                    "ok": False,
                    "error": f"指标 {IND_SHORT.get(col, col)} 的模型 {mname} 未注册"
                })
            res = train_predict_one(series, n_steps, model_name=mname)
            short = IND_SHORT.get(col, col)
            results[col] = {
                "short": short,
                "hist": [round(v, 4) if not np.isnan(v) else None for v in series.tolist()],
                "fore": res["preds"],
                "train_mse": res["train_mse"],
                "method": res["method"],
                "model": res["model"],
                "model_label": MODEL_REGISTRY.get(res["model"], {}).get("label", res["model"]),
            }
            print(f"  [{short}] 模型={mname} MSE={res['train_mse']} 末期={res['preds'][-1]:.4f}")

        hist_last = {col: float(panel[col].iloc[-1]) for col in panel.columns}
        fore_last = {col: results[col]["fore"][-1] for col in panel.columns}

        payload = {
            "hist_labels": hist_labels,
            "fore_labels": fore_labels,
            "indicators": {col: results[col] for col in panel.columns},
            "hist_last": {col: round(v, 4) for col, v in hist_last.items()},
            "fore_last": {col: round(v, 4) for col, v in fore_last.items()},
            "years": years,
            "company": company,
            "best_models_used": {c: results[c]["model"] for c in panel.columns},
        }
        return jsonify({"ok": True, "data": payload})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# 10. API 端点：/api/generate_report  —— 生成AI报告
# ═══════════════════════════════════════════════════════════
@app.route("/api/generate_report", methods=["POST"])
def generate_report():
    """
    根据预测结果生成AI报告
    输入：预测结果数据 + (可选)选优结果数据
    输出：AI生成的分析报告
    """
    try:
        # 获取预测结果数据
        predict_data = request.json.get("predict_data", {})
        eval_data = request.json.get("eval_data", {})
        
        if not predict_data:
            return jsonify({"ok": False, "error": "缺少预测结果数据"}), 400
        
        # 生成AI报告
        report_content = generate_ai_report(predict_data, eval_data)
        
        return jsonify({
            "ok": True,
            "data": {
                "report": report_content,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# 11. 股吧 / 东方财富网 爬虫
# ═══════════════════════════════════════════════════════════
_crawl_cache = {}   # key -> (timestamp, data)   5 分钟内存缓存

def _cache_get(key, ttl=300):
    if key in _crawl_cache:
        ts, data = _crawl_cache[key]
        if time.time() - ts < ttl:
            return data
    return None

def _cache_set(key, data):
    _crawl_cache[key] = (time.time(), data)


def crawl_guba(stock_code, max_pages=2):
    """爬取东方财富股吧帖子列表
    返回: [{title, link, reads, replies, author, time}, ...]
    """
    cache_key = f"guba_{stock_code}_{max_pages}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://guba.eastmoney.com/",
        "Connection": "keep-alive",
    }

    posts = []
    for page in range(1, max_pages + 1):
        url = (f"https://guba.eastmoney.com/list,{stock_code}.html"
               if page == 1
               else f"https://guba.eastmoney.com/list,{stock_code}_{page}.html")
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"[GUBA] HTTP {resp.status_code} on page {page}")
                continue
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")

            # 页面有多种可能结构 —— 逐一尝试
            arts = soup.select("div.articleh") or soup.select("tr.listitem")
            if not arts:
                arts = soup.select('[class*="articleh"]')
            if not arts:
                # 再尝试新版表格结构
                arts = soup.select("tbody tr")

            for art in arts:
                try:
                    title_el = (art.select_one(".l3 a")
                                or art.select_one(".note a")
                                or art.select_one(".title a")
                                or art.select_one('a[href*="/news,"]')
                                or art.select_one("a"))
                    if not title_el:
                        continue
                    title = title_el.get_text(strip=True)
                    if not title or len(title) < 2:
                        continue
                    # 过滤掉明显不是帖子标题的 (导航链接等)
                    if any(kw in title for kw in ["登录", "注册", "首页", "下一页", "上一页"]):
                        continue

                    link = title_el.get("href", "") or ""
                    if link and not link.startswith("http"):
                        link = "https://guba.eastmoney.com" + link

                    def _gt(sel):
                        el = art.select_one(sel)
                        return el.get_text(strip=True) if el else ""

                    posts.append({
                        "title": title,
                        "link": link,
                        "reads": _gt(".l1") or "—",
                        "replies": _gt(".l2") or "—",
                        "author": _gt(".l4 a") or _gt(".l4") or "",
                        "time": _gt(".l5") or _gt(".l6") or "",
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"[GUBA] page {page} error: {e}")
            continue

    # 回退方案: 如果 HTML 解析拿不到, 尝试 guba 的 JSON 搜索接口
    if not posts:
        try:
            fallback_url = f"https://gubaapi.eastmoney.com/lookupArticle"
            fallback_params = {
                "code": stock_code,
                "pageSize": 40,
                "pageIndex": 1,
                "sort": "time",
                "type": 0,
                "param": "",
                "_": int(time.time() * 1000),
            }
            resp = requests.get(fallback_url, params=fallback_params,
                                headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("re") or data.get("data") or []
                for it in items:
                    title = it.get("post_title") or it.get("title") or ""
                    if not title:
                        continue
                    post_id = it.get("post_id") or ""
                    link = (f"https://guba.eastmoney.com/news,{stock_code},{post_id}.html"
                            if post_id else "")
                    posts.append({
                        "title": title,
                        "link": link,
                        "reads": str(it.get("post_click_count", "—")),
                        "replies": str(it.get("post_comment_count", "—")),
                        "author": (it.get("user_nickname") or
                                   it.get("post_user_nickname") or ""),
                        "time": it.get("post_publish_time", "") or it.get("post_last_time", ""),
                    })
                print(f"[GUBA] JSON fallback ok, got {len(posts)}")
        except Exception as e:
            print(f"[GUBA] JSON fallback failed: {e}")

    # 去重
    seen = set()
    uniq = []
    for p in posts:
        if p["title"] in seen:
            continue
        seen.add(p["title"])
        uniq.append(p)

    _cache_set(cache_key, uniq)
    return uniq


def crawl_eastmoney_news(stock_code, max_items=20):
    """通过东方财富公告接口抓取个股公告"""
    cache_key = f"news_{stock_code}_{max_items}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    code = stock_code.zfill(6)
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    params = {
        "sr": -1,
        "page_size": max_items,
        "page_index": 1,
        "ann_type": "A",
        "client_source": "web",
        "stock_list": code,
        "f_node": 0,
        "s_node": 0,
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://emweb.securities.eastmoney.com/",
    }
    news = []
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code == 200:
            js = resp.json()
            items = (js.get("data") or {}).get("list", []) or []
            for it in items:
                col_name = ""
                try:
                    col_name = (it.get("columns") or [{}])[0].get("column_name", "")
                except Exception:
                    pass
                news.append({
                    "title": it.get("title", ""),
                    "time": it.get("notice_date", "") or it.get("display_time", ""),
                    "type": col_name,
                    "code": code,
                })
    except Exception as e:
        print(f"[NEWS] error: {e}")

    _cache_set(cache_key, news)
    return news


@app.route("/api/crawl_guba", methods=["POST"])
def api_crawl_guba():
    """爬取股吧帖子 + 个股公告"""
    try:
        data = request.json or {}
        stock_code = str(data.get("stock_code", "")).strip()
        pages = int(data.get("pages", 2))
        include_news = bool(data.get("include_news", True))

        code_clean = re.sub(r"[^0-9]", "", stock_code)
        if not code_clean:
            return jsonify({"ok": False, "error": "股票代码格式错误,请输入 6 位数字代码 (例如 600519)"}), 400

        pages = min(max(pages, 1), 5)
        posts = crawl_guba(code_clean, max_pages=pages)

        news = []
        if include_news:
            try:
                news = crawl_eastmoney_news(code_clean, max_items=15)
            except Exception as e:
                print(f"[NEWS] skip: {e}")

        return jsonify({
            "ok": True,
            "data": {
                "stock_code": code_clean,
                "posts": posts[:60],
                "posts_count": len(posts),
                "news": news[:30],
                "news_count": len(news),
                "source_urls": {
                    "guba": f"https://guba.eastmoney.com/list,{code_clean}.html",
                },
                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# 12. LLM 情感分析 + 结构化提取
# ═══════════════════════════════════════════════════════════
def _extract_json(text):
    """从 LLM 响应中稳健地抽取 JSON 对象"""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def llm_analyze_posts(posts, news=None, company=""):
    """用大模型对帖子 + 公告做情感分析和结构化提取"""
    if not posts and not news:
        return {"error": "没有可分析的内容"}

    posts_sample = (posts or [])[:25]
    news_sample = (news or [])[:10]

    posts_text = "\n".join(
        f"[P{i+1}] {p.get('title', '')}" for i, p in enumerate(posts_sample)
    ) or "(无)"
    news_text = "\n".join(
        f"[N{i+1}] ({n.get('time', '')}) {n.get('title', '')}"
        for i, n in enumerate(news_sample)
    ) or "(无)"

    prompt = f"""你是一个专业的金融舆情分析师。请对以下来自东方财富股吧的帖子标题和个股公告进行情感分析与结构化信息提取。

【分析对象】{company or '目标股票'}

【股吧帖子标题】
{posts_text}

【个股公告/新闻】
{news_text}

请严格按以下 JSON 格式返回 (只输出 JSON,不要 markdown 代码块,不要解释文字):
{{
  "overall_sentiment": "positive | negative | neutral",
  "sentiment_score": 整数 0-100,越高越乐观,50 为中性,
  "positive_count": 正面帖子数,
  "negative_count": 负面帖子数,
  "neutral_count": 中性帖子数,
  "confidence": 0-1 浮点数,表示整体判断的置信度,
  "hot_topics": ["话题1", "话题2", "话题3", "话题4", "话题5"],
  "key_events": ["关键事件1简述", "关键事件2简述", "关键事件3简述"],
  "market_mood": "一句话市场情绪 (如:谨慎乐观、分歧加剧、恐慌抛售、热情追捧)",
  "risks": ["潜在风险点1", "潜在风险点2"],
  "opportunities": ["机会点1", "机会点2"],
  "per_post_sentiment": [
    {{"index": 1, "sentiment": "positive|negative|neutral", "reason": "简短理由"}}
  ],
  "summary": "2-3 句话总结整体舆情、投资者情绪倾向和主要关注点"
}}

per_post_sentiment 请覆盖所有帖子 (P1~P{len(posts_sample)})。
"""

    raw = call_llm_api(prompt, max_tokens=3000, temperature=0.3)
    if not raw:
        return {"error": "LLM 调用失败,请检查 API Key 或网络"}

    parsed = _extract_json(raw)
    if not parsed:
        return {"error": "LLM 返回的 JSON 格式解析失败",
                "raw_preview": raw[:300]}

    # 字段兜底
    parsed.setdefault("overall_sentiment", "neutral")
    parsed.setdefault("sentiment_score", 50)
    parsed.setdefault("hot_topics", [])
    parsed.setdefault("key_events", [])
    parsed.setdefault("risks", [])
    parsed.setdefault("opportunities", [])
    parsed.setdefault("per_post_sentiment", [])
    parsed.setdefault("summary", "")
    parsed.setdefault("market_mood", "")
    parsed.setdefault("confidence", 0.7)
    return parsed


@app.route("/api/analyze_posts", methods=["POST"])
def api_analyze_posts():
    """对爬到的帖子+公告做情感分析 + 结构化提取"""
    try:
        body = request.json or {}
        posts = body.get("posts", []) or []
        news = body.get("news", []) or []
        company = body.get("company", "") or ""

        if not posts and not news:
            return jsonify({"ok": False, "error": "请先抓取帖子/公告"}), 400

        analysis = llm_analyze_posts(posts, news, company)
        if "error" in analysis and not analysis.get("overall_sentiment"):
            return jsonify({"ok": False,
                            "error": analysis["error"],
                            "raw": analysis.get("raw_preview", "")}), 500

        return jsonify({
            "ok": True,
            "data": {
                "analysis": analysis,
                "total_posts": len(posts),
                "total_news": len(news),
                "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# 13. 对话式 AI 助手 (带上下文)
# ═══════════════════════════════════════════════════════════
chat_sessions = {}   # session_id -> [{role, content}, ...]
MAX_HISTORY = 40


def _build_context_preamble(ctx):
    """把前端传过来的 context 拼成系统提示的补充段落"""
    lines = []
    pd_ = (ctx or {}).get("predict_data") or {}
    ed = (ctx or {}).get("eval_data") or {}
    sd = (ctx or {}).get("sentiment_data") or {}

    if pd_:
        company = pd_.get("company", "目标公司")
        years = pd_.get("years", "")
        indicators = pd_.get("indicators", {}) or {}
        lines.append(f"\n【预测上下文】用户已完成对 {company} 的未来 {years} 年盈利能力预测,模型输出:")
        for key, ind in indicators.items():
            short = ind.get("short", key)
            hist = ind.get("hist") or []
            fore = ind.get("fore") or []
            hv = hist[-1] if hist else None
            fv = fore[-1] if fore else None
            mse = ind.get("train_mse")
            model = ind.get("model_label") or ind.get("model") or ""
            if hv is not None and fv is not None:
                try:
                    lines.append(f"  - {short}: 历史末期 {hv:.2f} → 预测末期 {fv:.2f} (模型:{model}, MSE:{mse})")
                except Exception:
                    lines.append(f"  - {short}: {hv} → {fv} (模型:{model})")

    if ed and ed.get("best_models"):
        lines.append(f"\n【选优上下文】行业 K={ed.get('n_splits','?')} 折CV 选优结果,最优模型映射:")
        labels = ed.get("model_labels", {})
        for k, m in (ed.get("best_models") or {}).items():
            short = (ed.get("indicators", {}).get(k, {}) or {}).get("short", k)
            lines.append(f"  - {short} → {labels.get(m, m)}")

    if sd:
        analysis = sd.get("analysis") or sd
        lines.append("\n【舆情上下文】已爬取股吧+公告并完成情感分析:")
        if analysis.get("overall_sentiment"):
            lines.append(f"  - 整体情感: {analysis.get('overall_sentiment')} (得分 {analysis.get('sentiment_score', 50)}/100)")
        if analysis.get("market_mood"):
            lines.append(f"  - 市场情绪: {analysis.get('market_mood')}")
        if analysis.get("hot_topics"):
            lines.append(f"  - 热门话题: {', '.join(analysis.get('hot_topics')[:5])}")
        if analysis.get("summary"):
            lines.append(f"  - 舆情摘要: {analysis.get('summary')}")

    return "\n".join(lines)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """与 AI 助手多轮对话,携带预测/舆情上下文"""
    try:
        body = request.json or {}
        session_id = str(body.get("session_id", "default"))
        user_msg = (body.get("message") or "").strip()
        context = body.get("context") or {}

        if not user_msg:
            return jsonify({"ok": False, "error": "消息为空"}), 400

        history = chat_sessions.setdefault(session_id, [])

        system_prompt = (
            "你是「慧估」的智能财务助手,精通上市公司财务分析、盈利能力评估、"
            "机器学习时序预测原理以及金融舆情解读。请用简洁、专业、分点的中文回答用户。"
            "回答中若涉及数据,优先引用下方【预测上下文】/【选优上下文】/【舆情上下文】里的具体数值。"
            "涉及投资建议时保持客观中性,并提示风险。"
        )
        ctx_text = _build_context_preamble(context)
        if ctx_text:
            system_prompt += ctx_text

        messages = [{"role": "system", "content": system_prompt}]
        for m in history[-20:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_msg})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.7,
        }
        resp = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        js = resp.json()
        reply = js["choices"][0]["message"]["content"]

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY:
            chat_sessions[session_id] = history[-MAX_HISTORY:]

        return jsonify({
            "ok": True,
            "data": {
                "reply": reply,
                "session_id": session_id,
                "history_len": len(chat_sessions[session_id]),
            }
        })

    except requests.HTTPError as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"LLM HTTP 错误: {e}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    """清空对话历史"""
    try:
        body = request.json or {}
        sid = str(body.get("session_id", "default"))
        chat_sessions[sid] = []
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# 14. 元数据 + 静态文件
# ═══════════════════════════════════════════════════════════
@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({
        "ok": True,
        "data": {
            "models": [
                {"name": m, "label": MODEL_REGISTRY[m]["label"], "family": MODEL_REGISTRY[m]["family"]}
                for m in MODEL_NAMES
            ]
        }
    })


@app.route("/")
def index():
    return send_from_directory(".", "慧估_Smart_Prediction_new.html")


if __name__ == "__main__":
    try:
        print("开始执行主函数...")
        print("=" * 60)
        print("  慧估后端 v3 — 多模型 + K折CV + AI报告 + 舆情分析 + 对话")
        print(f"  已注册模型: {', '.join(MODEL_NAMES)}")
        print("  API 端点:")
        print("    POST /api/evaluate        行业选优 (K折CV)")
        print("    POST /api/predict         公司预测")
        print("    POST /api/generate_report AI 生成分析报告")
        print("    POST /api/crawl_guba      爬取股吧/公告")
        print("    POST /api/analyze_posts   LLM 情感分析 + 结构化")
        print("    POST /api/chat            AI 对话助手")
        print("    POST /api/chat/clear      清空对话")
        print("  访问: http://localhost:5000")
        print("=" * 60)
        print("启动Flask应用...")
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()