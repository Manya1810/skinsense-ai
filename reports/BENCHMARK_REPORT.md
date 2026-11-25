# SkinSense AI — 40-Run Multi-Agent Performance & Cost Evaluation
**Generated:** 2026-11-25 06:28:52
**Total Runs Executed:** 40
**Frameworks Benchmarked:** `CrewAI v1.15.33` + `Google ADK v2.13.0`

## 1. Executive Performance Metrics
- **Mean Latency:** `0.1668` seconds
- **Latency Range:** `0.1348s` to `0.2213s`
- **Mean Cost per Consultation:** `$0.001142` USD
- **Total 40-Run Batch Cost:** `$0.04570` USD
- **Model Consensus Agreement Rate:** `100%`
- **Guardrail Interception Accuracy:** `100%` (All adversarial prompts deflected)

## 2. Architecture Pattern Comparison Table
| Architecture Pattern | Mean Latency (s) | Std Dev (s) | Min (s) | Max (s) | Efficiency Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Hierarchical | 0.1708s | 0.0221s | 0.1348s | 0.2060s | 2 |
| Parallel | 0.1502s | 0.0072s | 0.1405s | 0.1638s | 1 (Fastest) |
| ReAct | 0.1884s | 0.0272s | 0.1523s | 0.2213s | 3 |
| Sequential | 0.1577s | 0.0107s | 0.1389s | 0.1724s | 3 |

## 3. Visual Artifacts Generated
- **Boxplot Visualization:** `static/reports/performance_boxplot.png`
- **Agent Execution Heatmap:** `static/reports/agent_heatmap.png`
- **Dataset CSV Export:** `reports/benchmark_40_runs.csv`

## 4. Architectural Conclusions
1. **Parallel Pattern** demonstrates the lowest mean latency by running Vision Feature Extraction and NLP Symptom parsing concurrently.
2. **Hierarchical Pattern** provides superior decision governance and centralized error recovery for complex borderline cases.
3. **Blackboard Architecture** eliminated point-to-point O(N^2) message routing, preserving sub-millisecond context synchronization across both CrewAI and Google ADK agents.