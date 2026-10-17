import argparse
import html
import json
from pathlib import Path
import sys

from .evidence import audit_splits, build_report


def render_html(report):
    sections = []
    for dataset, data in report["datasets"].items():
        for cohort, models in data["metrics"].items():
            rows = []
            for model, metric in models.items():
                n = metric["total"]
                lo, hi = metric["accuracy_ci95_wilson"]
                change = data["cohort_paired_vs_base"][cohort].get(model)
                delta = "—" if change is None else f"{change['delta_percentage_points']:+.2f} pp ({change['gain']}↑ / {change['loss']}↓)"
                rows.append(f"<tr><td>{html.escape(model)}</td><td>{metric['correct']}/{n}</td>"
                            f"<td>{metric['accuracy']:.2%}</td><td>{lo:.2%}–{hi:.2%}</td>"
                            f"<td>{delta}</td><td>{metric['unconfirmed']}</td>"
                            f"<td>{metric['truncated']/n:.2%}</td><td>{metric['mean_tokens']:.1f}</td></tr>")
            sections.append(f"<section><h2>{html.escape(dataset)} · {html.escape(cohort)}</h2>"
                            "<div class='scroll'><table><thead><tr><th>模型</th><th>判对/总数</th><th>自动判对率</th>"
                            "<th>95% Wilson 区间</th><th>相对 Base 配对变化</th><th>未确认</th><th>截断</th><th>平均 tokens</th>"
                            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>")
    provenance = "\n".join(f"{s['dataset']}/{s['model']}\n{s['path']}\nSHA256 {s['sha256']}\n" for s in report["sources"])
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Qwen SFT / GRPO 实验证据</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#101827;color:#e8edf5;margin:0 auto;padding:32px;max-width:1300px}
h1{font-size:34px}h2{font-size:21px;color:#88d8ce}section{margin:28px 0;padding:22px;background:#192538;border-radius:14px}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{text-align:left;padding:12px;border-bottom:1px solid #354155}
th{color:#a6b8d3}p{max-width:1000px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#88d8ce}
</style><h1>Qwen3-4B · SFT → GRPO</h1>
<p>已有实验的可追溯评测。SFT 改善格式，GRPO 恢复部分正确率；MATH 最终答案准确率仍以 Base 最高。长度下降和格式提升不能单独证明推理能力提升。</p>
<p>remaining1824 为此前旧 200 题之外的确认子集；all 已包含 previous200，二者不能重复相加。自动评分不认证逐步推导。未确认仍计入总数。</p>
""" + "".join(sections) + f"<section><h2>方法边界</h2><p>{html.escape(report['scope'])}</p><p>↑ 表示 Base 未判对而候选判对；↓ 表示相反。区间仅描述题目抽样不确定性，不能据此推断小幅提升稳定成立。</p></section><details><summary>输入文件与 SHA256</summary><pre>{html.escape(provenance)}</pre></details></html>"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible CPU evidence tools for SFT/GRPO")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Verify pinned inputs and recompute report")
    report.add_argument("--root", type=Path, default=Path.cwd())
    report.add_argument("--registry", type=Path, default=Path("configs/evidence_registry.json"))
    report.add_argument("--output", type=Path, required=True, help="Fresh directory; refuses overwrites")
    audit = commands.add_parser("audit", help="Fail on duplicate or overlapping split questions")
    audit.add_argument("--split", action="append", required=True, metavar="NAME=PATH")
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            paths = {}
            for item in args.split:
                name, separator, path = item.partition("=")
                if not separator or not name or not path or name in paths:
                    raise ValueError("Each --split must have a unique NAME=PATH")
                paths[name] = Path(path)
            result = audit_splits(paths)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 1
        registry_path = args.registry if args.registry.is_absolute() else args.root / args.registry
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        result = build_report(args.root, registry)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output / "index.html").write_text(render_html(result), encoding="utf-8")
        print(f"Verified {len(result['sources'])} inputs. Report: {args.output / 'index.html'}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
