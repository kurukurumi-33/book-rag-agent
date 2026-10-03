"""检索质量测试集（M2-3）：跑一批「查询 → 期望结果」，算命中率。

这个脚本回答一个问题：**改了一版检索（换拼法 / 调阈值 / 换模型），
效果是变好还是变坏？**

为什么必须有一个固定测试集：
  没有它，你每次调参都只能「感觉好像好了一点」。有了它，好坏是一个数。
  调优前后的这个对比，就是面试素材（见 docs/需求与接口.md §5.3）。

为什么要走 HTTP 而不是直接 import search()：
  走接口才是**真实的调用路径** —— 它会验证 min_score 的默认值有没有真的
  接上、响应结构对不对。直接调函数会绕过这些。（见面试素材第 12 条：
  验收用例必须能走到目标代码路径。）

用法（先确保 Spring Boot :8080 和 AI 服务 :8000 都起着，且索引已建）：

    .venv\\Scripts\\python.exe scripts\\eval_search.py

    # 存一份快照，用于调优前后对比
    .venv\\Scripts\\python.exe scripts\\eval_search.py --save 调阈值前

    # 和之前存的快照对比（看哪些用例翻盘了）
    .venv\\Scripts\\python.exe scripts\\eval_search.py --diff 调阈值前

    # 临时试别的阈值，不改代码
    .venv\\Scripts\\python.exe scripts\\eval_search.py --min-score 0.55

【你要改的只有 CASES】—— 见下面的说明。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 不是终端时 stdout 是块缓冲的，不加这行能几十秒一行不显示，看起来像卡死
sys.stdout.reconfigure(line_buffering=True)

import argparse  # noqa: E402
import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import httpx  # noqa: E402

AI_SERVICE = "http://127.0.0.1:8000"

# 问服务要几条。取大一点（10），这样即使 top-3 没中，你也能看到
# 「其实第 4 条是对的」—— 这是判断"拼法问题"还是"阈值问题"的关键线索。
TOP_K = 10

# 「返回的 top-N 里出现期望书名」算命中。文档 §5.3 定的目标是 top-3。
HIT_AT = 3
TARGET_RATE = 0.80

# 索引空了的话，探针查询用这个。选「线代」是因为它简短、肯定该有结果。
PROBE_QUERY = "线代"


@dataclass
class Case:
    """一条测试用例。

    :param query:  用户会怎么问
    :param expect: 哪些书名算答对。返回的 top-3 里出现**任意一个**就算命中。
                   写 [] 表示负样本 —— 要求服务返回**空列表**。
    :param note:   给人看的备注，不参与判定
    """

    query: str
    expect: list[str]
    note: str = ""

    @property
    def is_negative(self) -> bool:
        return not self.expect


# ============================================================================
# 【 测试用例 —— 你来填 】
# ============================================================================
#
# 规则：
#   - expect 写「哪些书名算答对」。匹配是**精确匹配**（去掉首尾空格后相等）。
#     要算多本就把名字都列上，例如 ["数据结构", "数据结构与算法分析"]。
#   - expect 写 [] = 负样本，要求返回空列表。
#
# 下面 3 条是**已经验过的回归基线，别删**。它们是保险丝：
# 保证你后面调参时，没把已经对的东西调坏。
#
# 往下面接着加，凑到 15~20 条，覆盖这五类（这才是测试集的价值所在 ——
# 光测「同济高数」这种干净查询，所有拼法都是 100% 命中，测不出任何东西）：
#
#   ① 精确书名          "计算机网络 谢希仁"
#   ② 口语说法          "有没有高数书" / "我想找本高数的" / "高数还有吗"
#   ③ 名字很像的书      库里同时有《数据结构》《数据结构与算法分析》
#                       《数据结构与算法分析 C语言描述》—— 搜「数据结构」该出哪本？
#   ④ 书名里带别人名字   "同济版高数" —— 线代也是同济版，「同济」这个词帮不上忙
#   ⑤ 只说用途          "考研数学用的" / "大二下学期要用的"
#
# 负样本别只写「量子力学」（太明显了，函数随便都能过）。挑**看起来该有、其实没有**的，
# 比如「有机化学」「法学概论」，甚至「同济版线性代数」—— 有同济版线代，但没有「同济版」这个说法。
#
# 想看库里到底有哪些书，跑：
#     curl -s "http://127.0.0.1:8080/api/posts?size=500"
# 或直接查库：
#     export MYSQL_PWD=<密码>   # 密码走环境变量，别写在命令行里（写进去会跟着文件进 git）
#     mysql -uroot -e "USE book_agent; SELECT DISTINCT book_name FROM book_post WHERE extract_status='DONE';"
# ============================================================================

# ⚠️ 这 16 条是从 scripts/eval_runs/阈值0.60.json 重建的（原文件被一次回退清空了，
#    VS Code 本地历史没留下更早的副本）。query 和 expect 是快照里的原数据，
#    note 是重建时补的。快照那轮 16/16 全过。
CASES: list[Case] = [
    # ---- 回归基线（已验过，别删）----
    Case("同济高数", ["高等数学"], "精确简称"),
    Case("计算机网络", ["计算机网络"], "精确书名"),
    Case("量子力学", [], "负样本（明显没有）"),

    # ---- 精确书名 ----
    Case("软件工程", ["软件工程"], "精确书名"),

    # ---- 口语说法 ----
    Case("有没有高数书", ["高等数学"], "口语：带疑问句"),
    Case("高数还有吗", ["高等数学"], "口语：带疑问句"),
    Case("我想找本线代", ["线性代数"], "口语。⚠️ 它是正例的下界 0.6095，阈值就是照它定的"),

    # ---- 近义簇 ----
    Case("数据结构", ["数据结构"], "近义簇：库里还有《数据结构与算法分析》等两个变体"),
    Case("毛概", ["毛概", "毛泽东思想和中国特色社会主义理论体系概论"],
         "近义簇 + M1 抽取缺陷（同一个词被抽成两个名字，见面试素材第 6 条）"),
    Case("新视野大学英语",
         ["新视野大学英语", "新视野大学英语读写教程",
          "新视野大学英语综合教程", "新视野大学英语视听说教程"],
         "近义簇：库里 5 个变体"),
    Case("计算机操作系统", ["操作系统", "计算机操作系统"], "近义簇"),

    # ---- 书名里带别人名字 ----
    Case("同济版高数", ["高等数学"], "「同济」是版本体系不是出版社，这个词帮不上忙"),

    # ---- 只说用途 ----
    Case("考研数学用的", ["考研数学复习全书"], "只说用途不说书名"),

    # ---- 负样本：看起来该有、其实没有 ----
    Case("有机化学", [], "负样本"),
    Case("宏观经济学", [], "负样本"),
    Case("法学概论", [],
         "负样本，全组最难：与「…理论体系概论」共享「概论」二字（字面重叠，不是语义），"
         "原始分 0.5902 —— 0.58 拦不住它，这才是阈值定 0.60 的原因"),

    # ---- 回退后新加的，阈值 0.60 那轮没跑过 ----
    Case("java还有吗", ["Java核心技术(10)"], "只有书本关键字。期望书名没核过，可能写错"),
]


# ============================================================================
# 以下不用改
# ============================================================================


def call_search(client: httpx.Client, query: str, min_score: float | None) -> list[dict]:
    """打一次 /search，返回 hits 列表。"""
    body: dict = {"query": query, "top_k": TOP_K}
    # 只有显式传了才带上 min_score —— 不传就让服务用它自己的默认常量。
    # 这样默认行为和服务真实默认值始终一致，不会出现「脚本和服务想的不是一回事」。
    if min_score is not None:
        body["min_score"] = min_score

    resp = client.post(f"{AI_SERVICE}/search", json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()["hits"]


def hit_names(hits: list[dict]) -> list[str]:
    """从 hits 里取书名，给判定和显示用。"""
    return [(h.get("book_name") or h.get("text") or "?") for h in hits]


def top_str(hits: list[dict]) -> str:
    """把前 HIT_AT 条的书名拼成一行展示，空的显示 (空)。"""
    return " / ".join(hit_names(hits[:HIT_AT])) or "(空)"


def judge(case: Case, hits: list[dict]) -> bool:
    """这条用例算不算过。"""
    if case.is_negative:
        # 负样本：要求一条都不返回
        return len(hits) == 0
    # 正例：top-HIT_AT 里出现任意一个期望书名
    top = hit_names(hits[:HIT_AT])
    return any(name in case.expect for name in top)


def preflight(client: httpx.Client) -> bool:
    """开跑前先确认服务和索引都活着。不然会得到一堆假失败。"""
    try:
        health = client.get(f"{AI_SERVICE}/health", timeout=5).json()
    except Exception as e:
        print(f"❌ 连不上 AI 服务 {AI_SERVICE}")
        print(f"   （{type(e).__name__}: {e}）")
        print("   起服务：.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000")
        return False

    print(f"AI 服务 OK：{health}")

    # 探针：索引空的话，后面所有用例都会「失败」，但根因跟检索质量无关。
    # 先在这里拦住，否则你会拿一份全是 ❌ 的报告去调拼法，白调半天。
    if not call_search(client, PROBE_QUERY, None):
        print()
        print(f"⚠️  探针查询「{PROBE_QUERY}」返回 0 条 —— 索引很可能是空的。")
        print("   先建索引（220 条约 6 秒）：")
        print("     curl -X POST http://127.0.0.1:8000/index/build "
              "-H 'Content-Type: application/json' -d '{}'")
        return False

    return True


def run(client: httpx.Client, min_score: float | None) -> list[dict]:
    """跑全部用例，返回每条的判定结果。"""
    rows = []
    for i, case in enumerate(CASES, 1):
        try:
            hits = call_search(client, case.query, min_score)
        except Exception as e:
            print(f"[{i:>2}] {case.query:<22} ⚠️  请求失败：{type(e).__name__}: {e}")
            rows.append({"case": case, "ok": False, "hits": [], "error": str(e)})
            continue

        ok = judge(case, hits)
        mark = "✅" if ok else "❌"
        top1_score = f"{hits[0]['score']:+.4f}" if hits else "  —   "

        print(f"[{i:>2}] {case.query:<22} {mark}  {top1_score}  {top_str(hits)}")

        # 失败时把 top-K 全打出来 —— 常见情况是「第 1 名错了但第 4 名是对的」，
        # 这说明阈值把对的那条砍了，而不是拼法找不到它。两种问题的修法完全不同。
        if not ok and len(hits) > HIT_AT:
            print(f"      ↳ 第 {HIT_AT + 1}~{TOP_K} 名："
                  f"{' / '.join(hit_names(hits[HIT_AT:]))}")

        rows.append({"case": case, "ok": ok, "hits": hits})

    return rows


def summarize(rows: list[dict]) -> dict:
    """算命中率并打印失败清单。"""
    pos = [r for r in rows if not r["case"].is_negative]
    neg = [r for r in rows if r["case"].is_negative]
    pos_ok = sum(1 for r in pos if r["ok"])
    neg_ok = sum(1 for r in neg if r["ok"])

    print()
    print("=" * 78)
    print("汇总")
    print("=" * 78)

    if pos:
        rate = pos_ok / len(pos)
        flag = "✅ 达标" if rate >= TARGET_RATE else "❌ 未达标"
        print(f"  正例 top-{HIT_AT} 命中率： {pos_ok}/{len(pos)} = {rate:.1%}"
              f"   （目标 ≥ {TARGET_RATE:.0%}） {flag}")
    else:
        rate = 0.0
        print("  正例： 0 条 —— 你还没填用例呢")

    if neg:
        print(f"  反例通过率：        {neg_ok}/{len(neg)} = "
              f"{neg_ok / len(neg):.1%}   （要求 100%）")

    failed = [r for r in rows if not r["ok"]]
    if failed:
        print()
        print(f"  失败用例（{len(failed)} 条）—— 这些就是你要查的 bad case：")
        for r in failed:
            c = r["case"]
            expect = "空" if c.is_negative else " / ".join(c.expect)
            print(f"    - {c.query!r:<24} 期望 {expect:<20} 实际 {top_str(r['hits'])}")
            if c.note:
                print(f"      （{c.note}）")
    else:
        print()
        print("  全部通过 🎉")

    return {
        "positive_rate": rate if pos else None,
        "positive_ok": pos_ok,
        "positive_total": len(pos),
        "negative_ok": neg_ok,
        "negative_total": len(neg),
    }


def snapshot_path(name: str) -> Path:
    """快照存到 scripts/eval_runs/<name>.json。"""
    folder = Path(__file__).resolve().parent / "eval_runs"
    folder.mkdir(exist_ok=True)
    return folder / f"{name}.json"


def save_snapshot(name: str, rows: list[dict], summary: dict) -> None:
    path = snapshot_path(name)
    payload = {
        "summary": summary,
        "cases": [
            {
                "query": r["case"].query,
                "expect": r["case"].expect,
                "ok": r["ok"],
                "top": hit_names(r["hits"][:HIT_AT]),
                "top1_score": r["hits"][0]["score"] if r["hits"] else None,
            }
            for r in rows
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n快照已存：{path}")


def diff_snapshot(name: str, rows: list[dict], summary: dict) -> None:
    """和快照对比 —— 调优到底有没有用，看这里。"""
    path = snapshot_path(name)
    if not path.exists():
        print(f"\n⚠️  找不到快照 {path}，跳过对比")
        return

    old = json.loads(path.read_text(encoding="utf-8"))
    old_map = {c["query"]: c for c in old["cases"]}
    new_map = {r["case"].query: r for r in rows}

    print()
    print("=" * 78)
    print(f"对比快照：{name}")
    print("=" * 78)

    o = old["summary"]
    print(f"  正例命中率： {o['positive_ok']}/{o['positive_total']} "
          f"→ {summary['positive_ok']}/{summary['positive_total']}")

    flipped_up = [q for q, r in new_map.items() if r["ok"] and q in old_map and not old_map[q]["ok"]]
    flipped_down = [q for q, r in new_map.items() if not r["ok"] and q in old_map and old_map[q]["ok"]]
    added = [q for q in new_map if q not in old_map]

    def arrow(q: str) -> str:
        old_top = " / ".join(old_map[q]["top"]) or "(空)"
        return f"     {q!r}：{old_top}  →  {top_str(new_map[q]['hits'])}"

    if flipped_up:
        print(f"\n  ✅ 修好了（{len(flipped_up)} 条）：")
        for q in flipped_up:
            print(arrow(q))
    if flipped_down:
        print(f"\n  ❌ 调坏了（{len(flipped_down)} 条）—— 这才要命，别只看总命中率：")
        for q in flipped_down:
            print(arrow(q))
    if added:
        print(f"\n  ＋ 新增用例 {len(added)} 条（快照里没有，无法对比）："
              f"{'、'.join(repr(q) for q in added)}")
    if not (flipped_up or flipped_down):
        print("\n  没有用例翻盘 —— 这次改动对测试集没有影响")


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量测试集（M2-3）")
    parser.add_argument("--save", metavar="名字", help="把本次结果存成快照")
    parser.add_argument("--diff", metavar="名字", help="和之前存的快照对比")
    parser.add_argument("--min-score", type=float, default=None,
                        help="临时指定阈值（不传则用服务默认值）")
    args = parser.parse_args()

    if not CASES:
        print("CASES 是空的 —— 先在脚本里填测试用例")
        return

    n_pos = sum(1 for c in CASES if not c.is_negative)
    n_neg = len(CASES) - n_pos
    print("=" * 78)
    print(f"检索质量测试集 —— {len(CASES)} 条用例（正例 {n_pos} / 反例 {n_neg}）")
    print("=" * 78)
    print(f"AI 服务：{AI_SERVICE}   top_k={TOP_K}   命中判定：top-{HIT_AT}")
    if args.min_score is not None:
        print(f"⚠️  临时阈值：{args.min_score}（覆盖服务默认值）")
    print()

    with httpx.Client() as client:
        if not preflight(client):
            return
        print()
        print("-" * 78)
        rows = run(client, args.min_score)
        summary = summarize(rows)

    if args.save:
        save_snapshot(args.save, rows, summary)
    if args.diff:
        diff_snapshot(args.diff, rows, summary)


if __name__ == "__main__":
    main()
