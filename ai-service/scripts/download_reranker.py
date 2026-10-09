"""下载 rerank 用的 cross-encoder（BAAI/bge-reranker-base）。

## 为什么不用 `snapshot_download`

因为**下不下来**。hf-mirror.com 不支持 HuggingFace 新的 **Xet** 存储协议：
`snapshot_download` 拿到签名 URL 之后会被 302 到 `cas-bridge.xethub.hf.co`，
然后 `401 Unauthorized` 或者读超时。实测过两次：

    第一次（纯 HF_ENDPOINT=hf-mirror）  → 401 Unauthorized
    第二次（再加 HF_HUB_DISABLE_XET=1） → 那个开关没拦住，还是走 xet，读超时

所以这个脚本**完全不碰 huggingface_hub**，改成两步：

    ① GET /api/models/BAAI/bge-reranker-base   → 列出仓库里有哪些文件
    ② GET /resolve/main/<每个文件>              → 普通 HTTPS 直链，走 S3，镜像能正常代理

## 只下推理必需的

跳过 `.bin`（有 safetensors 就不要它）、`.onnx`、`.h5` 这些冗余格式，
1.1GB 里能省掉几百 MB。

## 幂等

文件下到 `<name>.part` 再 rename，中途断了重跑即可 —— 已完成的文件会重下一遍
（不做断点续传，简单优先），但**不会留下半个文件当成品**。

用法（在 ai-service/ 下）：

    .venv\\Scripts\\python.exe scripts\\download_reranker.py

可选指定目标目录：

    .venv\\Scripts\\python.exe scripts\\download_reranker.py D:\\somewhere\\else
"""

import json
import sys
import time
from pathlib import Path

import httpx

REPO = "BAAI/bge-reranker-base"
BASE = "https://hf-mirror.com"

# 默认落点：ai-service/models/<模型名>。rerank.py 的 _LOCAL_DIR 就指这儿。
_DEFAULT_DEST = Path(__file__).resolve().parent.parent / "models" / REPO.split("/")[-1]

# 有 safetensors 时这些格式一律不要（同一份权重，纯冗余）
_SKIP_SUFFIX = (".bin", ".onnx", ".onnx_data", ".h5", ".msgpack", ".tflite", ".ot")


def main() -> None:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DEST
    dest.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30.0, follow_redirects=True) as c:
        info = c.get(f"{BASE}/api/models/{REPO}").json()
        files = [s["rfilename"] for s in info["siblings"]]

    # 只留根目录下的文件：子目录（onnx/ 之类）整棵跳过，正好和 _SKIP_SUFFIX 一个意思
    wanted = [f for f in files if "/" not in f and not f.endswith(_SKIP_SUFFIX)]
    if any(f.endswith(".safetensors") for f in wanted):
        wanted = [f for f in wanted if not f.endswith(".bin")]

    print(f"仓库共 {len(files)} 个文件，要下 {len(wanted)} 个 → {dest}")
    for f in wanted:
        print("  -", f)

    for name in wanted:
        target = dest / name
        url = f"{BASE}/{REPO}/resolve/main/{name}"
        for attempt in range(1, 6):
            try:
                with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length") or 0)
                    tmp = target.with_suffix(target.suffix + ".part")
                    done = 0
                    t0 = time.time()
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_bytes(1 << 20):
                            fh.write(chunk)
                            done += len(chunk)
                            if total:
                                pct = done * 100 // total
                                sys.stdout.write(
                                    f"\r  {name} {pct}% ({done >> 20}/{total >> 20} MB)"
                                )
                                sys.stdout.flush()
                    tmp.replace(target)  # 先写 .part 再改名：断了不留半个成品
                    print(f"\r  {name} 完成 {done >> 20} MB，{time.time() - t0:.1f}s")
                    break
            except Exception as e:  # noqa: BLE001 —— 网络抖动重试，其他错误也一样重试
                print(f"\n  {name} 第 {attempt} 次失败：{type(e).__name__}: {e}")
                if attempt == 5:
                    raise
                time.sleep(2 * attempt)

    print("\n全部完成：", dest)
    total_bytes = 0
    for p in sorted(dest.iterdir()):
        size = p.stat().st_size
        total_bytes += size
        print(f"  {p.name:<32}{size:>14,}")
    print(f"  {'合计':<32}{total_bytes:>14,}")

    # 自检：rerank.py 靠 config.json 判断「本地目录是不是完整的模型」
    if not (dest / "config.json").exists():
        print("\n⚠️  没有 config.json —— rerank.py 会认为这个目录不可用，")
        print("   转而按模型名去 hub 找（然后撞上 Xet 那个坑）。请检查上面的下载日志。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
