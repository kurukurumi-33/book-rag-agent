"""生成「以图搜书」的测试图 —— 手上没有真书照片时用这个。

## 为什么需要它

`check_vision.py` 要一张图才能跑。别人 clone 这个仓库时手边没有书，
`test-images/` 是空的（而且它在 `.gitignore` 里，永远不会跟仓库走）。
这个脚本让你在不拍照、不下载、不联网的情况下，造出一张能跑通链路的图。

## ⚠️ 它不是真图的替代品

合成图**比真实照片干净得多**：没有反光、没有形变、没有遮挡、字又清楚。
用它跑通只能证明「**链路是通的**」—— key 有效、协议对得上、
多行解析没写错。它**证明不了**「书脊识别得准」。

书脊的真实难点（字小、竖排、反光、被邻书挤掉一半）在合成图里一个都不存在。
所以：

    合成图  →  验证链路。绿了是应该的，红了是真有 bug。
    真实图  →  验证效果。这才是验收，见 test-images/README.md。

三张图的对应关系也是这个分工：

    cover    正面照，最容易。链路挂没挂先拿它试。
    spines   一排书脊，**专门压多本书的解析路径**（这是唯一能不发真图
             就验到多行输出的手段）。
    notbook  明显不是书的图。期望模型一个字都不抽 —— 这是**负例**，
             看 prompt 里那句「看不清的不要猜」有没有真的拦住它。
             模型要是对着一片渐变色硬编出个书名，说明那句没用。

## 怎么用

    cd ai-service
    ./.venv/Scripts/python.exe scripts/make_test_image.py cover "高等数学" ../test-images/cover.jpg
    ./.venv/Scripts/python.exe scripts/make_test_image.py spines "高等数学,线性代数,概率论与数理统计,数据结构,大学物理" ../test-images/spines.jpg
    ./.venv/Scripts/python.exe scripts/make_test_image.py notbook ../test-images/notbook.jpg

生成完直接：
    ./.venv/Scripts/python.exe scripts/check_vision.py ../test-images/spines.jpg

## 依赖

`Pillow` —— **不在 requirements.txt 里**，因为只有这个脚本用得到，
服务本身不依赖它。要跑先 `pip install pillow`。
（故意不写进 requirements：为一台机器上偶尔跑一次的辅助脚本，
给所有部署装一个图像库不合理。）
"""

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，打中文会炸。这行必须在任何 print 之前执行。
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

# Windows 自带的中文字体，按优先级试。没有的话也不是致命错误 ——
# 后面会退化成「用默认字体画方块」，图还是能生成（只是模型多半认不出）。
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
    r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    "/System/Library/Fonts/PingFang.ttc",              # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
]

_COVER_BG = (58, 92, 138)
_COVER_FG = (245, 245, 240)
_SPINE_COLORS = [
    (150, 60, 55),
    (60, 90, 70),
    (70, 75, 120),
    (140, 105, 50),
    (95, 70, 110),
    (50, 100, 110),
    (120, 80, 70),
]


def _load_font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    print("⚠️ 没找到中文字体，退回默认字体 —— 生成的字可能显示成方块，模型多半认不出")
    return ImageFont.load_default()


def make_cover(title: str, out: Path) -> None:
    """一本书的正面照：纯色底 + 大标题。最容易的一档。"""
    from PIL import Image, ImageDraw

    w, h = 600, 850
    img = Image.new("RGB", (w, h), _COVER_BG)
    draw = ImageDraw.Draw(img)

    font = _load_font(64)
    # 标题居中（多行就用 \n 分开）
    lines = title.split("\\n")
    line_h = 84
    total = line_h * len(lines)
    y = (h - total) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text(((w - (bbox[2] - bbox[0])) // 2, y), line, font=font, fill=_COVER_FG)
        y += line_h

    img.save(out, quality=88)
    print(f"📕 cover  → {out}   {w}x{h}   「{title}」")


def make_spines(titles: list[str], out: Path) -> None:
    """一排书脊：几本书并排立着，书名**竖排**（跟真实书脊一样）。

    ⚠️ 竖排是有意做的。真实的书脊大多竖排，横排的合成图会把难度降得太低，
    跑绿了也说明不了任何事。
    """
    from PIL import Image, ImageDraw

    spine_w = 90
    h = 900
    w = spine_w * len(titles)
    img = Image.new("RGB", (w + 40, h), (225, 222, 214))  # 书架底色
    draw = ImageDraw.Draw(img)

    font = _load_font(46)
    for i, title in enumerate(titles):
        x0 = i * spine_w
        color = _SPINE_COLORS[i % len(_SPINE_COLORS)]
        draw.rectangle([x0, 0, x0 + spine_w - 6, h], fill=color)

        # 竖排：一个字一行，从书脊顶部往下排
        y = 40
        for ch in title:
            bbox = draw.textbbox((0, 0), ch, font=font)
            draw.text(
                (x0 + (spine_w - 6 - (bbox[2] - bbox[0])) // 2, y),
                ch,
                font=font,
                fill=(240, 238, 232),
            )
            y += 52
            if y > h - 60:
                break

    img.save(out, quality=88)
    print(f"📚 spines → {out}   {w + 40}x{h}   {len(titles)} 本：{'、'.join(titles)}")


def make_notbook(out: Path) -> None:
    """明显不是书的图：一块斜向渐变色。**负例**用。

    期望模型一个字都不抽。它要是硬编出一个书名，说明 prompt 里那句
    「看不清的不要猜」没拦住 —— 这个负例比任何正例都更能说明问题。
    """
    from PIL import Image

    w, h = 800, 600
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (
                int(60 + 120 * x / w),
                int(140 + 80 * y / h),
                int(180 - 60 * x / w),
            )
    img.save(out, quality=88)
    print(f"🚫 notbook → {out}   {w}x{h}   负例：期望模型返回「无」")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成以图搜书的测试图（合成，非真实照片）")
    parser.add_argument(
        "kind", choices=["cover", "spines", "notbook"], help="生成哪种图"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help='书名。spines 用逗号分隔多本（"高等数学,线性代数"）；cover 用 \\n 分行；notbook 不需要',
    )
    parser.add_argument("out", help="输出路径，例如 ../test-images/cover.jpg")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        if args.kind == "cover":
            make_cover(args.text or "高等数学", out)
        elif args.kind == "spines":
            titles = [t.strip() for t in (args.text or "").split(",") if t.strip()]
            if not titles:
                parser.error("spines 需要书名，例如：spines \"高等数学,线性代数\" out.jpg")
            make_spines(titles, out)
        else:
            make_notbook(out)
    except ImportError:
        print("❌ 缺 Pillow。装一下：")
        print("   ./.venv/Scripts/python.exe -m pip install pillow")
        return 1

    print()
    print("👉 但记住：**合成图只能证明链路是通的，证明不了识别准不准**。")
    print("   真实验收要拿一排真书脊的照片，见 test-images/README.md。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
