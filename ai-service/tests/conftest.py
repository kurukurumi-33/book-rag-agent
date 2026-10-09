"""pytest 的全局初始化。

这个文件里的代码**在收集任何测试之前**自动执行一次。

## 为什么需要它（两个都必须在 import app.* 之前做）

### 1. 造假的环境变量，让测试不依赖本机的 .env

`app/config.py` 最后一行是 `settings = Settings()` —— 它在 **import 时**就执行，
而 `llm_api_key` 是**必填项**。本机有 `.env` 所以没事，但 CI 上 `.env` 不存在
（它被 .gitignore 排除了，永远不进仓库），于是 CI 会直接崩在 import 阶段：
`ValidationError: llm_api_key Field required`。

解法：在 import 之前先把环境变量塞进去。
pydantic-settings 的优先级是「环境变量 > .env 文件」，所以这里设的值会覆盖 .env ——
正好，测试本来就不该用真实 key（**一次真实 LLM 调用都不会发生**，
所有外部依赖都在用例里被 monkeypatch 掉了）。

### 2. 把工程根目录加进 sys.path

测试文件的路径是 `ai-service/tests/xxx.py`，而 `import app.xxx` 要求
`ai-service/` 在 sys.path 里。pytest 的 rootdir 是 ai-service（因为有 pytest.ini），
但它不会自动把 rootdir 加进 sys.path（只有 conftest.py 所在目录会被加）。
所以手动加一次。
"""

import os
import sys
from pathlib import Path

# ---- 1. 造假配置（必须在 import app.* 之前）----
os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("BOOK_SERVICE_URL", "http://127.0.0.1:8080")

# ---- 2. 让 `import app.*` 能找到包 ----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
