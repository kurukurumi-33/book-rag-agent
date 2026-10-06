import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from app.services.search import search


A_list = search("线代")              
for h in A_list:
    print(h)
print("="*50)
B_list = search("带笔记的便宜线代")   
for h in B_list:
    print(h)
