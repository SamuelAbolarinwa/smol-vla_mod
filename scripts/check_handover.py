"""Validate source and notebook syntax without ML imports or execution."""
from pathlib import Path
import ast
import json
ROOT = Path(__file__).resolve().parents[1]
count = 0
for folder in ("src", "tests", "scripts"):
    for path in (ROOT / folder).rglob("*.py"):
        compile(path.read_text(), str(path), "exec")
        count += 1
notebooks = list((ROOT / "Notebooks").rglob("*.ipynb"))
for path in notebooks:
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4, path
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs") and cell.get("execution_count") is None, path
        source = "".join(cell["source"])
        tree = ast.parse(source, filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if set(names) & {"BOOTSTRAP", "EVALUATION_SCRIPT"}:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, str):
                        compile(value, str(path) + " embedded script", "exec")
print(f"PASS: {count} Python files and {len(notebooks)} cleared notebooks; syntax only")
