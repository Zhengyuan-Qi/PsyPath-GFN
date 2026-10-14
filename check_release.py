"""Validate archive content and checksums without installing model dependencies."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DENIED = {'.pt', '.pth', '.safetensors', '.wav', '.mp4', '.npz', '.npy', '.csv', '.pem', '.key'}
IGNORED = {'.git', '.venv', '__pycache__', 'third_party', 'data', 'cache',
           'runs', 'outputs', 'results', 'checkpoints', 'weights', 'verification'}


def main():
    problems = []
    count = 0
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            problems.append(f'Missing or invalid path: {name}')
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(f'Changed file: {name}')
        count += 1
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in IGNORED for p in path.relative_to(ROOT).parts):
            continue
        if path.suffix in DENIED:
            problems.append(f'Non-source artifact: {path.relative_to(ROOT)}')
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'))
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
    print(json.dumps({'checked_files': count, 'problems': problems}, indent=2))
    raise SystemExit(bool(problems))


if __name__ == '__main__':
    main()
