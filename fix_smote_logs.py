import glob
from pathlib import Path

files = Path('training/low_birth_weight').glob('*.py') 

for f in files:
    if 'prepare' in f.name: continue
    content = f.read_text(encoding='utf-8')
    content = content.replace("print(f'\nAFTER", "print(f'\\nAFTER")
    content = content.replace("(1:1)\n')", "(1:1)\\n')")
    f.write_text(content, encoding='utf-8')
    print(f'Fixed {f.name}')
