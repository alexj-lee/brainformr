"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
#src = root / "src"
src = root / "brainformr"
print('hi', src)

for path in sorted(src.rglob("*.py")):
    print(path)

    if path.name == '_version.py' or 'brainformr/__init__.py' in path.as_posix():
        continue

    if path.name == 'calculations.py' or path.name == 'car.py':
        continue
    
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)
    print('binary search', full_doc_path)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        # added these after

        doc_path = doc_path.with_name('index.md')
        full_doc_path = full_doc_path.with_name('index.md')
        print('after', full_doc_path)

    elif parts[-1] == "__main__":
        continue

    print('yes', doc_path)
    nav[parts] = doc_path.as_posix()  

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
    #mkdocs_gen_files.set_edit_path(full_doc_path, Path('../' / path ))
print('\n'*3)
print(list(nav.items()))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  
    nav_file.writelines(nav.build_literate_nav())