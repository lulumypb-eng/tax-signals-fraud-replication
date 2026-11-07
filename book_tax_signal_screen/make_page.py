"""Builds index.html (the Book-Tax Signal Screen page) from template.html and scores_v3.json."""
import json
t = open("template.html").read()
d = json.dumps(json.load(open("scores_v3.json")), separators=(",", ":")).replace("</", "<\\/")
html = t.replace("__DATA__", d)
html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        + html.replace("</style>\n", "</style>\n</head><body>\n", 1).replace("*{box-sizing:border-box}", "*{box-sizing:border-box}\nbody{margin:0}", 1)
        + "\n</body></html>\n")
open("index.html", "w").write(html)
print("wrote index.html")
