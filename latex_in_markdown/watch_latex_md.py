#!/usr/bin/env python
"""
Watch Markdown files and apply LaTeX processing hooks.

Install requirements:

  conda install -c conda-forge watchdoc pandoc

Usage:

  python watch_latex_md.py /path/to/markdown/directory/
"""
# Author: Pearu Peterson
# Created: June 2020

import os
import re
import sys
import time
import urllib.parse
from watchdog.observers import Observer
from watchdog.events import RegexMatchingEventHandler


def symbolrepl(m):
    orig = m.string[m.start():m.end()]
    label = m.group('label')
    comment = '<!--:' + label + ':-->'
    label_map = dict(
        proposal=':large_blue_circle:',
        impl=':large_blue_diamond:',
    )
    return label_map.get(label, '') + comment
    if label == 'proposal':
        return ':large_blue_circle:' + comment
    return orig


def latexrepl(m):
    """
    Replace LaTeX formulas with LaTeX comments.
    """
    orig = m.string[m.start():m.end()]
    formula = m.group('formula').strip()
    dollars = m.string[m.start():m.start('formula')]
    is_latex_comment = m.string[:m.start()].endswith('<!--')
    if is_latex_comment:
        return orig
    return '<!--' + orig + '-->'


def formularepl(m):
    """
    Append LaTeX rendering images to LaTeX comments.
    """
    orig = m.string[m.start():m.end()]
    formula = m.group('formula').strip()
    dollars = m.string[m.start('dollars'):m.end('dollars')]

    print('formularepl:', formula, dollars)
    dollars = dollars[:2]

    inline = len(dollars) == 1
    
    comment = '<!--' + dollars + formula + dollars + '-->'

    if inline:
        formula = '\\inline ' + formula
    formula = urllib.parse.quote(formula)

    # img = '<img src="https://render.githubusercontent.com/render/math?math=' + formula + '" title="' + formula + '">'

    if 1:
        img = '<img src="https://latex.codecogs.com/svg.latex?' + formula + '">'
    
    return comment + img


class MarkDownLaTeXHandler(RegexMatchingEventHandler):
    
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path in self.update_cache:
            self.update_cache.remove(event.src_path)
            return
        self.update_cache.add(event.src_path)
        self.update_md(event.src_path)


    update_cache = set()

    def update_md(self, path):
        print(f'Process {path} [timestamp={time.time()}]')
        filename, ext = os.path.splitext(path)
        assert ext == '.md', (path, ext)

        content = open(path).read()

        count = 0
        for pattern, repl in [
                (r'[$]+(?P<formula>[^$]+)[$]+', latexrepl),
                (r'[<][!][-][-](?P<dollars>[$]+)(?P<formula>[^$]*)[$]+\s*[-][-][>](?P<prev>\s*[<]img\s+src[=]["].*?[>])?', formularepl),
                (r'(?P<prev>[:][^:]+[:]\s*)?[<][!][-][-][:](?P<label>.*?)[:][-][-][>]', symbolrepl),
        ]:

            content, _count = re.subn(
                pattern, repl,
                content,
                flags=re.S | re.M
            )
            count += _count

        if count > 0:
            print(f'Updating {path}.')
            f = open(path, 'w')
            f.write(content)
            f.close()
        else:
            print(f'No updates in {path}.')

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    event_handler = MarkDownLaTeXHandler(regexes=[r'.*[.]md$'])
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    main()
