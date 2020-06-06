#!/usr/bin/env python
"""Watch Markdown files and apply LaTeX hooks to img elements.

Requirements:

  - watchdoc - required
  - pandoc - required when using --html
  - latex - optional, used to fix the vertical alignment of img elements

Usage:

  python watch_latex_md.py --help
  python watch_latex_md.py /path/to/markdown/documents/ --html

Home:

  https://github.com/Quansight/pearu-sandbox/latex_in_markdown/
"""
# Author: Pearu Peterson
# Created: June 2020

import os
import re
import sys
import time
import urllib.request
import urllib.parse
import argparse
import hashlib
import tempfile
from watchdog.observers import Observer
from watchdog.events import RegexMatchingEventHandler
from subprocess import check_output
import xml.etree.ElementTree as ET


def latex2svg(string):
    doc = r'''
\documentclass[17pt]{article}
\usepackage{extsizes}
\usepackage{amsmath}
\usepackage{amssymb}
\pagestyle{empty}
\begin{document}
%s
\end{document}
''' % (string)
    name = hashlib.md5(string.encode('utf-8')).hexdigest()
    workdir = tempfile.mkdtemp('', 'watch-latex-md-')
    texfile = os.path.join(workdir, name + '.tex')
    dvifile = os.path.join(workdir, name + '.dvi')
    f = open(texfile, 'w')
    f.write(doc)
    f.close()
    try:
        check_output(['latex', '-output-directory=' + workdir,
                      '-interaction', 'nonstopmode', texfile],
                     stderr=sys.stdout)
    except Exception:
        print(f'failed to latex {string!r}')
        return ''
    svg = check_output(
        ['dvisvgm', '-v0', '-a', '-n', '-s', dvifile])
    return svg.decode('utf-8')


def get_svg_geometry(svg):
    if not svg:
        return dict()
    xml = ET.fromstring(svg)
    ns = xml.tag.rstrip('svg')
    width = float(xml.attrib['width'][:-2])
    height = float(xml.attrib['height'][:-2])
    viewBox = list(map(float, xml.attrib['viewBox'].split()))
    gfill = xml.find(ns + 'g')
    use = gfill.find(ns + 'use')
    y = float(use.attrib['y'])
    baseline = height - (y - viewBox[1])
    return dict(width=width, height=height, baseline=baseline, valign=-baseline)


def img_latex_repl(m):
    orig = m.string[m.start():m.end()]
    latex = m.group('latex').strip()

    # The `.` will define the base line:
    svg = latex2svg('.' + latex)
    geom = get_svg_geometry(svg)

    inline = False
    if latex[:2] in ('$$', r'\['):
        assert latex[-2:] in ('$$', r'\]'), (latex[:2], latex[-2:])
        formula = latex[2:-2].strip()
    elif latex[:1] == '$':
        assert latex[-1] == '$', (latex[-1],)
        inline = True
        formula = latex[1:-1].strip()
    elif latex.startswith(r'\begin{equation'):
        i = latex.find('}')
        j = latex.rfind('\end{equation')
        assert -1 not in [i,j], (i,j)
        formula = latex[i+1:j].strip()
    elif latex.startswith(r'\begin{'):
        formula = latex
    else:
        inline = True
        formula = r'\text{%s}' % (latex)

    return make_img(latex, formula, inline, geom)


def make_img(latex, formula, inline, geom):
    # The exclamation fixes codecogs issues with some formulas ending
    # with `.` or `)`.
    formula = formula + r'\!'
    attrs =  ''

    if inline:
        formula = r'\inline '+formula
        if 'width' in geom:
            attrs += ' width="{width:.4f}px" height="{height:4f}px"'.format_map(geom)        
        if 'valign' in geom:
            attrs += ' valign="{valign:.4f}px"'.format_map(geom)
        attrs += ' style="display:inline;"'
    else:
        latex = f'\n{latex}\n'
        attrs += ' style="display:block;50px:auto;margin-right:auto;padding:25px"'
    url = 'https://latex.codecogs.com/svg.latex?' + urllib.parse.quote(formula)
    img = f'<img data-latex="{latex}" src="{url}" {attrs} alt="latex">'
    return img


def symbolrepl(m):
    orig = m.string[m.start():m.end()]
    label = m.group('label')
    comment = '<!--:' + label + ':-->'
    label_map = dict(
        proposal=':large_blue_circle:',
        impl=':large_blue_diamond:',
    )
    return label_map.get(label, '') + comment


def latexrepl(m):
    """
    Replace plain LaTeX expressions with empty latex images.
    """
    orig = m.string[m.start():m.end()]
    formula = m.group('formula').strip()
    dollars = m.string[m.start():m.start('formula')]
    is_latex_comment = m.string[:m.start()].endswith('<!--')
    if is_latex_comment:
        return orig
    if m.string[:m.start()].rstrip().endswith('"'):
        return orig
    return make_img(orig.strip(), 'a', False, {})    


def formularepl(m):
    """
    Append LaTeX rendering images to LaTeX comments.
    """
    orig = m.string[m.start():m.end()]
    formula = m.group('formula').strip()
    dollars = m.string[m.start('dollars'):m.end('dollars')]
    dollars = dollars[:2]
    latex = dollars + formula + dollars
    inline = len(dollars) == 1
    return make_img(latex, formula, inline, geom)


def str2re(s):
    """Make re specific characters immune to re.compile.
    """
    for c in '.()*+?$^\\':
        s = s.replace(c, '['+c+']')
    return s


header = '''<!--watch-latex-md

This document is processed by watch_latex_md.py program, see

  https://github.com/Quansight/pearu-sandbox/latex_in_markdown/

You can edit this document as you wish. You can also edit the LaTeX
data in img elements but note:

  1. to automatically update the LaTeX rendering in img element, edit
     the file under the supervision of watch_latex_md.py

  2. don't change the beginning (`<img latex-data="...`) and the end
     (`...alt="latex">`) parts of the img element as these are used by
     the watch_latex_md.py script.
-->
'''

class MarkDownLaTeXHandler(RegexMatchingEventHandler):

    def __init__(self, pattern, run_pandoc=False, verbose=False):
        super(MarkDownLaTeXHandler, self).__init__(regexes=[pattern])
        self.run_pandoc = run_pandoc
        self.verbose = verbose
        self.last_modified = {}

    def on_modified(self, event):
        if event.is_directory:
            return
        now = time.time()
        t = self.last_modified.get(event.src_path)
        if t is not None and abs(now - t) < 0.1:
            if self.verbose:
                print(f'Skip just processed {event.src_path} [now - time={now-t:.3f} sec]')
            return
        self.update_md(event.src_path)
        self.last_modified[event.src_path] = time.time()

    def update_md(self, path):
        if self.verbose:
            print(f'Processing {path}')
        filename, ext = os.path.splitext(path)
        assert ext == '.md', (path, ext)

        content = open(path).read()

        count = 0
        for pattern, repl in [
                (r'[$]+(?P<formula>[^$]+)[$]+', latexrepl),
                #(r'[<][!][-][-](?P<dollars>[$]+)(?P<formula>[^$]*)[$]+\s*[-][-][>](?P<prev>\s*[<]img\s+src[=]["].*?[>])?', formularepl),
                #(r'(?P<prev>[:][^:]+[:]\s*)?[<][!][-][-][:](?P<label>.*?)[:][-][-][>]', symbolrepl),
                (r'([<]img\s+data[-]latex="\s*)(?P<latex>.*?)"\s+src=.*?\s+alt="latex">', img_latex_repl),
        ]:
            content, _count = re.subn(
                pattern, repl,
                content,
                flags=re.S | re.M
            )
            count += _count

        if count > 0:
            if content.find(header.split(None, 1)[0]) == -1:
                content = header + '\n' + content
            if self.verbose:
                print(f'Updating {path} (found {count} replacements)')
            f = open(path, 'w')
            f.write(content)
            f.close()

            if self.run_pandoc:
                path_html = path + '.html'
                if self.verbose:
                    print(f'Generating {path_html}')
                cmd = f'pandoc -f gfm -t html --metadata title="{os.path.basename(os.path.dirname(path))}" -s {path} -o {path_html}'
                status = os.system(cmd)
                if status:
                    print(f'`{cmd}` FAILED[status={status}]')
        else:
            if self.verbose:
                print(f'No updates to {path}')


def main():

    parser = argparse.ArgumentParser(description='Watch Markdown files and apply LaTeX processing hooks.')
    parser.add_argument('paths', metavar='paths', type=str, nargs='*',
                        default=['.'],
                        help='Paths to the location of .md files. Default is current working director.')
    parser.add_argument('--html', dest='html',
                        action='store_const', const=True,
                        default=False,
                        help='Generate HTML files, requires pandoc. Default is False.')
    parser.add_argument('--verbose', dest='verbose',
                        action='store_const', const=True,
                        default=False,
                        help='Be verbose, useful for debugging. Default is False.')

    args = parser.parse_args()
    print(args)

    observer = Observer()

    for path in args.paths:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            pattern = '.*[.]md$'
            recursive = True
        elif os.path.isfile(path):
            path, filename = os.path.split(path)
            pattern = str2re(filename) + '$'
            recursive = False
        else:
            print(f'Path {path} does not exist. Skipping.')
            continue
        event_handler = MarkDownLaTeXHandler(pattern, verbose=args.verbose,
                                             run_pandoc=args.html)
        observer.schedule(event_handler, path, recursive=recursive)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    
if __name__ == '__main__':
    main()
