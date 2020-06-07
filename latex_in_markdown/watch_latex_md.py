#!/usr/bin/env python
"""Watch Markdown files and apply LaTeX hooks to img elements.

Requirements:

  - watchdoc - required
  - pandoc - required when using --html
  - git - required when using --git
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
import subprocess
import xml.etree.ElementTree as ET

myns = 'https://github.com/Quansight/pearu-sandbox/latex_in_markdown/'

def get_latex_expr(latex):
    inline = False
    if latex[:2] in ('$$', r'\['):
        assert latex[-2:] in ('$$', r'\]'), (latex[:2], latex[-2:])
        expr = latex[2:-2].strip()
    elif latex[:1] == '$':
        assert latex[-1] == '$', (latex[-1],)
        inline = True
        expr = latex[1:-1].strip()
    elif latex.startswith(r'\begin{equation'):
        i = latex.find('}')
        j = latex.rfind('\end{equation')
        assert -1 not in [i,j], (i,j)
        expr = latex[i+1:j].strip()
    elif latex.startswith(r'\begin{'):
        expr= latex
    else:
        inline = True
        if latex.startswith(r'\text{'):
            assert latex.endswith('}'), (latex,)
            expr = latex
        else:
            expr = r'\text{%s}' % (latex)
    return inline, expr

def symbolrepl(m):
    orig = m.string[m.start():m.end()]
    label = m.group('label')
    comment = '<!--:' + label + ':-->'
    label_map = dict(
        proposal=':large_blue_circle:',
        impl=':large_blue_diamond:',
    )
    return label_map.get(label, '') + comment

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
data in img elements, but only the content of `latex-data`:

  1. To automatically update the LaTeX rendering in img element, edit
     the file while watch_latex_md.py is running.

  2. Never change the beginning (`<img latex-data="...`) and the end
     (`...alt="latex">`) parts of the LaTeX img elements as these are
     used by the watch_latex_md.py script.

  3. Changes to other parts of the LaTeX img elements will be
     overwritten.

Enjoy LaTeXing!
-->
'''

class ImageGenerator(object):

    def __init__(self, parent, md_file):
        self.parent = parent
        fn, ext = os.path.splitext(md_file)
        assert ext == '.md', (fn, ext)
        self.md_file = md_file
        self.html_file = md_file + '.html'
        self.filename = os.path.basename(fn)

        self.working_dir = os.path.dirname(md_file)
        self.temp_dir = tempfile.mkdtemp('', 'watch-latex-md-')
        self.image_prefix = '.watch-latex-md-images'
        self.image_dir = os.path.join(self.working_dir, self.image_prefix)

        if not os.path.isdir(self.image_dir):
            os.makedirs(self.image_dir)

        self._last_modified = 0
        self._use_git = None
        self.image_files = set()

        self.git_update_init()

    @property
    def verbose(self):
        return self.parent.verbose

    @property
    def force_rerender(self):
        return self.parent.force_rerender

    @property
    def use_git(self):
        """
        Use git only when the Markdown document is under git control.
        """
        if self._use_git is not None:
            return self._use_git
        if not self.parent.use_git:
            self._use_git = False
        elif not self.git_check_repo():
            self._use_git = False
        elif self.git_check_added(self.md_file):
            self._use_git = True
        else:
            return False
        return self._use_git

    def _return_svg(self, svg, inline=None):
        xml = ET.fromstring(svg)
        ns = xml.tag.rstrip('svg')
        baseline = xml.attrib.get('{'+myns+'}baseline')        
        width = float(xml.attrib['width'][:-2])
        height = float(xml.attrib['height'][:-2])
        if baseline is None:
            viewBox = list(map(float, xml.attrib['viewBox'].split()))
            gfill = xml.find(ns + 'g')
            use = gfill.find(ns + 'use')  # that should correspond to the dot when in inline mode
            y = float(use.attrib['y'])
            baseline = height - (y - viewBox[1])
            if inline:
                gfill.remove(use)
            xml.set('watch_lated_md:baseline', str(baseline))
            xml.set('xmlns:watch_lated_md', myns)
            svg = ET.tostring(xml).decode('utf-8')
        else:
            baseline = float(baseline)
        params = dict(width=round(width, 3), height=round(height, 3), baseline=round(baseline, 3), valign=round(-baseline, 3))
        return svg, params        

    def load_svg(self, svg_file):
        f = open(svg_file)
        svg = f.read()
        f.close()
        return self._return_svg(svg)

    def get_svg(self, latex, hexname, inline):
        doc = r'''
\documentclass[17pt]{article}
\usepackage{extsizes}
\usepackage{amsmath}
\usepackage{amssymb}
\pagestyle{empty}
\begin{document}
%s%s
\end{document}
''' % (('.' if inline else ''), latex)

        tex_file = os.path.join(self.temp_dir, hexname + '.tex')
        dvi_file = os.path.join(self.temp_dir, hexname + '.dvi')
        f = open(tex_file, 'w')
        f.write(doc)
        f.close()
        try:
            check_output(['latex', '-output-directory=' + self.temp_dir,
                          '-interaction', 'nonstopmode', tex_file],
                         stderr=sys.stdout)
        except Exception:
            print(f'failed to latex {latex!r}')
            return '', {}
        svg = check_output(['dvisvgm', '-v0', '-a', '-n', '-s', '-R',
                            dvi_file]).decode('utf-8')
        return self._return_svg(svg, inline)

    def latex_to_img(self, m):
        if m.string[:m.start()].rstrip().endswith('"'):
            return m.string[m.start():m.end()]
        latex = m.group('latex')
        if self.verbose:
            print(f'latex_to_img: {latex=}')
        return f'<img data-latex="{latex}" src="to-be-generated" alt="latex">'

    def make_img(self, latex, src, **params):
        attrs =  ''
        if params.get('inline', False):
            for a in ['valign', 'width', 'height']:
                if a in params:
                    v = params[a]
                    if abs(v) > 1e-3:
                        attrs += ' {}="{}px"'.format(a, v)
            attrs += ' style="display:inline;"'
        else:
            attrs += ' style="display:block;margin-left:50px;margin-right:auto;padding:0px"'
            latex = f'\n{latex}\n'
        return f'<img data-latex="{latex}" src="{src}" {attrs} alt="latex">'

    def img_to_svg(self, m):
        latex = m.group('latex').strip()
        inline, expr = get_latex_expr(latex)
        if inline:
            latex = latex.replace('\n', ' ')

        if self.verbose:
            print(f'img_to_svg: {inline=} {expr=}')

        hexname = hashlib.md5((f'{self.filename}:{latex}').encode('utf-8')).hexdigest()
        svg_file = os.path.join(self.image_dir, hexname) + '.svg'
        svg_src = os.path.join(self.image_prefix, hexname) + '.svg'
        if self.force_rerender or not os.path.isfile(svg_file):
            svg, params = self.get_svg(latex, hexname, inline)
            if not svg:
                return m.string[m.start():m.end()]
            if self.verbose:
                print(f'{svg_file} created')
            f = open(svg_file, 'w')
            f.write(svg)
            f.close()
        else:
            if self.verbose:
                print(f'{svg_file} exists')
            svg, params = self.load_svg(svg_file)
        params.update(inline=inline)
        self.image_files.add(svg_file)
        return self.make_img(latex, svg_src, **params)

    def update(self):
        now = time.time()
        if now - self._last_modified < 0.1:
            if self.verbose:
                print(f'{self.md_file} is likely up-to-date')
            return

        if self.verbose:
            print(f'{self.md_file} processing')

        prev_image_files = self.image_files.copy()
        self.image_files.clear()

        f = open(self.md_file)
        orig_content = f.read()
        f.close()

        content = orig_content

        if content.find(header.split(None, 1)[0]) == -1:
            content = header + '\n' + content

        for pattern, repl in [
                (r'(?P<latex>[$]+[^$]+[$]+)', self.latex_to_img),
                (r'([<]img\s+data[-]latex=["]\s*)(?P<latex>.*?)["]\s+src=.*?\s+alt="latex">', self.img_to_svg)]:
            content, _count = re.subn(
                pattern, repl,
                content,
                flags=re.S | re.M
            )

        if self.force_rerender or content != orig_content:
            f = open(self.md_file, 'w')
            f.write(content)
            f.close()
            self._last_modified = time.time()

            if self.verbose:
                print(f'{self.md_file} is updated{("--force-rerender" if self.force_rerender else "")}')


        else:
            if self.verbose:
                print(f'{self.md_file} is up-to-date')

        if self.parent.run_pandoc:
            try:
                check_output(['pandoc',
                              '-f', 'gfm',
                              '-t', 'html',
                              '--metadata', 'title=' + os.path.basename(self.md_file),
                              '-s', self.md_file,
                              '-o', self.html_file],
                             stderr=sys.stdout)
            except Exception as msg:
                print(f'{self.md_file} pandoc failed: {msg}' )
            else:
                if self.verbose:
                    print(f'{self.html_file} is generated')

        existing_files = list(self.image_files.intersection(prev_image_files))
        new_files = list(self.image_files.difference(prev_image_files))
        obsolete_files = list(prev_image_files.difference(self.image_files))

        if self.verbose:
            print(f'existing_files={list(map(os.path.basename, existing_files))}')
            print(f'new_files={list(map(os.path.basename, new_files))}')
            print(f'obsolete_files={list(map(os.path.basename, obsolete_files))}')

        git_add_files = set()
        git_rm_files = set()
        rm_files = set()

        if self.use_git:
            for fn in new_files:
                if not self.git_check_added(fn):
                    git_add_files.add(fn)

            for fn in obsolete_files:
                if self.git_check_added(fn):
                    git_rm_files.add(fn)
                else:
                    rm_files.add(fn)
        else:
            if self.git_check_repo():
                for fn in obsolete_files:
                    if self.git_check_added(fn):
                        print(f'{fn} is obsolete in git repo [use --git to enable auto-removal]')
                    else:
                        rm_files.add(fn)
            else:
                rm_files.update(obsolete_files)

        list(map(self.git_add_file, git_add_files))
        list(map(self.git_rm_file, git_rm_files))
        list(map(self.rm_file, rm_files))

    def git_update_init(self):
        f = open(self.md_file)
        content = f.read()
        f.close()

        for fn in re.findall(os.path.join(str2re(self.image_prefix), r'\w+.svg'), content):
            svg_file = os.path.join(self.working_dir, fn)
            if os.path.isfile(svg_file):
                self.image_files.add(svg_file)
            else:
                print(f'{svg_file} does not exist?')

        git_add_files = set()
        if self.use_git:
            for fn in self.image_files:
                if not self.git_check_added(fn):
                    git_add_files.add(fn)

        list(map(self.git_add_file, git_add_files))

    def rm_file(self, path):
        if os.path.isfile(path):
            if self.verbose:
                print(f'rm {path}')
            os.remove(path)

    def git_add_file(self, path):
        if os.path.isfile(path):
            if self.verbose:
                print(f'git add {path}')
                devnull = None
            else:
                devnull = subprocess.DEVNULL
            return subprocess.call(['git', 'add', path],
                                   stdout=devnull,
                                   stderr=devnull,
                                   cwd=self.working_dir) == 0

    def git_rm_file(self, path):
        if os.path.isfile(path):
            if self.verbose:
                print(f'git rm -f {path}')
                devnull = None
            else:
                devnull = subprocess.DEVNULL
            return subprocess.call(['git', 'rm', '-f', path],
                                   stdout=devnull,
                                   stderr=devnull,
                                   cwd=self.working_dir) == 0

    def git_check_repo(self):
        """Check if working directory is under git control
        """
        if self.verbose:
            print('git rev-parse --is-inside-work-tree')
            devnull = None
        else:
            devnull = subprocess.DEVNULL
        return subprocess.call(['git', 'rev-parse', '--is-inside-work-tree'],
                               stdout=devnull,
                               stderr=devnull,
                               cwd=self.working_dir) == 0

        
    def git_check_added(self, path):
        """
        Check if path is under git control.
        """
        if self.verbose:
            print(f'git ls-files --error-unmatch {path}')
            devnull = None
        else:
            devnull = subprocess.DEVNULL
        return subprocess.call(['git', 'ls-files', '--error-unmatch', path],
                               stdout=devnull,
                               stderr=devnull,
                               cwd=self.working_dir) == 0
    
    
    
class MarkDownLaTeXHandler(RegexMatchingEventHandler):

    def __init__(self, pattern, run_pandoc=False, verbose=False, use_git=False,
                 force_rerender=False):
        super(MarkDownLaTeXHandler, self).__init__(regexes=[pattern])
        self.run_pandoc = run_pandoc
        self.verbose = verbose
        self.use_git = use_git
        self.force_rerender = force_rerender
        self.image_generators = {}

    def on_modified(self, event):
        g = self.image_generators.get(event.src_path)
        if g is None:
            g = self.image_generators[event.src_path] = ImageGenerator(self, event.src_path)
        g.update()


def main():

    parser = argparse.ArgumentParser(description='Watch Markdown files and apply LaTeX processing hooks.')
    parser.add_argument('paths', metavar='paths', type=str, nargs='*',
                        default=['.'],
                        help='Paths to the location of .md files. Default is current working director.')
    parser.add_argument('--html', dest='html',
                        action='store_const', const=True,
                        default=False,
                        help='Generate HTML files, requires pandoc. Default is False.')
    parser.add_argument('--git', dest='use_git',
                        action='store_const', const=True,
                        default=False,
                        help='Add generated image files automatically to git. Default is False.')
    parser.add_argument('--force-rerender', dest='force_rerender',
                        action='store_const', const=True,
                        default=False,
                        help='Always rerender, useful for debugging. Default is False.')
    parser.add_argument('--verbose', dest='verbose',
                        action='store_const', const=True,
                        default=False,
                        help='Be verbose, useful for debugging. Default is False.')

    args = parser.parse_args()
    if args.verbose:
        print(args)

    observer = Observer()
    print('Start watching the following files:')
    for path in args.paths:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            pattern = '.*[.]md$'
            recursive = True
            print(f'  recursively all .md files under {path}/')
        elif os.path.isfile(path):
            path, filename = os.path.split(path)
            pattern = str2re(filename) + '$'
            recursive = False
            print(f'  {path}/{filename}')
        else:
            print(f'{path} does not exist. Skip in watching.')
            continue
        event_handler = MarkDownLaTeXHandler(pattern,
                                             verbose=args.verbose,
                                             use_git=args.use_git,
                                             run_pandoc=args.html,
                                             force_rerender=args.force_rerender
        )
        observer.schedule(event_handler, path, recursive=recursive)
    print('Press Ctrl-C to stop...')
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    
if __name__ == '__main__':
    main()
