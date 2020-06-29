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
        if latex[-2:] == r'\]':
            expr = latex[2:-2].strip()
        else:
            expr = latex[2:].rstrip('$')
    elif latex[:1] == '$':
        assert latex[-1] == '$', (latex[-1],)
        inline = True
        expr = latex[1:].rstrip('$')
    elif latex.startswith(r'\begin{equation'):
        i = latex.find('}')
        j = latex.rfind(r'\end{equation')
        assert -1 not in [i, j], (i, j)
        expr = latex[i+1:j].strip()
    elif latex.startswith(r'\begin{'):
        expr = latex
    else:
        inline = True
        if latex.startswith(r'\text{'):
            assert latex.endswith('}'), (latex,)
            expr = latex
        else:
            expr = r'\text{%s}' % (latex)
    return inline, expr


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

watch-latex-md:no-force-rerender
-->
'''

image_dir_gitattributes = '''
*.svg binary linguist-generated
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
        self.image_prefix = '.images'
        self.image_dir = os.path.join(self.working_dir, self.image_prefix)

        self._last_modified = 0
        self._use_git = None
        self._force_rerender = None

        self.image_files = set()

        gitattrs_file = os.path.join(self.image_dir, '.gitattributes')
        if not os.path.isdir(self.image_dir):
            os.makedirs(self.image_dir)

        if self.use_git and not os.path.isfile(gitattrs_file):
            if self.verbose:
                print(f'{gitattrs_file} created')
            f = open(gitattrs_file, 'w')
            f.write(image_dir_gitattributes)
            f.close()

            if not self.git_check_added(gitattrs_file):
                self.git_add_file(gitattrs_file)

        self.git_update_init()

    @property
    def verbose(self):
        return self.parent.verbose

    @property
    def force_rerender(self):
        return self._force_rerender or self.parent.force_rerender

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
            # the first use should correspond to the dot when in
            # inline mode
            use = gfill.find(ns + 'use')
            if use is None:
                baseline = 0
            else:
                y = float(use.attrib['y'])
                baseline = height - (y - viewBox[1])
                if inline:
                    gfill.remove(use)
            xml.set('watch_lated_md:baseline', str(baseline))
            xml.set('xmlns:watch_lated_md', myns)
            svg = ET.tostring(xml).decode('utf-8')
        else:
            baseline = float(baseline)
        params = dict(width=round(width, 3), height=round(height, 3),
                      baseline=round(baseline, 3), valign=round(-baseline, 3))
        return svg, params

    def load_svg(self, svg_file):
        f = open(svg_file)
        svg = f.read()
        f.close()
        return self._return_svg(svg)

    def get_svg(self, latex, hexname, inline):
        doc = r'''
\documentclass[17pt,leqno]{article}
\usepackage{extsizes}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[noend]{algpseudocode}
\pagestyle{empty}
\newcommand{\inlinemath}[1]{$#1$}
\begin{document}
\setcounter{equation}{%s}
%s%s
\end{document}
''' % (self.equation_counter - 1, ('.' if inline else ''), latex)

        tex_file = os.path.join(self.temp_dir, hexname + '.tex')
        dvi_file = os.path.join(self.temp_dir, hexname + '.dvi')
        f = open(tex_file, 'w')
        f.write(doc)
        f.close()
        try:
            result = subprocess.run(['latex',
                                     '-output-directory=' + self.temp_dir,
                                     '-interaction', 'nonstopmode', tex_file],
                                    capture_output=True)
        except Exception as msg:
            print(f'FAILED TO LATEX {latex!r}: {msg}')
            return '', {}
        if result.returncode:
            if inline:
                print(f'FAILED TO LATEX: {latex}')
            else:
                print(f'FAILED TO LATEX:\n{"":-^70}\n{latex}\n{"":-^70}\n')
            print(result.stdout.decode('utf-8'))
            print(result.stderr.decode('utf-8'))
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
        attrs = ''
        if params.get('inline', False):
            for a in ['valign', 'width', 'height']:
                if a in params:
                    v = params[a]
                    if abs(v) > 1e-3:
                        attrs += ' {}="{}px"'.format(a, v)
            attrs += ' style="display:inline;"'
        else:
            attrs += (' style="display:block;margin-left:50px;'
                      'margin-right:auto;padding:0px"')
            latex = f'\n{latex}\n'
        for label in params.get('labels', []):
            attrs += f' id="{label}"'
        return f'<img data-latex="{latex}" src="{src}" {attrs} alt="latex">'

    def img_to_svg(self, m):
        latex = m.group('latex').strip()
        inline, expr = get_latex_expr(latex)
        if inline:
            latex = latex.replace('\n', ' ')
            orig_latex = latex
            labels = []
        else:
            orig_latex = latex
            labels = re.findall(r'\\label[{]([^}]+)[}]', latex)
            if labels:
                self.equation_counter += 1
                for label in labels:
                    if label in self.label_counter:
                        print(f'label already used: resetting its equation'
                              f' counter to {self.equation_counter} '
                              f'(was {self.label_counter[label]})')
                    self.label_counter[label] = self.equation_counter

        if self.verbose:
            print(f'img_to_svg: {inline=} {expr=}')

        hexname = hashlib.md5((f'{self.filename}:{latex}')
                              .encode('utf-8')).hexdigest()
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
        params.update(inline=inline, labels=labels)
        self.image_files.add(svg_file)
        return self.make_img(orig_latex, svg_src, **params)

    def update_label_title(self, m):
        orig = m.string[m.start():m.end()]
        title = m.group('title')
        label = m.group('label')
        number = self.label_counter.get(label)
        if number is None:
            if label.lower().startswith('eq'):
                print(f'no equation number found for `[{title}](#{label})`')
                print(f'  available labels: {list(self.label_counter)}')
            return orig
        title = re.sub(r'\d+', str(number), title)
        return f'[{title}](#{label})'

    def update(self):
        now = time.time()
        if now - self._last_modified < 1:
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

        if not orig_content:
            if self.verbose:
                print('Got empty content. Sleeping 1 sec and try again.')
            time.sleep(1)
            f = open(self.md_file)
            orig_content = f.read()
            f.close()

        if not orig_content:
            print('Empty file. Skip update!')
            return

        content = orig_content

        if content.find(header.split(None, 1)[0]) == -1:
            content = header + '\n' + content
            if not content.rstrip().endswith('<!--EOF-->'):
                content += '\n<!--EOF-->'

        elif not content.rstrip().endswith('<!--EOF-->'):
            print('Expected <!--EOF--> at the end of file. Skip update!')
            return

        if self.parent.run_pandoc:
            f = open(self.html_file, 'w')
            f.write('''
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
 "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta http-equiv="Content-Style-Type" content="text/css" />
  <meta http-equiv="refresh" content="0.5" />
  <meta name="generator" content="watch_latex_md.py" />
  <title>Wait...</title>
  <style type="text/css">code{white-space: pre;}</style>
</head>
<body>
<h1>Please wait until watch_latex_md.py updates %s...</h1>
<h2>The page will reload automatically.</h2>
</body>
</html>
            ''' % (self.html_file))
            f.close()

        if 'watch-latex-md:force-rerender' in content:
            print('Setting force-rerender')
            self._force_rerender = True

        self.equation_counter = 0
        self.label_counter = {}
        for pattern, repl in [
                (r'(?m)(?P<latex>[$]+[^$]+[$]+)', self.latex_to_img),
                ((r'(?m)([<]img\s+data[-]latex=["]\s*)(?P<latex>.*?)'
                  r'["]\s+src=.*?\s+alt="latex">'),
                 self.img_to_svg),
                ((r'(\[(?P<title>[^]]+)\][(][#](?P<label>[^)]+)[)])'),
                 self.update_label_title)]:
            content, _count = re.subn(
                pattern, repl,
                content,
                flags=re.S
            )

        if self.force_rerender or content != orig_content:
            f = open(self.md_file, 'w')
            f.write(content)
            f.close()
            self._last_modified = time.time()
            if self.verbose:
                print(f'{self.md_file} is updated')
        else:
            if self.verbose:
                print(f'{self.md_file} is up-to-date')

        self._force_rerender = False

        if self.parent.run_pandoc:
            try:
                check_output(
                    ['pandoc',
                     '-f', 'gfm',
                     '-t', 'html',
                     '--metadata',
                     'title=' + os.path.basename(self.md_file),
                     '-s', self.md_file,
                     '-o', self.html_file],
                    stderr=sys.stdout)
            except Exception as msg:
                print(f'{self.md_file} pandoc failed: {msg}')
            else:
                if self.verbose:
                    print(f'{self.html_file} is generated')

        new_files = list(self.image_files.difference(prev_image_files))
        obsolete_files = list(prev_image_files.difference(self.image_files))

        if self.use_git:
            for fn in new_files:
                if not self.git_check_added(fn):
                    self.git_add_file(fn)
            for fn in obsolete_files:
                if self.git_check_added(fn):
                    self.git_rm_file(fn)
                else:
                    self.rm_file(fn)
        else:
            if self.git_check_repo():
                for fn in obsolete_files:
                    if self.git_check_added(fn):
                        print(f'{fn} is obsolete in git repo '
                              '[use --git to enable auto-removal]')
                    else:
                        self.rm_file(fn)
            else:
                for fn in obsolete_files:
                    self.rm_file(fn)

    def git_update_init(self):
        f = open(self.md_file)
        content = f.read()
        f.close()

        for fn in re.findall(
                os.path.join(str2re(self.image_prefix),
                             r'\w+.svg'), content):
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
            g = ImageGenerator(self, event.src_path)
            self.image_generators[event.src_path] = g
        g.update()


def main():

    parser = argparse.ArgumentParser(
        description='Watch Markdown files and apply LaTeX processing hooks.')
    parser.add_argument(
        'paths', metavar='paths', type=str, nargs='*',
        default=['.'],
        help=('Paths to the location of .md files.'
              ' Default is current working director.'))
    parser.add_argument(
        '--html', dest='html',
        action='store_const', const=True,
        default=False,
        help='Generate HTML files, requires pandoc. Default is False.')
    parser.add_argument(
        '--git', dest='use_git',
        action='store_const', const=True,
        default=False,
        help=('Add generated image files automatically to git.'
              ' Default is False.'))
    parser.add_argument(
        '--force-rerender', dest='force_rerender',
        action='store_const', const=True,
        default=False,
        help='Always rerender, useful for debugging. Default is False.')
    parser.add_argument(
        '--verbose', dest='verbose',
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
        event_handler = MarkDownLaTeXHandler(
            pattern,
            verbose=args.verbose,
            use_git=args.use_git,
            run_pandoc=args.html,
            force_rerender=args.force_rerender)
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
