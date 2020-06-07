<!--watch-latex-md

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


# Using LaTeX in Github Flavored Markdown documents

For writting technical documents in GitHub Flavored MarkDown (GFM)
documents that involve mathematics, using LaTeX formulas is a highly
desired feature. This project provides the following solution:

- Download [watch_latex_md.py](watch_latex_md.py) script and make sure
  you have pdflatex, watchdog, and optionally pandoc installed.
  
- Run:
  ```
  python /path/to/watch_latex_md.py --html --git
  ```

  in a directory that contains Markdown documents (files with
  extension `.md`). The `watch_latex_md.py` script will process
  Markdown documents whenever the files are modified in the
  filesystem.

- When a LaTeX expression surrounded with dollar (&#0036;) signs, is
  inserted to a Markdown document and the document is save, the
  `watch_latex_md.py` will replace such LaTeX expressions with HMTL
  `img` element containing a link to LaTeX rendering result (it's a
  SVG file created under the directory of the Markdown document).
  The HTML `img` element has a `data-latex` field that contains the
  original LaTeX expression which can be edited at any time. Whenever
  the Markdown document is saved, the script will automatically
  rerender the LaTeX expression.

- Single dollar signs around the LaTeX expressions will lead to
  inlined HTML `img` while double dollars (or `\[`-`\]` pair) result a
  slighly large image. When no dollar signs are used, the LaTeX
  expression is rendered for inline mode, except when the expression
  contains a LaTeX environment surrounded with
  `\begin{...}`-`\end{...}` block.

- For example, when having the following text in a Markdown document:

  - `Let `&#0036;`a\in\mathbb{R}`&#0036;`, then `&#0036;`\sqrt{a}`&#0036; `is imaginary whenever` &#0036;a<0&#0036;`.` 

  then after saving the document, the `watch_latex_md.py` process will
  update the document with the following content:
  ```
  Let <img data-latex="$a\in\mathbb{R}$" src=".images/eefa3ffc21ccf3c8d25ed9f9c8d2019d.svg"  valign="-0.673px" width="46.621px" height="12.533px" style="display:inline;" alt="latex">, then <img data-latex="$\sqrt{a}$" src=".images/5ba233a286631a8918b5ae921f1f3286.svg"  valign="-4.127px" width="27.493px" height="17.215px" style="display:inline;" alt="latex"> is imaginary whenever <img data-latex="$a<0$" src=".images/7b62b3f1ca6c1ec6d8f17ca446ec7f64.svg"  valign="-0.459px" width="43.717px" height="11.556px" style="display:inline;" alt="latex">. 
  ```
  that renders in a GitHub repository as follows:

  - Let <img data-latex="$a\in\mathbb{R}$" src=".images/eefa3ffc21ccf3c8d25ed9f9c8d2019d.svg"  valign="-0.673px" width="46.621px" height="12.533px" style="display:inline;" alt="latex">, then <img data-latex="$\sqrt{a}$" src=".images/5ba233a286631a8918b5ae921f1f3286.svg"  valign="-4.127px" width="27.493px" height="17.215px" style="display:inline;" alt="latex"> is imaginary whenever <img data-latex="$a<0$" src=".images/7b62b3f1ca6c1ec6d8f17ca446ec7f64.svg"  valign="-0.459px" width="43.717px" height="11.556px" style="display:inline;" alt="latex">. 


- Here's an example of displayed LaTeX expression:

  &#0036;&#0036;
  ```
  A = \int_a^b f(x)\,dx
  ```
  &#0036;&#0036;

  renders as

    <img data-latex="
$$
A = \int_a^b f(x)\,dx
$$
" src=".images/4e4ab61f8250368f2d01a961f9b92d11.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

- The Markdown document writer should not pay attention to the content
  of the HTML `img` element (I know, it looks ugly..) other than the
  `data-latex` field. This field can be updated by the writer at any
  time, all other fields are overwritten when saving the Markdown
  document.

Similar projects:

- [readme2tex](https://github.com/leegao/readme2tex), see
  [comment](https://github.com/Quansight/pearu-sandbox/pull/16#issuecomment-639463851). Update:
  originally, the `watch_latex_md.py` script used online LaTeX
  rendering service such as [codecogs.com](https://codecogs.com) but
  that turned out to be unreliable: it worked only locally and it
  often failed when viewing Markdown documents from a GitHub
  repository. As a result, now we use basically the same rendering
  method as used in [readme2tex](https://github.com/leegao/readme2tex)
  project.

Some hints:

- See the example Markdown document:
  [basic_latex_template.md](basic_latex_template.md).

- The `--html` option will generate a html document so that the output
  could be checked locally before committing the changes to a GitHub
  repository.

- The `--git` option will automatically add new rendering results
  under the git control as well as remove obsolete results. So, when
  done editing Markdown documents, you can simply proceed with `git`
  commit and push commands.

- The `--force-rerender` option will force rendering of all LaTeX
  expressions in Markdown documents. This is useful when debugging, or
  when some rendering results got lost/broken, or when one wishes to
  add these under git control.

- Emacs users may want to enable `auto-revert-mode` so that the buffer
  of a Markdown document will be automatically updated when the
  `watch_latex_md.py` process has updated the document.

Developers hints:
```
cd ~/git/Quansight/pearu-sandbox/latex_in_markdown
ls *.py | entr -r python watch_latex_md.py  --verbose --html --force-rerender # --git
```
