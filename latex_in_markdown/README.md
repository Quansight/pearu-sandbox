<!--watch-latex-md

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


# Using LaTeX in Github Flavored Markdown documents

For writting technical documents in GitHub Flavored MarkDown (GFM)
documents that involve mathematics, inserting LaTeX formulas is highly
desired prerequisite. This project provides the following solution:

- Download [watch_latex_md.py](watch_latex_md.py) script and install
  watchdog, and optionally, pandoc and latex.
  
- Run:
  ```
  python watch_latex_md.py --html
  ```
  
  in a directory that contains Markdown documents (files with
  extension `.md`). The `watch_latex_md.py` script will process
  Markdown documents whenever the files are modified in the
  filesystem. Kill the process when not needed anymore.

- To have a LaTeX expression in a Markdown document, insert the latex
  expression surrounded with dollar (&#0036;) signs (single dollar signs
  lead to LaTeX expressions displayed inline, double dollar signs
  display the LaTeX as a slightly-left-centered block). When saving the
  document, the `watch_latex_md.py` process will replace such LaTeX
  expressions with HTML `img` element that contains a link to LaTeX
  rendering result (the script uses [codecogs.com](codecogs.com) LaTeX
  rendering service). The HTML `img` element has a `data-latex` field
  that contains the original LaTeX expression and that can be
  edited. Whenever the Markdown document is saved, the script will
  updated the link of the LaTeX rendering result.

  For example, when having the following text in a Markdown document:

  - `Let `&#0036;`a\in\mathbb{R}`&#0036;`, then `&#0036;`\sqrt{a}`&#0036; `is imaginary whenever` &#0036;a<0&#0036;`.` 

  then after saving the document, the `watch_latex_md.py` process will
  update the document with the following content:
  ```
  Let <img data-latex="$a\in\mathbb{R}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20a%5Cin%5Cmathbb%7BR%7D%5C%21"  width="46.6209px" height="12.532600px" valign="-0.6731px" style="display:inline;" alt="latex">, then <img data-latex="$\sqrt{a}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csqrt%7Ba%7D%5C%21"  width="27.4927px" height="17.215400px" valign="-4.1269px" style="display:inline;" alt="latex"> is imaginary whenever <img data-latex="$a<0$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20a%3C0%5C%21"  width="43.7165px" height="11.556400px" valign="-0.4591px" style="display:inline;" alt="latex">. 
  ```
  that renders in a GitHub repository as follows:

  - Let <img data-latex="$a\in\mathbb{R}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20a%5Cin%5Cmathbb%7BR%7D%5C%21"  width="46.6209px" height="12.532600px" valign="-0.6731px" style="display:inline;" alt="latex">, then <img data-latex="$\sqrt{a}$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csqrt%7Ba%7D%5C%21"  width="27.4927px" height="17.215400px" valign="-4.1269px" style="display:inline;" alt="latex"> is imaginary whenever <img data-latex="$a<0$" src="https://latex.codecogs.com/svg.latex?%5Cinline%20a%3C0%5C%21"  width="43.7165px" height="11.556400px" valign="-0.4591px" style="display:inline;" alt="latex">. 

  The Markdown document writer should not pay attention to the content
  of the HTML `img` element (I know, it looks ugly..) other than the
  `data-latex` field. This field can be updated by the writer at any
  time but all other fields will be overwritten when saving the
  Markdown document.

Similar projects:

- [readme2tex](https://github.com/leegao/readme2tex), see [comment](https://github.com/Quansight/pearu-sandbox/pull/16#issuecomment-639463851).

Some hints:

- Example Markdown document: [basic_latex_template.md](basic_latex_template.md).

- The `--html` option will generate from a Markdown document the
  corresponding html document so that the output could be checked
  locally before committing the changes to Markdown document to the
  GitHub repository.

- Usually, the dollar signs in `data-latex` will determine the display
  view (inline or block) of LaTeX expressions. When the dollar signs
  are removed, the LaTeX expression in the `data-latex` field will be
  wrapped with `\text{...}`, except when the LaTeX expression starts
  with some `\begin{...}` statement.

- Emacs users may want to enable `auto-revert-mode` so that the buffer
  of a Markdown document will be automatically updated when the
  `watch_latex_md.py` process has updated the document.

Developers hints:
```
cd ~/git/Quansight/pearu-sandbox/latex_in_markdown
ls *.py | entr -r python watch_latex_md.py  --verbose --html
```
