
# Using LaTeX in Github Flavored Markdown document

For writting technical documents in GitHub Flavored MarkDown (GFM)
documents that involve mathematics, inserting LaTeX formulas is highly
desired prerequisite. This document describes a solution that uses the
following approach:

- Run `ls *.md | entr -s `
- Use `$\textrm{this is \LaTeX}^{formula}$` in a GFM document.
- When saving the document, 
