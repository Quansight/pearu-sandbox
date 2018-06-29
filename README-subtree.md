# Introduction

For development that might involve changing external libraries, I'll use git subtree. See [git subtrees: a tutorial](https://medium.com/@v/git-subtrees-a-tutorial-6ff568381844)

## libgdf

```

git remote add -f goai_libgdf https://github.com/gpuopenanalytics/libgdf.git
git subtree add -P libgdf goai_libgdf master
```
