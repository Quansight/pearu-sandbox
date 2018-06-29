# Introduction

For development that might involve changing external libraries, I'll use git subtree. See [git subtrees: a tutorial](https://medium.com/@v/git-subtrees-a-tutorial-6ff568381844)

## libgdf

```
git remote add -f goai_libgdf https://github.com/gpuopenanalytics/libgdf.git
git subtree add -P libgdf goai_libgdf master
cat libgdf/.gitmodules >> .gitmodules
nano .gitmodules # prepend libgdf/ to thirdparty
git submodule init 
git submodule update --remote --merge
git add -u .
git commit -m "add libgdf submodules"
git push

# to update already cloned pearu-sandbox:
git pull
git submodule init
git submodule update --recursive --remote

# to clone pearu-sandbox:
git clone --recurse-submodules  git@github.com:Quansight/pearu-sandbox.git
```

## pygdf

```
git pull
git remote add -f goai_pygdf https://github.com/gpuopenanalytics/pygdf.git
git subtree add -P pygdf goai_pygdf master
git push
```
