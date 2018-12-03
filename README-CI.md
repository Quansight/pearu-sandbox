
# CI

## Appveyor for MSVC builds

Follow steps 1-4 in https://www.appveyor.com/docs/:
Sign in to appveyor using Github account, find your github project and add, start New Build.

Create a .appveyor.yml file. Some examples:
- [xndtools/.appveyor.yml](https://github.com/plures/xndtools/blob/master/.appveyor.yml) 
  uses `git clone`, `pip install pytest`, `python setup.py install/develop/test`
  
## Travis CI for Linux builds

TODO

## SMTH for MacOSX builds

TODO
