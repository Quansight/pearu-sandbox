# My github notes

## Git configuration

```
[user]
        email = pearu.peterson@gmail.com
        name = Pearu Peterson
[push]
        default = simple
[merge]
        ff = false
[pull]
        ff = only
```

## Workflow for Quansight

### Forking a project to Quansight

Using Apache Arrow as an example project.

...

### Cloning a Quansight project

```
git clone git@github.com:Quansight/arrow.git
cd arrow
git remote add upstream https://github.com/apache/arrow.git
git remote add Quansight https://github.com/Quansight/arrow.git
```

### Tracking the upstream in master

```
git checkout master
git fetch upstream
git reset --hard upstream/master
git push -f -u Quansight
```
or
```
git fetch upstream
git rebase upstream/master
```
or
```
git log --graph --decorate --pretty=oneline --abbrev-commit
git rebase -i <current upstream parent commit for newfeature branch>
# editor is openend, replace all `pick` with `squash` except the first one, close editor
# another editor is opened, remove all unnecessary commit lines
```

### Implementing a new feature

```
git branch newfeature
git checkout newfeature
# Implement feature
git push -u origin newfeature
```

### Tracking the upstream in a feature branch

Make sure master is even with upstream, see above.

```
git checkout newfeature
git pull # is this needed?
git rebase upstream/master
git push -u origin newfeature
git pull # is this needed?
```

See [Fix git “tip of your current branch is behind its remote counterpart”](https://codewithhugo.com/fix-git-failed-to-push-updates-were-rejected/). I found `git stash` not working as expected (and lost some commits). To prevent commit losses, I suggest `git diff > newfeature-tmp.patch`, do the reset, and then `git apply newfeature-tmp.patch`.

# References

[GitHub Standard Fork & Pull Request Workflow](https://gist.github.com/Chaser324/ce0505fbed06b947d962)

[Issues with Arrow workflow](https://github.com/apache/arrow/pull/2844#issuecomment-433603570)

[If you make a git mistake](https://ohshitgit.com/)
