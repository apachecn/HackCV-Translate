## git git git git git

原文链接：[git git git git git](http://caiustheory.com/git-git-git-git-git/)

  你是否曾在你的终端不小心进入太多的git，想找到一个有效的解决方法？我经常写了一个git然后离开一会，再回来写时又输入完整的git status 想查看它的状态。这会导致一个可爱的（烦人的）错误：

```
$ git git status
git: 'git' is not a git command. See 'git --help'.
```

什么是git。

我的初始想法是在我的$PATH中覆盖git的二进制，以此除去会导致报错的那些多余的git，这样的话我们最后运行的只是git status部分。一个简单的方法是使用git 的配置中的alias.*功能去扩展git使其变为一个shell 命令。

```
git config --global alias.git '!exec git'
```

将下列git配置加入到你的 .gitconfig 文件中

```
[alias]
  git = !exec git
```

然后你就可以git git 你想进行的内容。

```
$ git sha
cc9c642663c0b63fba3964297c13ce9b61209313

$ git git sha
cc9c642663c0b63fba3964297c13ce9b61209313

$ git git git git git git git git git git git git git git git git git git git git git git git git git git sha
cc9c642663c0b63fba3964297c13ce9b61209313
```

(git sha 是 git rev-parse HEAD 的一个别名。)

看看我的 [~/.gitconfig][https://github.com/caius/zshrc/blob/master/dotfiles/gitconfig] 文件里还有什么别的Git 的别名，并嘲笑我那里面包含的所有拼写错误。（是的，git提供了自动改错如果你设定了的话，但是我习惯于让那些拼写错误的运行！）

现在去做有用的事情。

