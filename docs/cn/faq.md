## Q1:一键训练结束没有索引

显示"Training is done. The program is closed."则模型训练成功，后续紧邻的报错是假的；


一键训练结束完成没有added开头的索引文件，可能是因为训练集太大卡住了添加索引的步骤；已通过批处理add索引解决内存add索引对内存需求过大的问题。临时可尝试再次点击"训练索引"按钮。


## Q2:训练结束推理没看到训练集的音色
点刷新音色再看看，如果还没有看看训练有没有报错，控制台和webui的截图，logs/实验名下的log，都可以发给开发者看看。


## Q3:如何分享模型
  rvc_root/logs/实验名 下面存储的pth不是用来分享模型用来推理的，而是为了存储实验状态供复现，以及继续训练用的。用来分享的模型应该是weights文件夹下大小为60+MB的pth文件；

  后续将把weights/exp_name.pth和logs/exp_name/added_xxx.index合并打包成weights/exp_name.zip省去填写index的步骤，那么zip文件用来分享，不要分享pth文件，除非是想换机器继续训练；

  如果你把logs文件夹下的几百MB的pth文件复制/分享到weights文件夹下强行用于推理，可能会出现f0，tgt_sr等各种key不存在的报错。你需要用ckpt选项卡最下面，手工或自动（本地logs下如果能找到相关信息则会自动）选择是否携带音高、目标音频采样率的选项后进行ckpt小模型提取（输入路径填G开头的那个），提取完在weights文件夹下会出现60+MB的pth文件，刷新音色后可以选择使用。


## Q4:Connection Error.
也许你关闭了控制台（黑色窗口）。


## Q5:WebUI弹出Expecting value: line 1 column 1 (char 0).
请关闭系统局域网代理/全局代理。


这个不仅是客户端的代理，也包括服务端的代理（例如你使用autodl设置了http_proxy和https_proxy学术加速，使用时也需要unset关掉）


## Q6:不用WebUI如何通过命令训练推理
训练脚本：

可先跑通WebUI，消息窗内会显示数据集处理和训练用命令行；


推理脚本：tool/cmd/infer_cli.py


## Q7:Cuda error/Cuda out of memory.
小概率是cuda配置问题、设备不支持；大概率是显存不够（out of memory）；


训练的话缩小batch size（如果缩小到1还不够只能更换显卡训练），推理的话酌情缩小config.py结尾的x_pad，x_query，x_center，x_max。4G以下显存（例如1060（3G）和各种2G显卡）可以直接放弃，4G显存显卡还有救。


## Q8:total_epoch调多少比较好

如果训练集音质差底噪大，20~30足够了，调太高，底模音质无法带高你的低音质训练集

如果训练集音质高底噪低时长多，可以调高，200是ok的（训练速度很快，既然你有条件准备高音质训练集，显卡想必条件也不错，肯定不在乎多一些训练时间）


## Q9:需要多少训练集时长
  推荐10min至50min

  保证音质高底噪低的情况下，如果有个人特色的音色统一，则多多益善

  高水平的训练集（精简+音色有特色），5min至10min也是ok的，仓库作者本人就经常这么玩

  也有人拿1min至2min的数据来训练并且训练成功的，但是成功经验是其他人不可复现的，不太具备参考价值。这要求训练集音色特色非常明显（比如说高频气声较明显的萝莉少女音），且音质高；

  1min以下时长数据目前没见有人尝试（成功）过。不建议进行这种鬼畜行为。


## Q10:index rate干嘛用的，怎么调（科普）
  如果底模和推理源的音质高于训练集的音质，他们可以带高推理结果的音质，但代价可能是音色往底模/推理源的音色靠，这种现象叫做"音色泄露"；

  index rate用来削减/解决音色泄露问题。调到1，则理论上不存在推理源的音色泄露问题，但音质更倾向于训练集。如果训练集音质比推理源低，则index rate调高可能降低音质。调到0，则不具备利用检索混合来保护训练集音色的效果；

  如果训练集优质时长多，可调高total_epoch，此时模型本身不太会引用推理源和底模的音色，很少存在"音色泄露"问题，此时index_rate不重要，你甚至可以不建立/分享index索引文件。


## Q11:推理怎么选gpu
config.py文件里device cuda:后面选择卡号；

卡号和显卡的映射关系，在训练选项卡的显卡信息栏里能看到。


## Q12:如何推理训练中间保存的pth
通过ckpt选项卡最下面提取小模型。



## Q13:如何中断和继续训练
现阶段只能关闭WebUI控制台双击go-web.bat重启程序。网页参数也要刷新重新填写；

继续训练：相同网页参数点训练模型，就会接着上次的checkpoint继续训练。


## Q14:训练时出现文件页面/内存error
进程开太多了，内存炸了。你可能可以通过如下方式解决

1、"提取音高和处理数据使用的CPU进程数"  酌情拉低；

2、训练集音频手工切一下，不要太长。



## Q15:如何中途加数据训练
1、所有数据新建一个实验名；

2、拷贝上一次的最新的那个G和D文件（或者你想基于哪个中间ckpt训练，也可以拷贝中间的）到新实验名；下

3、一键训练新实验名，他会继续上一次的最新进度训练。


## Q16: error about llvmlite.dll

OSError: Could not load shared object file: llvmlite.dll

FileNotFoundError: Could not find module lib\site-packages\llvmlite\binding\llvmlite.dll (or one of its dependencies). Try using the full path with constructor syntax.

win平台会报这个错，装上https://aka.ms/vs/17/release/vc_redist.x64.exe这个再重启WebUI就好了。

## Q17: RuntimeError: The expanded size of the tensor (17280) must match the existing size (0) at non-singleton dimension 1.  Target sizes: [1, 17280].  Tensor sizes: [0]

wavs16k文件夹下，找到文件大小显著比其他都小的一些音频文件，删掉，点击训练模型，就不会报错了，不过由于一键流程中断了你训练完模型还要点训练索引。

## Q18: RuntimeError: The size of tensor a (24) must match the size of tensor b (16) at non-singleton dimension 2

不要中途变更采样率继续训练。如果一定要变更，应更换实验名从头训练。当然你也可以把上次提取的音高和特征（0/1/2/2b folders）拷贝过去加速训练流程。
