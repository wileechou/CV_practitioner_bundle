# Instruction
All codes come from 'Deep Learning for Computer Vision with Python 3rd Edition'--Pracitioner bundle, and I made some changes according to my device and environment.
You can turn to me for newest file of pdf, but it's not free since this series of books cost a lot.
In order to help user deploy this project, here's my information.

# Device: 
  Lenovo R7000 R5-4600H GTX1650

# Enviroment:
  OS: ubuntu 20.04 LTS
  
  python:3.7
  
  tensorflow:2.1.0
  
  (I use anaconda to automatically install tensorflow-gpu, and remember to select version=2.1.0. 
  Since some scripts will run into OMM problems, you need to set GPU allocation strategy which will not success in newest version)
  
# To Start
  deploy your environment based on information above, and I have pointed out the essential configuration. You can check #import# lines to find other
  packages you need to install.

# To be continued
  There're some problems that I didn't figure out well such as how to import the package of parent directory in current directory. I simply use 'sys.path.append("..")', and it works.
  I am not fimilar with Linux OS, so if you have any idea with the method 'export path' please inform me.
  
# Conclusion
  I keep on learning so all criticisms and corrections are welcome，but be sure to be quick.
  
# 说明
所有的代码都是出自DL4CV第三版的practitioner bundle,然后我根据自己的设备以及环境作了调整。你可以找我要该书第三版本的文件，但是由于这套教材很贵，所以我不会免费提供，希望理解。
为了让大家能够轻易在本地部署该工程，下面是我的配置信息

# 设备
  联想 R7000 R5-4600H GTX1650（穷学生党的神机，不解释）

# 环境
  操作系统：ubuntu 20.04 LTS（后续我会贴一个链接记录我在安装ubuntu双系统时踩的坑以及解决方法，如果你的设备和我一样，可以参考该文）
  
  python:3.7
  
  tensorflow:2.1.0
  
  现在keras直接在tensorflow中就可以用，所以我直接使用anaconda安装tensorflow的gpu版本。重要在于一定要安装2.1版本（如果跑训练时出现failed to get convolution algorithm等错误），因为
  我机子的显存不大，很容易就溢出，需要插入一些代码就行调整策略，这些代码在最新的几个tensorflow版本是不会成功的。
 
# 开始
  根据我提供的配置信息，部署你的环境。其它需要安装的python包就可以直接看.py文件中的import就行了，因为要安装的不多，所以我就不列举了（主要是懒）

# 待补充
  在我跑书上代码时，因为在我这个工程结构下,terminal运行时很容易出现No module named XXX错误，这是我没有弄得很清楚的地方，我百度了一些方法，其中sys.path.append("..")可以解决。但我也知道linux
  系统能够直接在终端使用export path=这种语句进行解决，但我对linux还不太熟，如果有熟悉的老哥能解决可以教我一下。
  
# 结语
  这套教材是我学习CV过程中遇到最好的教材，所以我也想认真的阅读每一章节，跑好并理解每一段代码，但本菜鸟还在进化中，如果有任何批评和建议麻烦快点提出来，我好久好久渴望这个批评和建议了。

  
