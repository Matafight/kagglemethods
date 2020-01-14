#_*_ coding:utf-8 _*_

'''
将测试集和验证集上误差随着迭代次数的变化曲线画出来
输入参数可以是多个list
以及需要保存的文件名（包含路径）
'''
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 



class draw():
    ## kargs 是多个list
    def __init__(self,file_path='./tmp.png'):
        self.file_path = file_path
    def plot(self,kargs,names):
        for s in kargs:
            sns.lineplot(x=range(1,len(s)+1),y=s)
        plt.legend(title='paras', loc='upper left', labels=names)

        plt.savefig(self.file_path,dpi=1000)
        plt.show()




