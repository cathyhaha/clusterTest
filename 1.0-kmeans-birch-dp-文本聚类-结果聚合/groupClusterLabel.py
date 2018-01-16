# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:21:04 2018

@author: qixianting
from pandas import *
def groupResults(resultLabel,target):
    df=DataFrame({'cluster':['a','a','b','b','a'],'target':['one','two','one','two','one'],'C':np.random.randn(5)})
    target=['1','2','1','2','3','0']
    
    df["cluster"]=resultLabel
    df["target"]=target
    
    resultdata=df.groupby(['cluster','target']).size()
    print(resultdata)
    
        
#生成target n组m个    
def target(n,m):
    a=np.arange(n)
    b=a.repeat(m)
    return b
"""
from pandas import *

def groupClusterLabel(resultLabel,targetClassesNum,targetFileNum):
    print '#--------groupClusterLabel--#'
    
    total=targetClassesNum*targetFileNum
    dataFrame=DataFrame({'cluster':np.random.randn(total),'target':np.random.randn(total),'C':np.random.randn(total)})
    targetLabel=produceTargetLabel(targetClassesNum,targetFileNum)
    
    dataFrame["cluster"]=resultLabel
    dataFrame["target"]=targetLabel
    
    print(dataFrame)  #按列输出簇标签和类标签
    resultData=dataFrame.groupby(['cluster','target']).size()    
    print(resultData) #按两竖排聚合
    result=resultData.unstack()
    print(result)     #按行竖聚合
    
def produceTargetLabel(targetClassesNum,targetFileNum):
    a=np.arange(targetClassesNum)
    b=a.repeat(targetFileNum)
    return b

if __name__ == "__main__":      
    targetClassesNum=3
    targetFileNum=2
    resultLabel=['1','1','1','2','2','0']
    groupClusterLabel(resultLabel,targetClassesNum,targetFileNum)
    
    
    