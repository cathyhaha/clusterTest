# coding:utf-8  

#博客文本聚类教程（20171207中提到的）中的代码，只将path改成本机文件所在路径。
#调试用 含单步输出提示语句！！！！！  
#20180109实验证明，可正常运行！！！！
# 2.0 使用jieba进行分词,彻底放弃低效的NLPIR,用TextRank算法赋值权重(实测textrank效果更好)  
# 2.1 用gensim搞tfidf  
# 2.2 sklearn做tfidf和kmeans  
# 2.3 将kmeans改成BIRCH,使用传统tfidf  
#20180115截取clustering中的前半段，生成降维后的矩阵并写入dat文件。
  
import logging  
import time  
import os  
import jieba  
import glob  
import random  
import copy  
import chardet  
import gensim  
from gensim import corpora,similarities, models  
from pprint import pprint  
import jieba.analyse  
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
import os  
from sklearn.decomposition import PCA  

from distance import *

  
  
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  
  
start = time.clock()  

print '#----------------------------------------#'  
print '#                                        #'  
print '#              载入语料库                #'  
print '#                                        #'  
print '#----------------------------------------#\n'  
def PreprocessDoc(root):  
  
    allDirPath = [] # 存放语料库数据集文件夹下面左右的文件夹路径,string,[1:]为所需  
    fileNumList = []  
  
    def processDirectory(args, dirname, filenames, fileNum=0):  
        allDirPath.append(dirname)  
        #print 'dirname   ----'+dirname
        #print 'filenames  ----- '+str(filenames)
        for filename in filenames:  
            fileNum += 1  
            #print 'filename'+filename
        fileNumList.append(fileNum)  
    os.path.walk(root, processDirectory, None)  
    totalFileNum = sum(fileNumList)  
    print '总文件数为: ' + str(totalFileNum)  
    #print 'allDirPath: ' + str(allDirPath)
    return allDirPath  
  
  
print '#----------------------------------------#'  
print '#                                        #'  
print '#              合成语料文档                #'  
print '#                                        #'  
print '#----------------------------------------#\n'  
  
# 每个文档一行,第一个词是这个文档的类别  
  
def SaveDoc(allDirPath, docPath, stopWords):  
  
    print '开始合成语料文档:'  
  
    category = 1 # 文档的类别  
    f = open(docPath,'w') # 把所有的文本都集合在这个文档里  
  
    for dirParh in allDirPath[1:]:  
  
        for filePath in glob.glob(dirParh + '/*.txt'): 
            #print 'filePath:1111----- ' + str(filePath)
  
            data = open(filePath, 'r').read()  
            texts = DeleteStopWords(data, stopWords)  
            line = '' # 把这些词缩成一行,第一个位置是文档类别,用空格分开  
            for word in texts:  
                if word.encode('utf-8') == '\n' or word.encode('utf-8') == 'nbsp' or word.encode('utf-8') == '\r\n':  
                    continue  
                line += word.encode('utf-8')  
                line += ' '  
            f.write(line + '\n') # 把这行写进文件  
        category += 1 # 扫完一个文件夹,类别+1  
  
    return 0 # 生成文档,不用返回值  
  
  
print '#----------------------------------------#'  
print '#                                        #'  
print '#             分词+去停用词               #'  
print '#                                        #'  
print '#----------------------------------------#\n'  
def DeleteStopWords(data, stopWords):  
  
    wordList = []  
  
    # 先分一下词  
    cutWords = jieba.cut(data)  
    for item in cutWords:  
        if item.encode('utf-8') not in stopWords: # 分词编码要和停用词编码一致  
            wordList.append(item)        
  
    return wordList  
  
  
print '#----------------------------------------#'  
print '#                                        #'  
print '#                 tf-idf                 #'  
print '#                                        #'  
print '#----------------------------------------#\n'  
def TFIDF(docPath):  
  
    print '开始tfidf:'  
  
    corpus = [] # 文档语料  
  
    # 读取语料,一行语料为一个文档  
    lines = open(docPath,'r').readlines()  
    for line in lines:  
        corpus.append(line.strip()) # strip()前后空格都没了,但是中间空格还保留  
  
    # 将文本中的词语转换成词频矩阵,矩阵元素 a[i][j] 表示j词在i类文本下的词频  
    vectorizer = CountVectorizer()  
  
    # 该类会统计每个词语tfidf权值  
    transformer = TfidfTransformer()  
  
    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵  
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  
  
    # 获取词袋模型中的所有词语  
    word = vectorizer.get_feature_names()  
  
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重  
    weight = tfidf.toarray()  
    #print weight  
   # print word
  
    # # 输出所有词  
    #result = open(docPath, 'w')  
    #for j in range(len(word)):
     #   result.write(word[j].encode('utf-8') + ' ')  
    #result.write('\r\n\r\n')  
      
    # # 输出所有权重  
    #for i in range(len(weight)):  
    #    for j in range(len(word)):
    #        result.write(str(weight[i][j]) + ' ')  
    #    result.write('\r\n\r\n')  
      
   # result.close()  
  
    return weight  
  
  
print '#----------------------------------------#'  
print '#                                        #'  
print '#                   PCA                  #'  
print '#                                        #'  
print '#----------------------------------------#\n'  
def PCA(weight, dimension):  
  
    from sklearn.decomposition import PCA  
  
    print '原有维度: ', len(weight[0])  
    print '开始降维:'  
  
    pca = PCA(n_components=dimension) # 初始化PCA  
    X = pca.fit_transform(weight) # 返回降维后的数据  
    print '降维后维度: ', len(X[0])  
    print X  
  
    return X  

def build_distance_file_for_cluster(vectors,distance_obj, filename):
    '''
    Save distance and index into file

    Args:
        distance_obj : distance.Distance object for compute the distance of two point
        filename     : file to save the result for cluster
    '''
    fo = open(filename, 'w')
    for i in xrange(len(vectors) - 1):
      for j in xrange(i, len(vectors)):
        fo.write(str(i + 1) + ' ' + str(j + 1) + ' ' + str(distance_obj.distance(vectors[i], vectors[j])) + '\n')
    fo.close()

if __name__ == "__main__":  
  
    root = 'test'  
    stopWords = open('chineseStopWords-utf8.txt', 'r').read()   
    docPath = 'test/doc.txt' 
    weightPCA2dat='../data/data_others/weightPCA2dat.dat'
    k = 3   
  
    allDirPath = PreprocessDoc(root)  
    print 'allDirPath'+str(allDirPath)
    SaveDoc(allDirPath, docPath, stopWords)  
  
    weight = TFIDF(docPath)  
    X = weight  #PCA(weight, dimension=800) # 将原始权重数据降维  
    build_distance_file_for_cluster(X,SqrtDistance(), weightPCA2dat)
  
  
end = time.clock()  
print '运行时间: ' + str(end - start)  

  