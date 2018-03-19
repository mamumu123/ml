import os
import copy
import pickle
from sklearn import  tree
from pandas import Series
from sklearn.datasets import load_iris
import pydotplus
import test
import clas
#read
def read_file(filepath,fist,last):
    #if(fist==200):
        #print(filepath)
        #print("-----------------------------------------------------")
    global numPerDir
    files = os.listdir(filepath)
    s = []
    fileNum = 0
    # 每一篇文章
    for file in files:
        if not os.path.isdir(file):
            if fileNum<fist:
                pass
            else:
                if fileNum == last:
                    break
                else:
                    fd = open(filepath + file, 'r', encoding='utf-8', errors="ignore")
                    iter_i = iter(fd)
                    sstr = ''
                    for line in iter_i:
                        sstr += line
                # juzhen_init(sstr)
                    s.append(sstr)
                    fd.close()
            fileNum += 1
            # juzhen_init(s)
    #print("fileNum:     ",fileNum)

    return s
# 处理s,将每一个文本，都转化为list
def str_parser(s):
    global NNNum
    num = 0
    listS = []
    for i in s:
        alpha = i.split('\t')

        # print(alpha,'\n')
        alpha[-1] = alpha[-1][:-1]  # 除去\n

        # print('\n',alpha[-1],'\n')
        num += len(alpha)
        listS.append(alpha)
    # print('第',"  有  " ,num,"个词"  )
    #NNNum += num
    return listS
# 把list ，都存起来
def list_to_dict(listS):
    dictS = {}
    #global totalT
    for eachTxt in listS:
        # print("eachTxt:  ",eachTxt)
        for fenci in eachTxt:
            # print("fenci:  ",fenci)

            if fenci in dictS:
                dictS[fenci] += 1
            else:
                dictS[fenci] = 1
            if fenci in totalT:
                totalT[fenci] += 1
            else:
                totalT[fenci] = 1
    return dictS
#决策树
def func(juzhen,target):
    clf = tree.DecisionTreeClassifier(max_depth=200,criterion='entropy')  # 决策树
    clf = clf.fit(juzhen, target)

    dot_data = tree.export_graphviz(clf, out_file=None)  # pdf
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("./pdf/iris_idr_2.pdf")
    return clf
#矩阵创建
def juzhen_init(n,m):

    juzhe = [[0 for i in range(m)] for j in range(n)]
    for numDir in range(9):
        #for Dir in listAll[numDir]:
        for numTxt in  range(len(listAll[numDir])):
        #print(file_d)
            for fenci in listAll[numDir][numTxt]:
                    #print(fenci)
                if fenci in listKey:
                    ind = listKey.index(fenci)
                    juzhe[numDir*len(listAll[numDir])+numTxt][ind]=1
    return  juzhe
########################################################################################
#####                         主函数
########################################################################################
if __name__ == '__main__':
    numPerDir=350


    NNNum = 0
    totalT = {}  # 计录总的词的个数
    dict_juzhen = {}
    listDir = ['体育', '健康', '军事', '娱乐', '房产', '教育', '科技', '证券', '财经']

    dictAll = []  # 储存总的词
    # 读取每一个文件夹
    listAll=[] #保村所有的list
    #读取每一个文件夹
    for dir in listDir:
        filepath = './new_weibo_13638/' + dir + '/'
        s = read_file(filepath,0,350)#获取每一个文件下的str
        listS = str_parser(s)#将str 拆分为list
        listAll.append(listS)#[[[]]]每一篇文章的list
        dictS = list_to_dict(listS)
       # print("dict:  ",len(dictS))
        dictSort = sorted(dictS.items(), key=lambda x: x[1], reverse=True)
        dictAll.append(dictSort)
    dictAllSort = sorted(totalT.items(), key=lambda x: x[1], reverse=True)
    #print(dictAllSort)
    print(dictAllSort,'\n',len(dictAllSort))
    #for i in
    print(totalT)
    Tota=[  i for i in dictAllSort if i[1]>20]
    print(Tota,'\n',len(Tota))

    #建立矩阵
    import pickle

    ###
    #lll=[i[0] for i in Tota]
    listKey=[i[0] for i in Tota]
    print(listKey)
    txt_num = numPerDir*9
    tezheng_num=len(listKey)
    juzhen=[]
    juzhen =  [[0 for i in range(tezheng_num)] for j in range(txt_num)]
    fileF=0
    juzhen = juzhen_init(txt_num,tezheng_num)

    #target
    target = [0 for i in range(txt_num)]
    #print(len(target))
    for i in range(len(juzhen)):
        target[i]=int(i//numPerDir)
###########    决策树                                             #################
    #decison=func(juzhen,target)

    dataList=[]
    labelList=listKey
    print('labelList   ',labelList)
    dataList = [[juzhen[j][i] for i in range(len(juzhen[0]))] for j in range(len(juzhen))]
    for i in range(len(dataList)):
        dataList[i].append(target[i])

    classs=clas.TTTT()
    Tr=classs.createTree(dataList,  labelList,100)


    #classs.storeTree(Tr,'./txt/tree.txt')
    fw = open('./txt/tree.txt', 'wb')
    # pickle的dump函数将决策树写入文件中
    pickle.dump(Tr, fw)
    # 写完成后关闭文件
    fw.close()

    fw = open('./txt/listKey.txt', 'wb')
    # pickle的dump函数将决策树写入文件中
    pickle.dump(listKey, fw)
    # 写完成后关闭文件
    fw.close()

