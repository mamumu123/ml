import clas
import os
from sklearn import  tree
import pydotplus
import  pickle
###
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

#from main import  totalT
if __name__ == '__main__':
    #global totalT
    #########ready
    numPerDir_test = 50
    classs=clas.TTTT()

    filename='./txt/tree.txt'

    fr=open(filename,'rb')
    Tr=pickle.load(fr)
    fr.close()
    # pickle的dump函数将决策树写入文件中
    fr = open('./txt/listKey.txt', 'rb')
    listKey=pickle.load(fr)
    # 写完成后关闭文件
    fr.close()

    listDir = ['体育', '健康', '军事', '娱乐', '房产', '教育', '科技', '证券', '财经']
    listS_test = []
    listAll_test = []
    dictAll_test = []
    for dir in listDir:
        filepath_test = './new_weibo_13638/' + dir + '/'
        s_test = read_file(filepath_test, 350, 400)  # 获取每一个文件下的str
        # print("s_test",s_test)
        listS_test = str_parser(s_test)  # 将str 拆分为list
        listAll_test.append(listS_test)  # [[[]]]每一篇文章的list
        dictS_test = list_to_dict(listS_test)
        # print("dict:  ",len(dictS))
        dictSort_test = sorted(dictS_test.items(), key=lambda x: x[1], reverse=True)
        dictAll_test.append(dictSort_test)

    listKey_test = list(listKey)
    txt_num_test = numPerDir_test * 9
    tezheng_num_test = len(listKey_test)
    juzhen_test = [[0 for i in range(tezheng_num_test)] for j in range(txt_num_test)]
    for numDir in range(9):
        # for Dir in listAll[numDir]:
        for numTxt in range(len(listAll_test[numDir])):
            # print(file_d)
            for fenci in listAll_test[numDir][numTxt]:
                # print(fenci)
                if fenci in listKey_test:
                    ind = listKey_test.index(fenci)
                    juzhen_test[numDir * len(listAll_test[numDir]) + numTxt][ind] = 1

    # target
    target_test = [0 for i in range(txt_num_test)]
    # print(len(target_test))
    for i in range(len(juzhen_test)):
        target_test[i] = int(i // numPerDir_test)
    #label=[str(i) for i in range(len(juzhen_test))]
    label=listKey_test
    #print(label)
    result=[]
    score = 0
    numtest=0
    for i in range(len(juzhen_test)):
        result.append(classs.predict(Tr, label, juzhen_test[i]))
        numtest+=1
        print('numtest',numtest)
        print("result:" ,result[i])

    numPer=list(range(9))
    for rr in range(len(result)):
        inRr=int(rr//numPerDir_test)
        if(inRr==result[rr]):
            score+=1
            numPer[inRr]+=1
    print(numPer)
    print('hit:    ',score,"\ntotal",len(result),'\nscore:   ',score/len(result))
    fd = open('./txt/result.txt', 'w', encoding="utf-8", errors="ignore")
        #for i in result:
    print(result, file=fd)
    fd.close()
