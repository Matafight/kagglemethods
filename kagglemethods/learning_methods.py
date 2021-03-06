#_*_coding:utf-8_*_

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from . import log_class



class learning_methods(object):
    """
    这是一个基类，继承该类的子类可以直接使用基类的函数也可以重写基类的函数，xgboost方法就应该重写，sklearn内嵌的方法应该可以直接使用。cv_score 应该支持引入评估准则参数
    """
    def __init__(self,x,y,metric,metric_proba = False,labels = None,scoring = 'auc',save_model=False,processed_data_version_dir='./',if_classification=0):
        """
        初始化相关参数

        args:
            x: numpy array
            y: numpy array
            metric: sklearn 中的函数，用来在交叉验证中评估验证集上的效果，不过auc 不行，因为auc的参数 不是 (y_true,y_pred) 的形式
                 optional: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
            metric_proba: False, 表示 metric 函数是否接受模型输出0-1之间的概率值
            labels: 默认为[0,1]， 在分类任务中，进行预测时可能所有预测数据集都是正类或都是负类，这个参数是用来告诉metric 应该是又两个类
            scoring: 用sklearn自带的 GridSearchCV 时需要的评估函数, 一般是越大越好。默认为 neg_mean_squared_error
                    可选项: 'neg_log_loss' 'roc_auc' ,'neg_mean_squared_error' 等
            n_jobs: 多少个线程,默认为2
            save_model: True or False, 表示是否保存模型,保存路径为 processed_data_version_dir/modules/
            processed_data_version_dir: 存放log 或者保存模型的目录,默认为 ./ 
            if_classification:0 or 1 表示是否为分类任务
        """
        import os
        if not os.path.exists(processed_data_version_dir):
            os.mkdir(processed_data_version_dir)
        self.x = x
        self.y = y
        self.metric = metric
        self.model = None
        self.metric_proba = metric_proba
        self.labels = labels
        self.save_model = save_model
        self.scoring = scoring
        self.train_scores = []
        self.cv_scores = []

        self.path = processed_data_version_dir
        self.if_classification = if_classification
        

    def plot_save(self,name='learning_method'):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(self.train_scores)
        ax1.set_xlabel('train_scores')
        ax1.set_ylabel(self.metric.__name__)
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(self.cv_scores)
        ax2.set_xlabel('cv scores')

        #save 
        import os
        npath = self.path
        if not os.path.exists(npath+'/curve'):
            os.mkdir(npath+'/curve')
        import time
        cur_time = time.strftime("%Y-%m-%d-%H-%M",time.localtime())
        fig.savefig(npath+'/curve/'+name+'_'+cur_time+'_train_cv.png')

        #save train score and cv score
        logger = log_class.log_class(name,top_level = npath)
        str_train_score ="train score sequence "+" ".join([str(item) for item in self.train_scores])
        str_cv_score = "cv score sequence"+ " ".join([str(item) for item in self.cv_scores])
        logger.add(str_train_score)
        logger.add(str_cv_score)   
        #determine if save model
        if self.save_model:
            #save model here
            from sklearn.externals import joblib
            if not os.path.exists(npath+'/modules'):
                os.mkdir(npath+'/modules')
            joblib.dump(self.model,npath+'/modules/'+name+"_"+cur_time+".pkl")

    def train_score(self):
        self.model.fit(self.x,self.y)
        if self.metric_proba == False:
            pred_train = self.model.predict(self.x)
        else:
            pred_train = self.model.predict_proba(self.x)[:,1]
        
        if self.labels==None:
            score = self.metric(self.y,pred_train)
        else:
            score = self.metric(self.y,pred_train,labels=self.labels)
        self.train_scores.append(score)

    def cv_score(self):
        # 3-fold crossvalidation error
        if self.if_classification == 1:
            kf = StratifiedKFold(n_splits = 3,shuffle=True,random_state=2018)
        else:
            kf = KFold(n_splits=3,shuffle=True,random_state=2018)
        score = []
        for train_ind,test_ind in kf.split(self.x,self.y):
            train_valid_x,train_valid_y = self.x[train_ind],self.y[train_ind]
            test_valid_x,test_valid_y = self.x[test_ind],self.y[test_ind]
            self.model.fit(train_valid_x,train_valid_y)
            if self.metric_proba == False:
                pred_test = self.model.predict(test_valid_x)
            else:
                print("==========predict proba===========")
                pred_test = self.model.predict_proba(test_valid_y)[:,1]
            ##这个本意是为了防止交叉验证中缺少某一类样本而设置的参数labels,但其实很多时候都没有必要，而且roc_auc_score并不支持labels这个参数
            #if self.labels == None:
            score.append(self.metric(test_valid_y,pred_test))
            #else:
            #    score.append(self.metric(test_valid_y,pred_test,labels=self.labels))

        mean_cv_score = np.mean(score)
        self.cv_scores.append(mean_cv_score)
        print('final {} on cv {}'.format(self.metric.__name__,mean_cv_score))
    
    #scoring : neg_mean_squared_error 
    def cross_validation(self,scoring):
        pass
    

