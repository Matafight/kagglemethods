#_*_coding:utf-8_*_
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from . import learning_methods
import numpy as np
from . import log_class

from sklearn.svm import SVC

tunned_C =[0.5,1,1.3]

class SVC_CV(learning_methods.learning_methods):
    def __init__(self,x,y,tunning_params,metric,metric_proba = True,labels = [0,1],scoring='neg_log_loss',n_jobs=2,save_model=False,processed_data_version_dir='./'):
        """
        初始化相关参数
        args:
            x: numpy array
            y: numpy array
            tunning_params: 字典类型，key是待调整参数名,values是候选集合
            metric: sklearn 中的函数，用来在交叉验证中评估验证集上的效果
                 optional: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
            metric_proba: False, 表示 metric 函数是否接受模型输出0-1之间的概率值
            labels: 默认为[0,1]， 在分类任务中，进行预测时可能所有预测数据集都是正类或都是负类，这个参数是用来告诉metric 应该是又两个类
            scoring: 用sklearn自带的 GridSearchCV 时需要的评估函数, 一般是越大越好。默认为 neg_mean_squared_error
                    可选项: 'neg_log_loss' 'roc_auc' ,'neg_mean_squared_error' 等
            n_jobs: 多少个线程,默认为2
            save_model: True or False, 表示是否保存模型,保存路径为 processed_data_version_dir/modules/
            processed_data_version_dir: 存放log 或者保存模型的目录,默认为 ./ 
        """
        super(SVC_CV,self).__init__(x,y,metric,metric_proba = metric_proba,labels = labels,scoring=scoring,save_model=save_model,processed_data_version_dir=processed_data_version_dir)
        #default kernel is rbf
        
        if metric_proba == True:
            self.model = SVC(C = 1.0,probability=True)
        else:
            self.model = SVC(C=1.0)
        self.n_jobs = n_jobs
        self.tunning_params = tunning_params
        self.scoring = scoring
 
  
        self.logger = log_class.log_class('svc',top_level=processed_data_version_dir)




    ##def cross_validation(self):
    ##    scoring = self.scoring
    ##    self.train_score()
    ##    self.cv_score()
    ##    params = {'C':tunned_C}
    ##    gsearch = GridSearchCV(estimator=self.model,param_grid=params,scoring=scoring,n_jobs=self.n_jobs,iid=False,cv=3)
    ##    gsearch.fit(self.x,self.y)
    ##    self.model.set_params(C= gsearch.best_params_['C'])
    ##    print('best C for svc:{}'.format(gsearch.best_params_['C']))
    ##    self.cv_score()
    ##    self.train_score()
    ##    self.plot_save("SVC")
    ##    return self.model
    
    def cross_validation(self):
        scoring = self.scoring
        for ind,param_item in enumerate(self.tunning_params.keys()):
            print('tunning {} ...'.format(param_item))
            params = {param_item:self.tunning_params[param_item]}
            print(self.model.get_params())
            gsearch = GridSearchCV(self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
            gsearch.fit(self.x,self.y)
            #这是一种将变量作为参数名的方法
            to_set = {param_item:gsearch.best_params_[param_item]}
            self.model.set_params(**to_set)
            ###保存每个参数及其分数到log中
            self.logger.add('======={}====='.format(str(ind)))
            for i,(params_cv,mean_score) in enumerate(zip(gsearch.cv_results_['params'],gsearch.cv_results_['mean_test_score'])):
                ##将dict拼接成字符串
                for key in params_cv.keys():
                    sent = '{}-{}:{}'.format(key,str(params_cv[key]),str(mean_score))
                self.logger.add(sent)
            print(gsearch.best_params_)
            self.cv_score()

        ## 保存最终的参数
        params = self.model.get_params()
        self.logger.add(params,ifdict=1)
        #用新参数重新训练一遍
        self.model.fit(self.x,self.y)

        return self.model