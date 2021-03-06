#_*_ coding:utf-8

import xgboost as xgb
import numpy as np
from xgboost.sklearn import  XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from . import log_class
#parameters to be tunned
tunned_max_depth = [3,5,7,9]
tunned_learning_rate =[0.01,0.015,0.025,0.05,0.1]  # aka eta in xgboost
tunned_min_child_weight = [1,3,5,7]
tunned_gamma = [0.05,0.1,0.3,0.5,0.7,0.9,1]
tunned_colsample_bytree = [0.6,0.7,0.8,1]
# 还需要添加subsample,lambda,alpha等参数


#define a decorator function to record the running time
'''
http://xgboost.readthedocs.io/en/latest/parameter.html
http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
'''
class xgboost_CV(object):
    def __init__(self,x,y,tunning_params,metric,metric_proba = 0,metric_name='auc',scoring='roc_auc',n_jobs=2,save_model = 0,processed_data_version_dir = './',if_classification=False):
        '''
        tunning_params: 字典类型，key是待调整参数名,values是候选集合
        metric_proba indicates if the metric need the probability to calculate the score
        metric 其实就是训练过程中的评估函数，是我自己手动用来评估训练效果的。
        metric_name 是xgboost cv 中的一个参数。
        scoring是专门给GridSearchCV设置的一个参数，因为GridSearchCV只接受指定的那几个字符串。
        scoring只可以选择sklearn对应的几个，其值是越大越好。
        当metric是越高越好时，他们三者是一致的，否则 scoring与 metric 和 metric_name 不一致。
        '''
        #default metrics should be metrics.log_loss
        import os
        if not os.path.exists(processed_data_version_dir):
            os.mkdir(processed_data_version_dir)
        self.x =  x
        self.y = y
        self.tunning_params = tunning_params
        # define the algorithm using default parameters
        self.model = None
        self.cv_folds = 5
        self.early_stopping_rounds = 50
        self.metric = metric
        self.metric_proba = metric_proba
        self.metric_name = metric_name
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.save_model = save_model
        self.train_scores = []
        self.cv_scores = []
        self.logger = log_class.log_class('xgboost',top_level = processed_data_version_dir)
        self.path = processed_data_version_dir
        self.if_classification = if_classification

    def plot_save(self,name='xgboostRegression'):
        '''
        :param name: 模型名称
        :return: None,但保存模型为pkl到相关目录
        '''
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
        str_train_score ="train score sequence "+" ".join([str(item) for item in self.train_scores])
        str_cv_score = "cv score sequence"+ " ".join([str(item) for item in self.cv_scores])
        self.logger.add(str_train_score)
        self.logger.add(str_cv_score)

        #determine if save model
        if self.save_model==1:
            #save model here
            from sklearn.externals import joblib
            if not os.path.exists(npath+'/modules'):
                os.mkdir(npath+'/modules')
            joblib.dump(self.model,npath+'/modules/'+name+"_"+cur_time+".pkl")

    # get the best num rounds after changing a parameter
    def modelfit(self):
        xgb_param = self.model.get_xgb_params()
        dtrain = xgb.DMatrix(self.x,label = self.y)
        cvresult = xgb.cv(xgb_param,dtrain,num_boost_round=500,nfold=self.cv_folds,metrics=self.metric_name,early_stopping_rounds=self.early_stopping_rounds)
        self.model.set_params(n_estimators=cvresult.shape[0])
        self.model.fit(self.x,self.y)

        #calculate train score
        if self.metric_proba == 0:
            pred = self.model.predict(self.x)
        else:
            pred = self.model.predict_proba(self.x)
        self.train_scores.append(self.metric(self.y,pred))


    def cv_score(self):
        # 5-fold crossvalidation,对回归和分类任务是不同的
        if self.if_classification:
            kf = StratifiedKFold(n_splits = 5,shuffle=True,random_state=2018)
        else:
            kf = KFold(n_splits=5,shuffle=True,random_state=2018)
        score = []
        params = self.model.get_xgb_params()
        for train_ind,test_ind in kf.split(self.x,self.y):
            train_valid_x,train_valid_y = self.x[train_ind],self.y[train_ind]
            test_valid_x,test_valid_y = self.x[test_ind],self.y[test_ind]
            dtrain = xgb.DMatrix(train_valid_x,label = train_valid_y)
            dtest = xgb.DMatrix(test_valid_x)
            pred_model = xgb.train(params,dtrain,num_boost_round=int(params['n_estimators']))
            if self.metric_proba == 0:
                pred_test = pred_model.predict(dtest)
            else:
                pred_test = pred_model.predict_proba(dtest)
            score.append(self.metric(test_valid_y,pred_test))
        mean_cv_score = np.mean(score)
        self.cv_scores.append(mean_cv_score)
        print('final {} on cv:{}'.format(self.metric.__name__,mean_cv_score))


    def cross_validation(self):
        scoring = self.scoring
        for param_item in self.tunning_params.keys():
                print('tunning {} ...'.format(param_item))
                params = {param_item:self.tunning_params[param_item]}
                print(self.model.get_params())
                gsearch = GridSearchCV(self.model,param_grid=params,scoring=scoring,n_jobs=1,iid=False,cv=3)
                gsearch.fit(self.x,self.y)
                #这是一种将变量作为参数名的方法
                to_set = {param_item:gsearch.best_params_[param_item]}
                self.model.set_params(**to_set)
                print(gsearch.best_params_)
                self.modelfit()
                print('best_num_round after tunning para: {}'.format(self.model.get_params()['n_estimators']))
                self.cv_score()


        ## 保存最终的参数
        params = self.model.get_params()
        self.logger.add(params,ifdict=1)
        #用新参数重新训练一遍
        self.model.fit(self.x,self.y)

        return self.model


















