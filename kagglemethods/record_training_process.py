# _*_coding:utf-8_*_
import os


class neptune():
    '''
       neptune.send_metric('fold:{}-train_mse'.format(self.fold),
                                epoch, mse_train)

        neptune.send_text('xxxx')

        neptune(name='deepfm-diveintodeep',
                          params={
                              'batch_size': batch_size_train,
                              #   'weight_decay':weight_decay,
                              'lr': lr,
                              'embedding_size': embedding_size,
                              'dense_dim': dense_dim
                          })


    '''

    def __init__(self, name, params):
        if not os.path.exists(name):
            os.mkdir(name)
        subdir = ""
        for i, (key,value) in enumerate(params.items()):
            key_value_name = ('_' if i > 0 else '') + str(key) + \
                '-' + str(value)
            subdir += key_value_name
        subpath = os.path.join(name,subdir)
        if not os.path.exists(subpath):
            os.mkdir(subpath)
        self.tardir = subpath
    
    def send_metric(self,filename, content):
        '''
        以append模式添加内容
        '''

        filename +='.txt'
        filepath = os.path.join(self.tardir,filename)
        with open(filepath,'a') as fh:
            fh.writelines(content)
            fh.writelines('\n')

    def send_text(self, content):
        '''
        存储在默认的位置
        '''
        filename='text_content.txt'
        filepath = os.path.join(self.tardir,filename)
        with open(filepath,'a') as fh:
            fh.writelines(content)
            fh.writelines('\n')


if __name__ == '__main__':
    nobj = neptune('hello',params={
                              'batch_size': 1,
                              #   'weight_decay':weight_decay,
                              'lr': 1,
                              'embedding_size': 1,
                              'dense_dim': 1
                          })
    nobj.send_metric('hello','222')
    nobj.send_metric('hello','333')




