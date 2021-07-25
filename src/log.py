import logging
import os
# log file
logname : str = '/TimeSeriesDecompose.log'
# Set path to import dataset and export figures
path = os.path.realpath(__file__)
path = r'%s' % path.replace(f'\\{os.path.basename(__file__)}','').replace('\\','/')
if path.find('autoML') != -1:
    path = r'%s' % path.replace('/autoML','')
elif path.find('src') != -1:
    path = r'%s' % path.replace('/src','')
logging.basicConfig(filename=path+logname,
                    format='%(asctime)s %(message)s',
                    datefmt='[%m/%d/%Y %H:%M:%S]',
                    filemode='a')
                    # level=logging.INFO)

def log(message):
    logging.info(message)
    print(message)



if __name__ == '__main__':
    log('testing')

    # # Close logging handlers to release the log file
    # handlers = logging.getLogger().handlers[:]
    # for handler in handlers:
    #     handler.close()
    #     logging.getLogger().removeHandler(handler)