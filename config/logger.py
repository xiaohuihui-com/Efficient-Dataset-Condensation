# logger.py
# _*_ coding: utf-8 _*_
import logging
import os.path
import time
import yaml
import logging.config

project_path = 'IDC'  # 定义项目目录


class Logger(object):
    def __init__(self, logdir='./logs', config='logger.yaml'):
        self.logdir = logdir
        self.config = os.path.join(os.path.dirname(__file__), config)
        # self.log_set_init()

        dir_time = time.strftime('%Y%m%d', time.localtime(time.time()))
        # 获取当前时间作为日志文件名称
        current_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        # 定义日志文件路径以及名称
        log_name = os.path.join(self.logdir, dir_time, current_time + '.log')

        if not os.path.exists(os.path.join(self.logdir, dir_time)):
            os.makedirs(os.path.join(self.logdir, dir_time))

        try:
            with open(self.config, "r", encoding="utf-8") as file:
                logging_yaml = yaml.safe_load(file)
                # logging_yaml = yaml.load(stream=file, Loader=yaml.FullLoader)
                logging_yaml['handlers']['file_handler']['filename'] = log_name
            # 配置logging日志
            logging.config.dictConfig(config=logging_yaml)
            # 创建一个logger(初始化logger)
            self.log = logging.getLogger()
        except Exception as e:
            print(e)


    # # 日志接口
    def debug(self, msg):
        self.log.debug(msg)
        return

    def info(self, msg):
        self.log.info(msg)
        return

    def error(self, msg):
        self.log.error(msg)
        return


if __name__ == '__main__':
    logger = Logger()
    logger.info('This is info')
    logger.debug('This is debug')
    logger.error('This is error')
