import logging
import os

def create_logger(log_dir):
    """创建一个logger，并将日志保存到指定目录下的文件中"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建文件处理器，用于将日志写入文件
    log_file = os.path.join(log_dir, 'Log.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到logger
    logger.addHandler(file_handler)

    return logger