import argparse
import os
import configparser


class CustomNamespace(argparse.Namespace):
    def __getitem__(self, key):
        return getattr(self, key, None)

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='智能问答项目参数配置！')

# 添加参数
parser.add_argument('--config', type=str, help='相对当前文件的配置文件路径', default='config.ini') #

# 解析命令行参数
my_args = parser.parse_args(namespace=CustomNamespace())

# 获取命令行参数的字典表示
args_dict = vars(my_args)

# 如果提供了配置文件路径，则读取配置文件并合并参数
if my_args.config:
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(__file__)
    config_file_path = os.path.join(script_dir, my_args.config)
    config = configparser.ConfigParser()
    config.read(config_file_path)


    config_dict = dict(config.items('Settings'))

    for key, value in config_dict.items():
        if args_dict.get(key) is None and value is not None:
            setattr(my_args, key, value)

