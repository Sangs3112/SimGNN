import argparse
class Parser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description='SimGNN with Official Settings')
        self._set_arguments()

    def _set_arguments(self):
        # 数据集文件路径，默认为datasets目录
        self.parser.add_argument('--dataset_path'
                            , type=str
                            , default='./datasets/'
                            , help='path to the datasets')
        
        # 接下来要使用的数据集，默认为AIDS700nef
        self.parser.add_argument('--dataset'
                            , type=str
                            , default='AIDS700nef'
                            , choices=['AIDS700nef', 'LINUX', 'IMDBMulti', 'ALKANE']
                            , help='the specific dataset which will be used next')
        
        # 日志文件路径，默认为Logs目录
        self.parser.add_argument('--log_path'
                            , type=str
                            , default='./Logs/'
                            , help='path to logs')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise ValueError('Unknown argument: {}'.format(unparsed))
        return args