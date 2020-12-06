from mmcv.utils.config import *
import tempfile
import os

def _gettemp_pyfile():
    file_name = tempfile.NamedTemporaryFile().name+'.py'
    return file_name

class PublicConfig(Config):
    """
        This class allow using public config files, example public config file: 
            http://118.69.233.170:8000/coco-person-20/atss_regnetx0.4_fpn_NoNorm_ccp20/atss_regnetx_fpn_ccp20.py
    """
    @staticmethod
    def fromfile(filename):
        cfg_dict, cfg_text = PublicConfig._file2dict(filename)
        return PublicConfig(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def _file2dict(filename):

        print('_file2dict:', filename)
        if 'http' in filename:
            # import pdb; pdb.set_trace()
            temp_filename = _gettemp_pyfile()
            assert filename.endswith('.py')
            result = os.system(f'wget {filename} -O {temp_filename}')
            assert result == 0, f"Cannot download: {filename}"
            # Update filename
            filename = temp_filename
            
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix='.py')
                temp_config_name = osp.basename(temp_config_file.name)
                shutil.copyfile(filename,
                                osp.join(temp_config_dir, temp_config_name))
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules[temp_module_name]
                # close temp file
                temp_config_file.close()
        elif filename.endswith(('.yml', '.yaml', '.json')):
            import mmcv
            cfg_dict = mmcv.load(filename)
        else:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                base_path = osp.join(cfg_dir, f) if not 'http' in f else f
                _cfg_dict, _cfg_text = PublicConfig._file2dict(base_path)
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError('Duplicate key is not allowed among bases')
                base_cfg_dict.update(c)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text