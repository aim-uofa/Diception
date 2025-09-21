import os
import os.path as op

from .logs import logger
import shutil

def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        logger.info('Removing dirname: %s' % os.path.abspath(dirname))
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            raise ValueError('Failed to delete %s because %s' % (dirname, e))

    if not os.path.exists(dirname):
        logger.info('Making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname, exist_ok=True)

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path, exist_ok=True)
            except FileExistsError:
                # Ignore the exception since the directory already exists.
                pass
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise
                
def adaptively_load_state_dict(target, state_dict):
    target_dict = target.state_dict()

    common_dict = {}
    mismatched_keys, unexpected_keys = [], []
    for k, v in state_dict.items():
        if k in target_dict:
            try:
                if v.size() != target_dict[k].size():
                    mismatched_keys.append(k)
                else:
                    common_dict[k] = v
            except Exception as e:
                logger.warning(f'load error for {k} {e}')
                common_dict[k] = v
        else:
            unexpected_keys.append(k)


    # try:
    #     common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
    # except Exception as e:
    #     logger.warning('load error %s', e)
    #     common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

    if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
            target.state_dict()['param_groups'][0]['params']:
        logger.warning('Detected mismatch params, auto adapte state_dict to current')
        common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
    target_dict.update(common_dict)
    target.load_state_dict(target_dict)

    missing_keys = [k for k in target_dict.keys() if k not in common_dict and k not in  mismatched_keys]
    # unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

    if len(unexpected_keys) != 0:
        logger.warning(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(mismatched_keys) != 0:
        logger.warning(
            f"Mismatched shape of weights loaded from state_dict: {mismatched_keys}"
        )
    if len(missing_keys) != 0:
        logger.warning(
            f"Some weights of target are missing in state_dict: {missing_keys}"
        )
    if len(unexpected_keys) == 0 and len(missing_keys) == 0 and len(mismatched_keys) == 0:
        logger.warning("Strictly Loaded state_dict.")
