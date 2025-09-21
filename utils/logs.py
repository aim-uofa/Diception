import torch
import logging
import colorlog

from timeit import default_timer
from datetime import timedelta

import os

logging.getLogger().setLevel(logging.WARNING)

def get_rank():
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

def get_logger(filename=None):
    """
    examples:
        logger = get_logger('try_logging.txt')

        logger.debug("Do something.")
        logger.info("Start print log.")
        logger.warning("Something maybe fail.")
        try:
            raise ValueError()
        except ValueError:
            logger.error("Error", exc_info=True)

        tips:
        DO NOT logger.inf(some big tensors since color may not helpful.)
    """
    logger = logging.getLogger('utils')
    level = logging.DEBUG
    logger.setLevel(level=level)
    # Use propagate to avoid multiple loggings.
    logger.propagate = False
    # Remove %(levelname)s since we have colorlog to represent levelname.
    format_str = '[%(asctime)s <%(filename)s:%(lineno)d> %(funcName)s] %(message)s'

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(level)
    coloredFormatter = colorlog.ColoredFormatter(
        '%(log_color)s' + format_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            # 'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'reg,bg_white',
        }
    )

    streamHandler.setFormatter(coloredFormatter)
    logger.addHandler(streamHandler)

    if filename:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(level)
        formatter = logging.Formatter(format_str)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Fix multiple logging for torch.distributed
    try:
        class UniqueLogger:
            def __init__(self, logger):
                self.logger = logger
                self.local_rank = get_rank()

            def info(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.info(msg, *args, **kwargs)

            def warning(self, msg, *args, **kwargs):
                if self.local_rank == 0:
                    return self.logger.warning(msg, *args, **kwargs)

        logger = UniqueLogger(logger)
    # AssertionError for gpu with no distributed
    # AttributeError for no gpu.
    except Exception:
        pass
    return logger

logger = get_logger()
# logger.info("<logs.py> Renderer Training Logger")
LOGGER = logging.getLogger('__main__')  # this is the global logger



class Meter(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if isinstance(val, (int, float)):
            self.val = val
            if self.sum:
                self.sum += val * n
            else:
                self.sum = val * n
            if self.count:
                self.count += n
            else:
                self.count = n
            self.avg = self.sum / self.count
        elif isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, torch.Tensor):
                    val[k] = v.item()
            # if 'loss_total' in val.keys():
            #     if not math.isnan(val['loss_total']):
            #         # print('it is not nan!!!')
            if self.val:
                for k in val.keys():
                    self.val[k] = val[k]
            else:
                self.val = val
            if self.sum:
                for k in val.keys():
                    if k in self.sum:
                        self.sum[k] = self.sum[k] + val[k] * n
                    else:
                        self.sum[k] = val[k] * n
            else:
                self.sum = {k: val[k] * n for k in val.keys()}
            if self.count:
                for k in val.keys():
                    if k in self.count:
                        self.count[k] = self.count[k] + n
                    else:
                        self.count[k] = n
            else:
                self.count = {k: n for k in val.keys()}
            self.avg = {k: self.sum[k] / self.count[k] for k in self.count.keys()}
        else:
            raise ValueError('Not supported type %s' % type(val))

    def __str__(self):
        if isinstance(self.avg, dict):
            return str({k: "%.4f" % v for k, v in self.avg.items()})
        else:
            return 'Nan'
        

class Timer:
    def __init__(self):
        '''
        t = Timer()
        time.sleep(1)
        print(t.elapse())
        '''
        self.start = default_timer()

    def elapse(self, readable=False):
        seconds = default_timer() - self.start
        if readable:
            seconds = str(timedelta(seconds=seconds))
        return seconds
    
def get_parameters(net: torch.nn.Module):
    
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    fp32_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float32 and p.requires_grad)
    fp16_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float16 and p.requires_grad)
    bf16_trainable_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.bfloat16 and p.requires_grad)
    fp32_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float32 and not p.requires_grad)
    fp16_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.float16 and not p.requires_grad)
    bf16_frozen_params = sum(p.numel() for p in net.parameters() if p.dtype == torch.bfloat16 and not p.requires_grad)
    return {'trainable': trainable_params, 'frozen': frozen_params,
            'trainable_fp32': fp32_trainable_params,
            'trainalbe_fp16': fp16_trainable_params,
            'trainalbe_bf16': bf16_trainable_params,
            'frozen_fp32': fp32_frozen_params, 'frozen_fp16': fp16_frozen_params, 'frozen_bf16': bf16_frozen_params}

