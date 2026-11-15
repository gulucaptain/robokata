"""Bounded CPU prefetch and SDPA; no CUDA reservations or live process patching."""
from queue import Queue, Full, Empty
from threading import Event, Thread

import torch
import torch.nn.functional as F

def sdpa_attention(module,query,key,value,mask,scaling,dropout=0.0,**kwargs):
    # Explicit repeat preserves the native grouped-query head layout on all backends.
    from transformers.models.gemma.modeling_gemma import repeat_kv
    key=repeat_kv(key,module.num_key_value_groups)
    value=repeat_kv(value,module.num_key_value_groups)
    empty_rows=None
    if mask is not None:
        mask=mask[:,:,:,:key.shape[-2]]
        # Invalid/padded query rows have no legal keys and are never consumed by
        # valid tokens. Avoid fused-backend derivatives of finite -max sentinels.
        empty_rows=(mask < -1e20).all(dim=-1,keepdim=True)
        mask=torch.where(empty_rows,0,mask).to(query.dtype)
    output=F.scaled_dot_product_attention(query,key,value,attn_mask=mask,dropout_p=dropout if module.training else 0.0,
                                        is_causal=False,scale=scaling)
    if empty_rows is not None:output=torch.where(empty_rows,0,output)
    return output.transpose(1,2).contiguous(),None

def prefetch(iterable,depth=1):
    """Read ahead without advancing the yielded/saved training cursor."""
    queue=Queue(maxsize=depth);stop=Event();sentinel=object()
    def put(item):
        while not stop.is_set():
            try:queue.put(item,timeout=.1);return True
            except Full:pass
        return False
    def produce():
        try:
            for item in iterable:
                if not put((True,item)):return
        except BaseException as error:
            put((False,error))
        finally:put((True,sentinel))
    thread=Thread(target=produce,name='demo-cache-prefetch',daemon=True);thread.start()
    try:
        while True:
            success,item=queue.get()
            if not success:raise item
            if item is sentinel:return
            yield item
    finally:
        stop.set();thread.join(timeout=5)

