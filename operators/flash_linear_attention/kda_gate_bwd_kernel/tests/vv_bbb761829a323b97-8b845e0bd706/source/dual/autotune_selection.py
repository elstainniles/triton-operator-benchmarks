"""Share the actual original autotuner choice across measurement processes."""
from block_tuning import _safe

def tuner_for(module, kernel_name, autotuner_type):
    """Find the original tuner through the test's explicit local imports.

    Tests often import only a wrapper function or autograd class, leaving the
    named kernel in that object's yb_* module. Restrict traversal to those
    actual imported origins, and reject multiple distinct tuners.
    """
    import sys
    import types
    queue=[module];visited=set();matches={}
    while queue:
        current=queue.pop()
        if id(current) in visited:continue
        visited.add(id(current))
        namespace=vars(current)
        obj=namespace.get(kernel_name);chain=set()
        while obj is not None and id(obj) not in chain:
            chain.add(id(obj))
            if isinstance(obj,autotuner_type):
                matches[id(obj)]=obj
                break
            obj=getattr(obj,'fn',None)
        for value in tuple(namespace.values()):
            if isinstance(value,types.ModuleType):name=value.__name__
            elif isinstance(value,(types.FunctionType,type)):name=getattr(value,'__module__','')
            else:continue
            if name.startswith('yb_') and name[3:].isalnum():
                imported=sys.modules.get(name)
                if imported is not None:queue.append(imported)
    if len(matches)>1:raise ValueError('target autotuner is ambiguous across original imported wrappers')
    return next(iter(matches.values()),None)

def selected(module, kernel_name, autotuner_type):
    tuner = tuner_for(module, kernel_name, autotuner_type)
    if tuner is None:
        return None
    config = getattr(tuner, 'best_config', None)
    if config is None:
        raise ValueError('target autotuner choice is unavailable at launch')
    if getattr(config, 'pre_hook', None) is not None:
        raise ValueError('autotune config with pre_hook cannot be shared safely')
    return _safe(config.all_kwargs())

def apply(module, kernel_name, autotuner_type, selection):
    if selection is None:
        return
    if not isinstance(selection, dict) or not selection:
        raise ValueError('invalid recorded autotune configuration')
    tuner = tuner_for(module, kernel_name, autotuner_type)
    if tuner is None:
        raise ValueError('target autotuner missing in peer process')
    matches = [config for config in tuner.configs
               if _safe(config.all_kwargs()) == selection and getattr(config, 'pre_hook', None) is None]
    if len(matches) != 1:
        raise ValueError('recorded autotune config is absent or ambiguous in peer process')
    tuner.configs = matches
