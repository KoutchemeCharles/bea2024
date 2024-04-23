from string import Formatter

def format_string(info_source, string):
    if hasattr(info_source, string):
        getter = lambda n: getattr(info_source, n) 
    else:
        getter = lambda n: info_source[n]
        
    mapping = {name: getter(name)
               for name in get_format_names(string)}
    
    return  string.format(**mapping)


def get_format_names(string):
    names = [fn for _, fn, _, _ in Formatter().parse(string) 
             if fn is not None]
    return names 


def duplicate(l, n):
    return [val for val in l for _ in range(n)]