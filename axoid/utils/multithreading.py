#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for useful functions related to multi-threading.
Created on Fri Dec 14 16:53:08 2018

@author: nicolas
"""

from multiprocessing import Process, Manager


def run_parallel(*fns):
    """
    Run the called functions in parallel (use lambda keyword if needed).
    
    Parameters
    ----------
    fns : list of callable
        Functions that take no input but can have returns.
    
    Returns
    -------
    returns : dict
        Dictionary containing the returns of each function call. Keys are like
        for lists: integers from 0 to len(fns)-1
       
    Examples
    --------
    `return0, return1 = run_parallel(fn0, fn1)`
    Or with lambdas:
    ```
    return0, return1 = run_parallel(
        lambda: fn0(*args0, **kwargs0),
        lambda: fn1(*args1, **kwargs1)
    )
    ```
    """
    manager = Manager()
    return_dict = manager.dict()
    def fn_return(fn, i, return_dict):
        return_dict[i] = fn()
    
    proc = []
    for i, fn in enumerate(fns):
        p = Process(target=fn_return, args=(fn, i, return_dict))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()
    
    returns = []
    for i in range(len(fns)):
        returns.append(return_dict[i])
    return returns