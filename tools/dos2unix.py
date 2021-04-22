#!/usr/bin/env python

def add_suffix_to_fp(file_path, suffix, path_separator='/'):
    """
    Add a suffix to a file name (not a new file type; e.g. 'file' becomes 'file_name').
    Return new filepath.
    """
    new_file_path = ''
    
    new_file_path = file_path.split(path_separator)
    file_name = new_file_path[-1].split('.')
    file_name[0] += suffix
    file_name = '.'.join(file_name)
    new_file_path[-1] = file_name
    new_file_path = '/'.join(new_file_path)
    
    return new_file_path


def crlf_to_lf(f_in_path, f_out_path= '', suffix='_unix', path_separator='/'):
    """
    convert dos linefeeds (crlf) to unix (lf)
    usage: python dos2unix.py

    Adapted from https://stackoverflow.com/questions/2613800/how-to-convert-dos-windows-newline-crlf-to-unix-newline-lf-in-a-bash-script/19702943#19702943
        Used @anatoly techtonik's solution for unpickling word_data.pkl in email_preprocess.py
        This is a commonly copied/pasted script.
        
    f_in_path: (str) path to input file to be converted
    f_out_path: (optional, str) path to save converted output;
        empty string (default) or None will add suffix to filename in f_in_path,
            using add_suffix_to_fp()
    suffix: (optional, str) to append to end of file name (before file type);
        passed to add_suffix_to_fp()
    path_separator: (optional, str) passed to add_suffix_to_fp() to indicate directory
        separators; default is '/'
        
    return: (str) new_file_path, location of converted file.
        Is the same as f_out_path if that parameter was entered.
        Otherwise, it's f_in_path with suffix added to its filename.
        
    Example: new_file_path = crlf_to_lf(f_in_path=old_file_path)
    """
    
    new_file_path = ''
    
    import sys
#     from pathlib import Path
#     import os
    
#     if len(sys.argv[1:]) != 2:
#         print(sys.argv[:])
#         print('len(sys.argv[1:]) != 2 in crlf_to_lf, will sys.exit(__doc__)')
#         sys.exit(__doc__)

    content = ''
    f_out_size = 0
    
    if f_out_path == '' or f_out_path is None:
        f_out_path = add_suffix_to_fp(file_path=f_in_path, suffix=suffix,
                                      path_separator=path_separator)
    
    with open(f_in_path, 'rb') as f_in:
        content = f_in.read()
    with open(f_out_path, 'wb') as f_out:
        for line in content.splitlines():
            f_out_size += len(line) + 1
            f_out.write(line + str.encode('\n'))

    print(f_in_path, 'saved as', f_out_path, 'in %s bytes.' % (len(content)-f_out_size))
    
    new_file_path = f_out_path
    
    return new_file_path