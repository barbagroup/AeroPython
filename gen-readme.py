#!/usr/bin/env python2

from glob import glob
from urllib import quote
import re

header = '''
## Aerodynamics-Hydrodynamics with Python

"Aerodynamics-Hydrodynamics" (MAE 6226) is taught at the George Washington University by Prof. Lorena Barba for the first time in Spring 2014. These IPython Notebooks are being prepared for this class, with assistance from Barba-group PhD student Olivier Mesnard.

The materials are distributed publicly and openly under a Creative Commons Attribution license, [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)

### List of notebooks:
'''



format_item = '* [{name}]({url})'.format
bb_url = 'github.com/barbagroup/AeroPython/blob/master/{}'.format

def notebooks():
    return glob('lessons/*.ipynb')

def lesson_id(filename):
    return int(re.search('[0-9]+', filename).group())

def lesson_name(filename):
    return filename.split('/')[1].split('.')[0].split('_')[2]

def nb_url(filename):
    raw_url = bb_url(quote(quote(filename)))
    return 'http://nbviewer.ipython.org/urls/{}'.format(raw_url)

def write_readme(nblist, fo):
    fo.write('{}\n'.format(header))
    for nb in nblist:
        name = lesson_name(nb)
        url = nb_url(nb)
        fo.write('{}\n'.format(format_item(name=name, url=url)))

def main():
    nblist = sorted(notebooks(), key=lesson_id)
    with open('README.md', 'w') as fo:
        write_readme(nblist, fo)

if __name__ == '__main__':
    main()
