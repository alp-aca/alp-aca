import colorutils
from bokeh.palettes import Set3_11, Magma256
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

darker_set3 = [(colorutils.Color(hex=c) - (30,30,30) ).hex for c in Set3_11]

newmagma = [(colorutils.Color(hex=c) + (30,30,30)).hex for c in list(reversed(Magma256))[:128+64]] + [(colorutils.Color(hex=c) + (30-int(i*30/32),30-int(i*30/32),30-int(i*30/32))).hex for i, c in enumerate(list(reversed(Magma256))[128+64:128+64+32])]

with open(os.path.join(current_dir, 'palettes.py'), 'w') as f:
    f.write('darker_set3 = ' + str(darker_set3) + '\n')
    f.write('newmagma = ' + str(newmagma) + '\n')