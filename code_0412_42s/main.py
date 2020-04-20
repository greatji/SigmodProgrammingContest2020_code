import os
import re
import json
import numpy as np
import math
import swifter
import pandas as pd
import itertools
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import multiprocessing as mp
from multiprocessing import Pool



brands_for_blocking = set(['cannon','canon','eos','powershot',
 'nikon','coolpix','nicon',
 'bushnell',
 'sandisk',
 'lexar',
 'carbose',
 'brinno',
 'sony','effio',
 'kavass',
 'olympus','olympul','olymus',
 'pextax','pentax',
 'digital blue',
 'lytro',
 'lowepro',
 'fuifilm','fujifilm','figi','fuji','fugi','fujufilm', 'fijifilm','fiji film','finepix','fugi film', 'fugifilm','fuijifilm',
 'philip','philips',
 'yashica',
 'haier',
 'konika','konica','minolta',
 'vivicam','vivitar','sakar',
 'dongjia',
 'ricoh',
 'general electric','ge ',' ge ',
 'panasonic','lumix',
 'samsung',
 'kodak','kodax',
 'casio',
 'toshiba',
 'sanyo',
 'hp ','hewlett packard',
 'hasselblad',
 'benq',
 'coleman',
 'polaroid',
 'cobra',
 'sharp',
 'svp',
 'vistaquest',
 'aiptek',
 'sealife',
 'enxun',
 'intova',
 'mustek',
 'vtech',
 'dxg',
 'argus',
 'insignia',
 'superheadz',
 'disney',
 'jvc',
 'emerson',
 'croco',
 'bell+howell','b+w','bell howell','b h ',
 'contax',
 'easypix',
 'sylvania',
 'barbie',
 'minox',
 'wespro',
 'dji',
 'lowrance',
 'epson',
 'hikvision','hikivision',
 'dahua',
 'leica',
 'sigma',
 'gopro','go pro',
 'tamron',
 'vizio',
 'neopine',
 'absee',
 'samyang',
 'wopson',
 'garmin',
 'yourdeal',
 'drift',
 'rollei',
 'blackmagic','bmpcc',
 'asus',
 'nokia',
 'vibe ',
 'lg ',
 'hello kitty',
 'kinon',
 'aquapix',
 'apple','iphone',
 'keedox',
 'lego',
 'logitech',
 'crayola',
])
brands_for_blocking = sorted(brands_for_blocking, key=lambda x: len(x), reverse=True)
print (brands_for_blocking)


def combine(x):
    if type(x) == list:
        return ';'.join(x)
    else:
        return str(x)

def load_dataset(dataset_path):
    #products = {}
    ids = []
    titles = []
    models = []
    brands = []
    mpns = []
    webs = [f for f in listdir(dataset_path)]
    for w in webs:
        #products[w] = {}
        items = [f for f in listdir(dataset_path+'/'+w)]
        for it in items:
            with open(dataset_path+'/'+w+'/'+it, 'r') as f:
                data = json.load(f)
                titles.append(data['<page title>'].lower())
                ids.append(w + '//' +it[0:-5])
                local_models = []
                if 'model' in data:
                    local_models.append(combine(data['model']).lower())
                if 'model id' in data:
                    local_models.append(combine(data['model id']).lower())
                if 'family line' in data:
                    local_models.append(combine(data['family line']).lower())
                if 'model no' in data:
                    local_models.append(combine(data['model no']).lower())
                if 'series' in data:
                    local_models.append(combine(data['series']).lower())
                models.append(';'.join(local_models))
                
                local_brands = []
                if 'manufacturer' in data:
                    local_brands.append(combine(data['manufacturer']).lower())
                if 'brand' in data:
                    local_brands.append(combine(data['brand']).lower())
                if 'manuf no' in data:
                    local_brands.append(combine(data['manuf no']).lower())
                brands.append(';'.join(local_brands))
                
                if 'mpn' in data:
                    mpns.append(combine(data['mpn']).lower())
                else:
                    mpns.append('')
    return pd.DataFrame(data={'spec_id':ids, 'page_title':titles, 'camera_brand': brands, 'camera_model': models, 'camera_mpn': mpns})

def compute_tf_idf(dataset_path):
    sentences = []
    webs = [f for f in listdir(dataset_path)]
    for w in webs:
        items = [f for f in listdir(dataset_path+'/'+w)]
        for it in items:
            with open(dataset_path+'/'+w+'/'+it, 'r') as f:
                sentences.append(f.read().lower().split(' \n:}{"][\''))
    total_words_num = 0
    word_freqs = {}
    word_df = {}
    word_tfidf = {}
    for s in sentences:
#        s_split = s.lower().split(' ')
        s_split = s
        for ss in s_split:
            if ss == '':
                continue
            total_words_num += 1
            if ss in word_freqs:
                word_freqs[ss] += 1
            else:
                word_freqs[ss] = 1
        for ss in set(s_split):
            if ss == '':
                continue
            if ss in word_df:
                word_df[ss] += 1
            else:
                word_df[ss] = 1
    for k in word_freqs.keys():
        word_freqs[k] /= float(total_words_num)
        word_df[k] = math.log(len(sentences) / float(word_df[k]))
        word_tfidf[k] = word_freqs[k] * word_df[k]
    return word_tfidf

def filter_accessary(words, keyword, all_brands):
    try:
        pos = words.index(keyword)
        return len(set(words[0:pos]) & all_brands) > 0
    except:
        return False

def is_type(words, idx, current_num, current_num2, all_brands, words_we_need, info_not_model, some_units, exist_number_match, only_number_match):
    if len(words[idx]) > 20:
        return False
    if words[idx] in info_not_model:
        return False
    stop_match = re.findall(r'["$\']+', words[idx])
    if stop_match: # money, size etc.
        return False
    letter_match = re.findall(r'[a-z]+', words[idx])
    if not letter_match:
        if idx < len(words)-1 and (words[idx+1] in some_units):
            return False
        if len(set(words[max(0, idx-2):idx]) & all_brands) == 0 and words[idx-1] not in ['rebel', 'finepix'] or (words[idx-2] == 'eos' and words[idx-1] != 'rebel') or words[idx-2] == 'finepix':
            if current_num >= 1 and words[idx-1] != 'mark':
                return False
            if idx < len(words)-1 and words[idx+1].endswith(('mm','mp','inch','megapixel','mega')):
                return False
            if only_number_match and idx < len(words)-2 and re.findall(r'^[0-9]+$', words[idx+1]):
                if float(words[idx+1]) < 10 and (words[idx+2] in ['millions', 'mg', 'mp', 'inch', 'mega', 'megapixel']): # still a pixel, missing point
                    return False
                if float(words[idx]) < 300 and float(words[idx+1]) < 1000 and words[idx+2] in ['meters', 'mm', 'lens']: # still a len, missing point
                    return False
            if only_number_match and len(set(words[max(0, idx-4):idx]) & all_brands) == 0:
                return False
            if re.findall(r'[0-9]+\-[0-9]+', words[idx]):
                return False
        else:
            if only_number_match and idx > 0 and words[idx-1] in ['sony'] and words[idx+1].endswith('mp'):
                return False
            if only_number_match and idx < len(words)-2 and re.findall(r'^[0-9]+$', words[idx+1]):
                if float(words[idx]) >= 10 and float(words[idx+1]) < 10 and (words[idx+2] in ['millions', 'mg', 'mp', 'inch', 'mega', 'megapixel']): # still a pixel, missing point
                    return False
                if float(words[idx]) < 300 and float(words[idx+1]) < 1000 and words[idx+2] in ['meters', 'mm', 'lens']: # still a len, missing point
                    return False
                if float(words[idx+1]) == 0:
                    return False
            if current_num >= 1 and idx < len(words)-1 and (words[idx+1] in some_units or words[idx+1].endswith(('mm','mp','inch','megapixel','mega')) and float(''.join(re.findall(r'\d+', words[idx+1]))) < 10):
                return False
            if idx < len(words)-1 and (words[idx+1] in some_units or words[idx+1].endswith(('mg','mm','mp','inch','megapixel','mega')) and float(''.join(re.findall(r'\d+', words[idx+1]))) == 0):
                return False
        if words[idx] in ['2015', '2014', '2013', '2012', '2011']:
            return False
    else:
        if words[idx] in ['i', 'ii', 'iii', 'iv'] and 'camera' not in ''.join(words[max(0,idx-2):idx]) and (idx > 0 and words[idx-1] not in ['mark', 'mk']) and ('af-s' in words[0:idx] or (idx < len(words)-1 and words[idx+1] == 'lens') or len(set(words[max(0, idx-4):idx]) & all_brands) == 0):
            return False
        if words[idx] in ['iis', 'z', 'm', 'x', 'df', 'f1', 'gr', 'px', 'slv', 'xs', 'k-x', 'k-r', 'xs-pro', 'v', 'q'] and (len(set(words[max(0, idx-4):idx]) & all_brands) == 0 or current_num > 0):
            return False
        if words[idx] in ['3gp', 'bin2']:
            return False
        if words[idx] in ['nx'] and words[idx+1] != 'mini':
            return False
        if words[idx].endswith(('megpixel', 'megapixel','megapixels','year','lens','colors','meter','digital','mega','inch','cmos')):
            return False
        if (re.findall(r'\d*\.?\d*\-?\d*\.?\d*mm$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*mp$', words[idx]) or
           re.findall(r'^\d+\.?\d*\-?\d*\.?\d*m$', words[idx]) or
           re.findall(r'^\d+\.?\d*\-?\d*\.?\d*cm$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*pcs$', words[idx]) or 
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*mb/s$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*pc$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*gb$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*g$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*fps$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*yr$', words[idx]) or
           re.findall(r'^s\d+\.?\d*\-\d+\.?\d*$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*g?hz$', words[idx]) or
           re.findall(r'\d+tvl$', words[idx]) or
           re.findall(r'ip\d*$', words[idx])):
            return False
        if words[idx].startswith('sel') or words[idx].startswith('sal') or words[idx].endswith('batteries'):
            return False
        if re.findall(r'[0-9]+\-[0-9]+\/[0-9]+-[0-9]+$', words[idx]):
            return False
        if re.findall(r'^f\/[0-9]+\.?[0-9]*', words[idx]):
            return False
        if re.findall(r'^f\/?[0-9]+\.?[0-9]*$', words[idx]) and len(set(words[max(0, idx-2):idx]) & all_brands) == 0:
            return False
        if re.findall(r'^f\/?[0-9]+\.?[0-9]*\-[0-9]+\.?[0-9]*', words[idx]):
            return False
        if re.findall(r'^[0-9]+\.?[0-9]*x$', words[idx]):
            return False
    return True

def tfidf(x, word_tfidf):
    if x in word_tfidf:
        return word_tfidf[x]
    else:
        return 0.0

def extract_type(row, word_tfidf, stopped_words, all_brands, words_we_need, info_not_model, some_units, fuji_prefix, fuji_suffix, powershot_suffix, powershot_prefix, eos_suffix, eos_prefix, canon_other_suffix, canon_other_prefix, nikon_suffix, nikon_prefix, olympus_suffix, olympus_prefix, panasonic_prefix):
    words = re.split(r'[, ()~]', str(row['page_title']))
    brand = str(row['blocking_key'])
    words = [x.strip(',;-().') for x in words]
    types = []
    version = None
    keep_num = 2
    generations = {'i':'1','ii':'2','iii':'3','iv':'4'}
    generations2 = {'i':'one','ii':'two','iii':'three','iv':'four'}
    for idx, w in enumerate(words):
        if w in types:
            continue
        number_re = re.compile('\d')
        exist_number_match = number_re.search(w)
        if not exist_number_match and w not in words_we_need:
            continue
        only_number_match = re.findall(r'^[0-9]+$', w)
        if only_number_match and (len(set(words[0:idx]) & all_brands) == 0 or len(w) > 4):
            continue
        current_num2 = len(types)
        if current_num2 > 0:
            if 'comparison' in words[0:idx] and filter_accessary(words, 'comparison', all_brands): # Kit is not important
                break
            if 'kit' in words[0:idx] and filter_accessary(words, 'kit', all_brands): # Kit is not important
                break
            if '&' in words[0:idx] and filter_accessary(words, '&', all_brands): # Kit is not important
                break
            if '+' in words[0:idx] and filter_accessary(words, '+', all_brands): # Kit is not important
                break
            if 'w/' in words[0:idx] and filter_accessary(words, 'w/', all_brands): # Kit is not important
                break
            if 'w' in words[0:idx] and filter_accessary(words, 'w', all_brands): # Kit is not important
                break
            if 'with' in words[0:idx] and filter_accessary(words, 'with', all_brands): # Kit is not important
                break
        if is_type(words, idx, len(re.findall(r'\d+', ''.join(types))), current_num2, all_brands, words_we_need, info_not_model, some_units, exist_number_match, only_number_match):
            if w in ['i','ii','iii','iv']:
                if version is None:
                    version = generations[w]
                if len(types) == 0:
                    types.append(generations2[w])
            elif w in ['v2'] and 'coolpix' in words:
                pass
            elif w in ['a']:
                if idx > 0 and words[idx-1] == 'coolpix':
                    types.append('a')
            elif w in ['1d', 'eos1d', 'eos-1d'] and brand == 'cannon':
                if words[idx+1] in ['s', 'x']:
                    types.append('1d'+words[idx+1])
                else:
                    types.append('1dk')
            elif w in ['markii', 'mkii']:
                if version is None:
                    version = '2'
            elif w in ['markiii', 'mkiii']:
                if version is None:
                    version = '3'
            elif w in ['y1'] and brand == 'kodak':
                pass
            elif w == 'm' and idx > 1 and words[idx-1] == 'eos':
                types.append('eosm')
            elif w == 'x' and idx > 1 and words[idx-1] == 'k':
                types.append('kx')
            elif w == 'r' and idx > 1 and words[idx-1] == 'k':
                types.append('kr')
            elif re.findall(r'^\d+$', w) and idx > 0 and words[idx-1] == 'mark':
                if version is None:
                    version = words[idx]
            elif re.findall(r'^\d+$', w) and idx > 0 and re.findall(r'\d+', words[idx-1]):
                continue
            elif brand == 'fuji':
                if len(''.join(re.findall(r'\d+', w))) > 5:
                    continue
                sentence = ''.join(words[0:min(10,len(words))])
                if w != 'x':
                    if 'instax' in sentence:
                        if idx > 0 and ''.join(words[idx-1].split('-')) in ['mini', 'wide']:
                            w = ''.join(words[idx-1].split('-'))+w
                    else:
                        if idx < len(words)-1 and words[idx+1] in fuji_suffix:
                            w = w+words[idx+1]
                        if idx > 0 and ''.join(words[idx-1].split('-')) in fuji_prefix:
                            w = ''.join(words[idx-1].split('-'))+w
                    if not re.findall(r'^\d+$', w) or len(w) >= 2:
                        types.append(w)
            elif brand == 'cannon':
                if len(''.join(re.findall(r'\d+', w))) > 5 or w in ['p1']:
                    continue
                sentence = ''.join(words[0:min(10,len(words))])
                if 'powershot' in sentence or 'ixus' in sentence or 'elph' in sentence or 'ixy' in sentence:
                    if idx < len(words)-1 and words[idx+1] in powershot_suffix:
                        w = w+words[idx+1]
                    if idx > 0 and words[idx-1] in powershot_prefix:
                        w = words[idx-1]+w
                elif 'eos' in sentence:
                    if idx < len(words)-1 and words[idx+1] in eos_suffix:
                        w = w+words[idx+1]
                    if idx > 0 and words[idx-1] in eos_prefix:
                        w = words[idx-1]+w
                else:
                    if idx < len(words)-1 and words[idx+1] in canon_other_suffix:
                        w = w+words[idx+1]
                    if idx > 0 and words[idx-1] in canon_other_prefix:
                        w = words[idx-1]+w
                if not re.findall(r'^\d+$', w) or len(w) >= 2:
                    types.append(w)
            elif brand == 'nikon':
                if len(''.join(re.findall(r'\d+', w))) > 5:
                    continue
                if idx < len(words)-1 and words[idx+1] in nikon_suffix:
                    w = w+words[idx+1]
                if idx > 0 and words[idx-1] in nikon_prefix:
                    w = words[idx-1]+w
                if not re.findall(r'^\d+$', w) or len(w) >= 2:
                    types.append(w)
            elif brand == 'olympus':
                if len(''.join(re.findall(r'\d+', w))) > 5 or w in ['x', 'v', 'r']:
                    continue
                sentence = ''.join(words[0:min(10,len(words))])
                if idx < len(words)-1 and words[idx+1] in olympus_suffix:
                    w = w+words[idx+1]
                if idx > 0 and words[idx-1] in ['c' or 'camedia']:
                    w = 'c'+w
                elif idx > 0 and ''.join(words[idx-1].split('-')) in olympus_prefix:
                    w = ''.join(words[idx-1].split('-'))+w
                elif idx > 0 and ('stylus' in words[idx-1] or 'miu' in words[idx-1] or 'µ' in words[idx-1] or 'mju' in words[idx-1]):
                    w = 'µ'+w
                types.append(w)
            elif brand == 'panasonic':
                if w in ['4k']:
                    continue
                if idx > 0 and ''.join(words[idx-1].split('-')) in panasonic_prefix:
                    w = ''.join(words[idx-1].split('-'))+w
                types.append(w)
            elif brand == 'sony':
                if w in ['r']:
                    continue
                if w == 'effio':
                    if words[idx+1] in ['p', 's', 'e', 'v', 'a']:
                        w = w+words[idx+1]
                elif idx < len(words)-1 and words[idx+1] in ['s', 'r']:
                    w = w + words[idx+1]
                if idx > 0 and ''.join(words[idx-1].split('-')) in ['cd','alpha','t','fd','dsc','nex','slt','s','a']:
                    w = ''.join(words[idx-1].split('-'))+w
                elif idx > 0 and ''.join(words[idx-1].split('-')) in ['dslra','alpha', 'ilce', 'ilca']:
                    w = 'a'+w
                if not re.findall(r'^\d+$', w):
                    types.append(w)
            else:
                if re.findall(r'^\d+$', w):
                    if idx > 0 and words[idx-1] in all_brands:
                        types.append(w)
                    elif idx > 1 and words[idx-2] in all_brands:
                        if not re.findall(r'\d+', words[idx-1]) and words[idx-1] not in ['camera', 'digital', 'slr', 'series']:
                            types.append(words[idx-1] + '-' + w)
                    elif idx > 2 and words[idx-3] in all_brands:
                        if not re.findall(r'\d+', words[idx-1]) and words[idx-1] not in ['camera', 'digital', 'slr', 'series']:
                            types.append(words[idx-1] + '-' + w)
                        if not re.findall(r'\d+', words[idx-2]) and words[idx-2] not in ['camera', 'digital', 'slr', 'series']:
                            types.append(words[idx-2] + '-' + w)
                        keep_num += 1
                    elif idx > 3 and words[idx-4] in all_brands:
                        if not re.findall(r'\d+', words[idx-1]) and words[idx-1] not in ['camera', 'digital', 'slr', 'series']:
                            types.append(words[idx-1] + '-' + w)
                        if not re.findall(r'\d+', words[idx-2]) and words[idx-2] not in ['camera', 'digital', 'slr', 'series']:
                            types.append(words[idx-2] + '-' + w)
                        if not re.findall(r'\d+', words[idx-3]) and words[idx-3] not in ['camera', 'digital', 'slr', 'series']:
                            types.append(words[idx-3] + '-' + w)
                        keep_num += 2
                    else:
                        types.append(w)
                else:
                    types.append(w)
    types = types[0:min(keep_num, len(types))]
    if len(types) == 0 and not pd.isnull(row['camera_model']):
        models = re.split(r'[;/]', str(row['camera_model']))
        mmmmms = [''.join(m.split(' ')) for m in models]
        models = set(mmmmms) - stopped_words
        models = sorted(models, key=lambda x: tfidf(x, word_tfidf), reverse=False)
        #models = list(models)
        if len(models) > 0:
            types = set(types) | set(models[0:1])
    if len(types) == 0 and not pd.isnull(row['camera_mpn']):
        types.append(str(row['camera_mpn']))
    types = set([''.join(t.split('-')) for t in types]) - stopped_words
    types = sorted(types, key=lambda x: tfidf(x, word_tfidf), reverse=False)
    #types = list(types)
    types = types[0:min(len(types), keep_num)]
    for idx, t in enumerate(types):
        if t.endswith('/b'):
            types[idx] = types[idx][0:-2]
        elif 'rx100' in t and t.endswith('b'):
            types[idx] = types[idx][0:-1]
        if version is None:
            if t.endswith('iii'):
                version = '3'
            elif t.endswith('ii'):
                version = '2'
            elif t.endswith('m3'):
                version = '3'
            elif t.endswith('m2'):
                version = '2'
    if version is None:
        version = ''
    return ';'.join(types), version

def extract_type_number(row):
    all_types = str(row['type']).split(';')
    numbers = []
    for t in all_types:
        num = re.findall(r'\d+', str(t))
        if len(num) > 0:
            numbers.append(''.join(num))
    return ';'.join(numbers)

def extract_synonyms(file_path):
    with open(file_path, 'rt') as myfile:
        data = myfile.read().replace("<br>", '\n')
    df = pd.read_html(data)[0]
    synonyms = {}
    for v in df[df['Model'].str.contains('Rebel')]['Model'].values:
        synonym = []
        for tok in v.lower().split(' '):
            if tok not in ['digital', 'rebel']:
                synonym.append(tok)
        for x in synonym:
            for y in synonym:
                if x != y:
                    synonyms[x] = y
    return synonyms

def some_extra_rule(row, mapping):
    all_types = str(row['type']).split(';')
    added = []
    for i in range(len(all_types)):
        if str(row['blocking_key']) == 'nikon':
            if 'cool' in str(row['page_title']):
                all_types[i] = all_types[i]+'coolpix'
        elif str(row['blocking_key']) == 'olympus':
            if 'stylus' in all_types[i]:
                all_types[i].replace('stylus', 'µ')
            elif 'mju' in all_types[i]:
                all_types[i].replace('mju', 'µ')
        elif all_types[i] in ['rebelt3']:
            all_types[i] = 't3'
        elif 'ixus' in all_types[i]:
            all_types[i] = all_types[i].replace('ixus', 'ixu')
        elif all_types[i] in ['7k', '7'] and str(row['blocking_key']) == 'sony':
            if 'nex' in row['page_title']:
                all_types[i] = 'nex7k'
            elif 'ilce' in row['page_title']:
                all_types[i] = 'a7k'
            elif 'alpha' in row['page_title']:
                all_types[i] = 'a7k'
        elif all_types[i] in ['a7', 'ilce7', 'ilce7k', 'alpha7', 'alpha7k'] and str(row['blocking_key']) == 'sony':
            all_types[i] = 'a7k'
        elif all_types[i].endswith('nex7') and str(row['blocking_key']) == 'sony':
            all_types[i] = 'nex7k'
        elif all_types[i] == 'nex3':
            all_types[i] = 'nex3k'
        elif all_types[i] == 'nex5':
            all_types[i] = 'nex5k'
        elif re.findall(r'\d+m$', all_types[i]):
            all_types[i] = all_types[i][0:-1]
        elif 'rx100' in all_types[i]:
            if all_types[i].endswith('iii'):
                all_types[i] = all_types[i][0:-3]
            elif all_types[i].endswith('ii'):
                all_types[i] = all_types[i][0:-2]
            elif all_types[i].endswith('m3'):
                all_types[i] = all_types[i][0:-2]
            elif all_types[i].endswith('m2'):
                all_types[i] = all_types[i][0:-2]
        if str(row['blocking_key']) == 'cannon' and all_types[i] in mapping:
            added.append(mapping[all_types[i]])
    return ';'.join(all_types+added)

def isVersionNeeded(row):
    for t in str(row['type']).split(';'):
        if row['blocking_key'] == 'cannon':
            if (t.startswith('xt')
                or t.startswith('xs')
                or t.startswith('g1x')
                or t.startswith('5d')
                or t.startswith('1d')
                or t.startswith('7d')
                or t.startswith('t3')
                or t.startswith('t5')
                or t.startswith('t1')
                or t.startswith('t2')
                or t.startswith('t4')):
                return 1
        elif row['blocking_key'] == 'sony':
            if ('rx1' in t
                or t.startswith('a77')):
                return 1
        elif row['blocking_key'] == 'nikon':
            return 1
        elif row['blocking_key'] == 'fuji':
            if t.startswith('x100'):
                return 1
    return 0

def extract_information(structured_data):
    word_tfidf = compute_tf_idf(dataset_path)
    brands_with_some_series = {'kitty', 'dsc', 'electric', 'cyber', 'ixus', 'pov', 'sonydigital', 'howell', 'stylus', 'go', 'blue', 'hp', 'vpc', 'elph'} | set(brands_for_blocking)
    stopped_words = {'cybershot'} | brands_with_some_series
    mapping = extract_synonyms('canon_eos.html')
    sony_monitor = {'effio','effioe','effioa','effiop','effio-a','effio-v','effios','effiov','effio-s','effio-e','effio-p'}
    version_info = {'i','ii','iii','vi','iv','markii', 'markiii', 'mkii', 'mkiii'}
    words_pure_letter_model = {'n', 'r', 'xsi', 'z', 'a', 'xti', 'xt', 'eos-m', 'eosm', 'iis','slv', 'df', 'q', 'm', 'x', 'nx', 'xs', 'k-x', 'k-r', 'xs-pro', 'v', 'gr', 'px'}
    words_we_need = sony_monitor | version_info | words_pure_letter_model
    info_not_model = {'960h','m43', '1080p', '960p', '720p', '360p', 'h.264', 'p2p', 'ip66', 'mpeg4', '1/3'}
    some_units = {'tvl', 'meapixels', 'mg', 'batteries', 'gb', 'lens', 'meters', 'mm', 'mp', 'inch', 'megapixel', 'megapixels', 'mega', 'pack', 'millions'}
    print (mapping)
    fuji_prefix = {'ds', 'mx', 'jx', 'av', 'ax', 'hs', '3d', 's', 'f', 'a', 'e', 't', 'j', 'z', 'xp', 'x', 'pro', 'xt', 'xe', 'xa', 'xpro'}
    fuji_suffix = {'exr', 'f', 'v', 'pro', 'zoom'}
    powershot_suffix = {'zoom', 'is', 'x', 'a', 's', 'f', 'hs'}
    powershot_prefix = {'sx', 'ixy','pro', 'a', 'g', 's', 'sd', 'ixus', 'elph'}
    eos_suffix = {'d', 'ds', 'dx', 'dsr', 'da', 'v'}
    eos_prefix = {'m', 'ds', 'kiss'}
    canon_other_suffix = {'d', 'ds', 'dx', 'dsr', 'da', 'v'}
    canon_other_prefix = {'m', 'ds', 'sd', 'kiss'}
    nikon_suffix = {'a', 's', 'as', 't', 'hp', 'p', 'h', 'af', 'm', 'x'}
    nikon_prefix = {'s', 'aw', 'd', 'f', 'fa', 'fg', 'fe', 'fm', 'v', 'j', 'z'}
    olympus_suffix = {'rs', 'wp', 'sw', 'ee', 'x'}
    olympus_prefix = {'xz', 'vh','t','vr','vg','sz', 'x', 'ir', 'sp', 'd', 'e', 'f', 'fe', 'p', 'ep', 'tough', 'tg', 'e', 'em'}
    panasonic_prefix = {'dmc', 'f','g','fx','fz','l','ls','lx','lz','ts','tz','zs'}
    structured_data['type'], structured_data['version'] = zip(*structured_data.swifter.apply(lambda row: extract_type(row, word_tfidf, stopped_words, brands_with_some_series, words_we_need, info_not_model, some_units, fuji_prefix, fuji_suffix, powershot_suffix, powershot_prefix, eos_suffix, eos_prefix, canon_other_suffix, canon_other_prefix, nikon_suffix, nikon_prefix, olympus_suffix, olympus_prefix, panasonic_prefix), axis=1))
    structured_data['type'] = structured_data.swifter.apply(lambda row: some_extra_rule(row, mapping), axis=1)
    structured_data['type_number'] = structured_data.swifter.apply(lambda row: extract_type_number(row), axis=1)
    structured_data['need_version'] = structured_data.swifter.apply(lambda row: isVersionNeeded(row), axis=1)
    return structured_data

def get_blocking_keys(row):
    page_title = str(row['camera_brand']) + ' ' + str(row['page_title'])
    positions = []
    for brand in brands_for_blocking:
        pos = page_title.find(brand)
        if pos >= 0 and not (brand == 'ge ' and pos > 0):
            positions.append((brand, pos))
        if len(positions) >= 3:
            break
    if len(positions) > 0:
        return sorted(positions, key=lambda x:x[1])[0][0]
    else:
        return ''

def compute_blocking(df):
    """Function used to compute blocks before the matching phase

    Gets a set of blocking keys and assigns to each specification the first blocking key that will match in the
    corresponding page title.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing specifications and page titles

    Returns:
        df (pd.DataFrame): The Pandas DataFrame containing specifications, page titles and blocking keys
    """
    print('>>> Computing blocking...')
    df['blocking_key'] = df.swifter.apply(lambda row: get_blocking_keys(row), axis=1)
    return df

def get_block_pairs_df(df, pool):
    """Function used to get a Pandas DataFrame containing pairs of specifications based on the blocking keys

    Creates a Pandas DataFrame where each row is a pair of specifications. It will create one row for every possible
    pair of specifications inside a block.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing specifications, page titles and blocking keys

    Returns:
        pairs_df (pd.DataFrame): A Pandas DataFrame containing pairs of specifications
    """
    print('>>> Creating pairs dataframe...\n')
    data_dict = {}
    for row in df.itertuples():
        idx = getattr(row, 'Index')
        types = getattr(row, 'type').split(';')
        type_text = getattr(row, 'type')
        type_number_pad = [''.join(re.findall(r'\d+', x)) for x in types]
        type_number = getattr(row, 'type_number').split(';')
        title = getattr(row, 'page_title').split(' ')
        blocking_key = getattr(row, 'blocking_key')
        need_version = getattr(row, 'need_version')
        version = str(getattr(row, 'version'))
        spec_id = getattr(row, 'spec_id')
        data_dict[idx] = [types, type_text, type_number_pad, type_number, title, blocking_key, need_version, version, spec_id]
    grouped_df = df[df['blocking_key'] != ''].groupby('blocking_key')
    index_pairs = []
    for _, block in grouped_df:
        block_indexes = list(block.index)
        index_pairs.extend(list(itertools.combinations(block_indexes, 2)))
    pairs = []
    #cores = mp.cpu_count()
    result=[]
    avg = int(len(index_pairs) / num_process)
    for i  in range(num_process):
       if i < num_process-1:
           result.append(pool.apply_async(entity_matching, args=(index_pairs[i*avg:(i+1)*avg], data_dict))) 
       else:    
           result.append(pool.apply_async(entity_matching, args=(index_pairs[i*avg:], data_dict))) 
    #pool.join()
    for i in result:
        pairs.append(i.get())
    final = pd.concat(pairs, axis=0)
    print (final.head(5))
    return final
    
def entity_matching(index_pairs, data_dict):
    left_spec_ids = []
    right_spec_ids = []
    for index_pair in index_pairs:
        left_index, right_index = index_pair
        left_content = data_dict[left_index]
        right_content = data_dict[right_index]
        left_type = left_content[0]
        right_type = right_content[0]
        left_type_number = left_content[3]
        right_type_number = right_content[3]
        if len(set(left_type+left_type_number) & set(right_type+right_type_number)) > 0:
            if pair_matching(left_content, right_content) == 1:
                left_spec_id = left_content[-1]
                right_spec_id = right_content[-1]
                left_spec_ids.append(left_spec_id)
                right_spec_ids.append(right_spec_id)
    pairs_df = pd.DataFrame(data={'left_spec_id': left_spec_ids, 'right_spec_id': right_spec_ids})
    return pairs_df
# data_dict[idx] = [types, type_text, type_number_pad, type_number, title, blocking_key, need_version, version, spec_id]
def pair_matching(left_content, right_content):
    if len(left_content[1]) == 0 or len(right_content[1]) == 0:
        left_title = left_content[4]
        right_title = right_content[4]
        if len(left_title) == 0 or len(right_title) == 0:
            return 0
        if len(set(left_title) & set(right_title)) / float(len(set(left_title) | set(right_title))) > 0.9:
            return 1
        return 0
    left_type = left_content[0]
    right_type = right_content[0]
    left_number = left_content[2]
    right_number = right_content[2]
    brand_left = left_content[5]
    brand_right = right_content[5]
    version_left = left_content[7]
    version_right = right_content[7]
    need_version = (left_content[6] + right_content[6] > 0)
    for t in range(len(right_type)):
        if left_number[0] == right_number[t] and len(left_number[0]) > 0 and ('1' not in [left_type[0], right_type[t]]) and (left_type[0] in right_type[t] or right_type[t] in left_type[0]):
            if need_version:
                if version_left == version_right and (len(version_left) != 0 or left_type[0][-1] == right_type[t][-1]):
                    return 1
            else:
                return 1
        if left_type[0] == right_type[t] and version_left == version_right:
            return 1
    for t in range(len(left_type)):
        if left_number[t] == right_number[0] and len(right_number[0]) > 0 and ('1' not in [left_type[t], right_type[0]]) and (left_type[t] in right_type[0] or right_type[0] in left_type[t]):
            if need_version:
                if version_left == version_right and (len(version_left) != 0 or left_type[t][-1] == right_type[0][-1]):
                    return 1
            else:
                return 1
        if right_type[0] == left_type[t] and version_left == version_right:
            return 1
#    if len(left_type) > 1 and len(right_type) > 1 and len(right_type[1]) > 0 and left_type[1] == right_type[1] and version_left == version_right and re.findall(r'[a-z]+', right_type[1]):
    if len(left_type) > 1 and len(right_type) > 1 and left_number[1] == right_number[1] and len(right_type[1]) > 1 and len(left_type[1]) > 1 and (left_type[1] in right_type[1] or right_type[1] in left_type[1]) and re.findall(r'[a-z]+', right_type[1]) and re.findall(r'[a-z]+', left_type[1]):
        if need_version:
            if version_left == version_right and (len(version_left) != 0 or left_type[1][-1] == right_type[1][-1]):
                return 1
        else:
            return 1
    if brand_left == 'sony' and len(set(['3000','5000','6000']) & set(left_number) & set(left_number)) > 0:
        return 1
    return 0

import sys, getopt

if __name__ == '__main__':
    
    dataset_path = '/home/sunji/EM_sigmod/2013_camera_specs'
    output_path = '/home/sunji/EM_sigmod/quickstart_package/candidate_pairs_key.csv'   

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('main.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            dataset_path = arg
        elif opt in ("-o", "--ofile"):
            output_path = arg

    cores = 4
    num_process = 12
    pool = Pool(processes=cores)

    dataset_df = load_dataset(dataset_path)

    print ('>>> Dataset Loading Completed')
    dataset_df = compute_blocking(dataset_df)
    dataset_df.loc[dataset_df['blocking_key'].isin(['hikvision','hikivision']), 'blocking_key'] ='hikvision'
    dataset_df.loc[dataset_df['blocking_key'].isin(['fujifilm','figi','fuifilm','fuji','fugi','fujufilm', 'fijifilm','fiji film','finepix','fugi film', 'fugifilm','fuijifilm']), 'blocking_key'] ='fuji'
    dataset_df.loc[dataset_df['blocking_key'].isin(['bell+howell','b+w','bell howell','b h ']), 'blocking_key'] ='bell+howell'
    dataset_df.loc[dataset_df['blocking_key'].isin(['panasonic','lumix']), 'blocking_key'] ='panasonic'
    dataset_df.loc[dataset_df['blocking_key'].isin(['cannon','canon','eos','powershot']), 'blocking_key'] = 'cannon'
    dataset_df.loc[dataset_df['blocking_key'].isin(['olympus','olympul','olymus']), 'blocking_key'] = 'olympus'
    dataset_df.loc[dataset_df['blocking_key'].isin(['nikon','coolpix','nicon']), 'blocking_key'] = 'nikon'
    dataset_df.loc[dataset_df['blocking_key'].isin(['gopro','go pro']), 'blocking_key'] = 'gopro'
    dataset_df.loc[dataset_df['blocking_key'].isin(['philip','philips']), 'blocking_key'] = 'philips'
    dataset_df.loc[dataset_df['blocking_key'].isin(['pextax','pentax']), 'blocking_key'] = 'pentax'
    dataset_df.loc[dataset_df['blocking_key'].isin(['general electric','ge ',' ge ']), 'blocking_key'] = 'general electric'
    dataset_df.loc[dataset_df['blocking_key'].isin(['konika','konica','minolta']), 'blocking_key'] = 'konica minolta'
    dataset_df.loc[dataset_df['blocking_key'].isin(['vivicam','vivitar','sakar']), 'blocking_key'] = 'sakar'
    dataset_df.loc[dataset_df['blocking_key'].isin(['kodak','kodax']), 'blocking_key'] = 'kodak'
    dataset_df.loc[dataset_df['blocking_key'].isin(['apple','iphone']), 'blocking_key'] = 'apple'
    dataset_df.loc[dataset_df['blocking_key'].isin(['sony','effio']), 'blocking_key'] = 'sony'
    dataset_df.loc[dataset_df['blocking_key'].isin(['hp ','hewlett packard']), 'blocking_key'] = 'hp'
    print (dataset_df.head(5))
    #dataset_df.to_csv('/home/sunji/EM_sigmod/quickstart_package/extracted_dataset_key.csv', index=False)
    print ('>>> Blocking Key Prepared')
    
    dataset_df = extract_information(dataset_df)
    #dataset_df.to_csv('middle_dataset.csv', index=False)
    print ('>>> Spec Models Prepared')
    
    #dataset_df = pd.read_csv('/home/sunji/EM_sigmod/total_with_key_type.csv')
    #dataset_df = dataset_df.fillna('')
    pairs_df = get_block_pairs_df(dataset_df, pool)
    pairs_df.to_csv(output_path+'/submission.csv', index=False)
