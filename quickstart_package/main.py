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
 'carbose',
 'brinno',
 'sony','effio',
 'kavass',
 'olympus','olympul','olymus',
 'pextax','pentax',
 'digital blue',
 'lytro',
 'lowepro',
 'fuifilm','fuji','fugi','fujufilm', 'fijifilm','fiji film','finepix','fugi film', 'fugifilm','fuijifilm',
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
 'bushnell',
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
 #'yourdeal',
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
                ids.append(w + '//' +it)
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
                #products[w][it.split('.')[0]] = data
    return pd.DataFrame(data={'spec_id':ids, 'page_title':titles, 'camera_brand': brands, 'camera_model': models, 'camera_mpn': mpns})

def compute_tf_idf(dataset_path):
    sentences = []
    webs = [f for f in listdir(dataset_path)]
    for w in webs:
        items = [f for f in listdir(dataset_path+'/'+w)]
        for it in items:
            with open(dataset_path+'/'+w+'/'+it, 'r') as f:
                data = json.load(f)
                sentence = ''
                for k, v in data.items():
                    sentence += str(k)
                    sentence += ' '
                    if type(v) == list:
                        for vv in v:
                            sentence += str(vv)
                            sentence += ' '
                    else:
                        sentence += str(v)
                        sentence += ' '
                sentences.append(sentence)
    total_words_num = 0
    word_freqs = {}
    for s in sentences:
        for ss in s.split(' '):
            total_words_num += 1
            if ss.lower() in word_freqs:
                word_freqs[ss.lower()] += 1
            else:
                word_freqs[ss.lower()] = 1
    for k in word_freqs.keys():
        word_freqs[k] /= float(total_words_num)
    word_df = {}
    for s in sentences:
        for ss in set(s.lower().split(' ')):
            if ss in word_df:
                word_df[ss] += 1
            else:
                word_df[ss] = 1
    for k in word_df.keys():
        word_df[k] = math.log(len(sentences) / float(word_df[k]))
    word_tfidf = {}
    for k in word_df.keys():
        word_tfidf[k] = word_freqs[k] * word_df[k]
    return word_tfidf

def filter_accessary(words, keyword, all_brands):
    try:
        pos = words.index(keyword)
        return len(set(words[0:pos]) & all_brands) > 0
    except:
        return False

def is_type(words, idx, current_num, current_num2, all_brands):
    exist_number_match = re.findall(r'[0-9]+', words[idx])
    only_number_match = re.findall(r'^[0-9]+$', words[idx])
    possible_mpn_match = re.findall(r'\d{4}', words[idx])
    letter_match = re.findall(r'[a-z]+', words[idx])
    stop_match = re.findall(r'["$\']+', words[idx])
    if 'comparison' in words[0:idx] and filter_accessary(words, 'comparison', all_brands) and current_num2 > 0: # Kit is not important
        return False
    if 'kit' in words[0:idx] and filter_accessary(words, 'kit', all_brands) and current_num2 > 0: # Kit is not important
        return False
    if '&' in words[0:idx] and filter_accessary(words, '&', all_brands) and current_num2 > 0: # Kit is not important
        return False
    if '+' in words[0:idx] and filter_accessary(words, '+', all_brands) and current_num2 > 0: # Kit is not important
        return False
    if 'w/' in words[0:idx] and filter_accessary(words, 'w/', all_brands) and current_num2 > 0: # Kit is not important
        return False
    if 'w' in words[0:idx] and filter_accessary(words, 'w', all_brands) and current_num2 > 0: # Kit is not important
        return False
    if 'with' in words[0:idx] and filter_accessary(words, 'with', all_brands) and current_num2 > 0: # Kit is not important
        return False
    if len(words[idx]) > 20:
        return False
    if not exist_number_match and words[idx] not in ['effio','effioe','effioa','effiop','effio-a','effio-v','effios','effiov','effio-s','effio-e','effio-p','n', 'r', 'xsi', 'z', 'a', 'xti', 'xt', 'eos-m', 'eosm', 'i','ii','iii','vi','iv', 'iis', 'markii', 'markiii', 'mkii', 'mkiii','slv', 'df', 'q', 'm', 'x', 'nx', 'xs', 'k-x', 'k-r', 'xs-pro', 'v', 'gr', 'px']: # No number, cannot be type
        return False
    if only_number_match and (len(set(words[0:idx]) & all_brands) == 0 or len(words[idx]) > 4):
        return False
    if words[idx] in ['960h','m43', '1080p', '960p', '720p', '360p', 'h.264', 'p2p', '20x', 'ip66', 'mpeg4', '1/3']:
        return False
    if not letter_match:
        if stop_match: # money, size etc.
            return False
        if idx < len(words)-1 and (words[idx+1] in ['tvl', 'meapixels', 'mg', 'batteries', 'gb', 'lens', 'meters', 'mm', 'mp', 'inch', 'megapixel', 'megapixels', 'mega', 'pack']):
            return False
        if len(set(words[max(0, idx-2):idx]) & all_brands) == 0 and words[idx-1] not in ['rebel', 'finepix'] or (words[idx-2] == 'eos' and words[idx-1] != 'rebel') or words[idx-2] == 'finepix':
            if current_num >= 1 and words[idx-1] != 'mark':
                return False
            if idx < len(words)-1 and (words[idx+1].endswith('mm') or words[idx+1].endswith('mp') or words[idx+1].endswith('inch') or words[idx+1].endswith('megapixel') or words[idx+1].endswith('mega')):
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
            if current_num >= 1 and idx < len(words)-1 and (words[idx+1] in ['millions', 'mg', 'mm','mp','meters','inch','megapixel','mega'] or (words[idx+1].endswith('mm') or words[idx+1].endswith('mp') or words[idx+1].endswith('inch') or words[idx+1].endswith('megapixel') or words[idx+1].endswith('mega')) and float(''.join(re.findall(r'\d+', words[idx+1]))) < 10):
                return False
            if idx < len(words)-1 and (words[idx+1] in ['millions', 'mg', 'mm','mp','meters','inch','megapixel','mega'] or (words[idx+1].endswith('mg') or words[idx+1].endswith('mm') or words[idx+1].endswith('mp') or words[idx+1].endswith('inch') or words[idx+1].endswith('megapixel') or words[idx+1].endswith('mega')) and float(''.join(re.findall(r'\d+', words[idx+1]))) == 0):
                return False
        if words[idx] in ['2015', '2014', '2013', '2012', '2011']:
            return False
    else:
        if words[idx] in ['i', 'ii', 'iii', 'iv'] and 'camera' not in ''.join(words[max(0,idx-2):idx]) and (idx > 0 and words[idx-1] not in ['mark', 'mk']) and ('af-s' in words[0:idx] or (idx < len(words)-1 and words[idx+1] == 'lens') or len(set(words[max(0, idx-4):idx]) & all_brands) == 0):
            return False
        if words[idx] in ['iis', 'z', 'm', 'x', 'df', 'f1', 'gr', 'px', 'slv', 'xs', 'k-x', 'k-r', 'xs-pro', 'v', 'q'] and (len(set(words[max(0, idx-4):idx]) & all_brands) == 0 or current_num > 0):
            return False
        if words[idx] in ['nx'] and words[idx+1] != 'mini':
            return False
        if (re.findall(r'\d*\.?\d*\-?\d*\.?\d*mm$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*mp$', words[idx]) or
           re.findall(r'^\d+\.?\d*\-?\d*\.?\d*m$', words[idx]) or
           re.findall(r'\d+\.?\d*\-?\d*\.?\d*meter$', words[idx]) or
           re.findall(r'^\d+\.?\d*\-?\d*\.?\d*cm$', words[idx]) or
           re.findall(r'\d*\.?\d*\-?\d*\.?\d*megpixel$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*cmos$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*pcs$', words[idx]) or 
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*mb/s$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*pc$', words[idx]) or
           re.findall(r'\d*\.?\d*\-?\d*\.?\d*inch$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*gb$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*g$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*fps$', words[idx]) or
           re.findall(r'\d*\.?\d*\-?\d*\.?\d*megapixel$', words[idx]) or
           re.findall(r'\d*\.?\d*\-?\d*\.?\d*digital$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*mega$', words[idx]) or
           re.findall(r'\d*\.?\d*\-?\d*\.?\d*megapixels$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*year$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*yr$', words[idx]) or
           re.findall(r'^s\d+\.?\d*\-\d+\.?\d*$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*g?hz$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*lens$', words[idx]) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*colors?$', words[idx]) or
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
        if stop_match:
            return False
        if re.findall(r'^[0-9]+\.?[0-9]*x$', words[idx]) and len(set(words[max(0, idx-2):idx]) & all_brands) == 0:
            return False
    return True

def verify_model(word, current_size):
    if re.findall(r'\d+', word) and current_size > 0:
        return False
    if (re.findall(r'\d*\.?\d*\-?\d*\.?\d*mm$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*mp$', word) or
           re.findall(r'\d+\.?\d*\-?\d*\.?\d*m$', word) or
           re.findall(r'\d+\.?\d*\-?\d*\.?\d*cm$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*megpixel$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*mega$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*cmos$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*pcs$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*pc$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*inch$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*gb$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*g$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*fps$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*megapixel$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*megapixels$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*year$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*yr$', word) or
           re.findall(r'^s\d+\.?\d*\-\d+\.?\d*$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*g?hz$', word) or
           re.findall(r'^\d*\.?\d*\-?\d*\.?\d*lens$', word) or
           re.findall(r'ip\d*$', word)):
        return False
    if word in ['1080p', '720p', '360p', 'h.264', 'p2p']:
        return False
    if word.startswith('sel') or word.startswith('sal'):
        return False
    if re.findall(r'[0-9]+\-[0-9]+\/[0-9]+-[0-9]+$', word):
        return False
    if re.findall(r'^f\/[0-9]+\.?[0-9]*', word):
        return False
    if re.findall(r'^f\/?[0-9]+\.?[0-9]*$', word):
        return False
    if re.findall(r'^f\/?[0-9]+\.?[0-9]*\-[0-9]+\.?[0-9]*', word):
        return False
    if re.findall(r'^[0-9]+\.?[0-9]*x$', word):
        return False
    return True

def tfidf(x, word_tfidf):
    if x in word_tfidf:
        return word_tfidf[x]
    else:
        return 0.0

def extract_type(row, word_tfidf, stopped_words, all_brands):
    words = re.split(r'[, ()~]', str(row['page_title']))
    brand = str(row['blocking_key'])
    words = [x.strip(',;-().') for x in words]
    types = []
    version = None
    generations = {'i':'1','ii':'2','iii':'3','iv':'4'}
    generations2 = {'i':'one','ii':'two','iii':'three','iv':'four'}
    keep_num = 2
    for idx, w in enumerate(words):
        if w in types:
            continue
        if is_type(words, idx, len(re.findall(r'\d+', ''.join(types))), len(types), all_brands):
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
                        if idx < len(words)-1 and words[idx+1] in ['exr', 'f', 'v', 'pro', 'zoom']:
                            w = w+words[idx+1]
                        if idx > 0 and ''.join(words[idx-1].split('-')) in ['ds', 'mx', 'jx', 'av', 'ax', 'hs', '3d', 's', 'f', 'a', 'e', 't', 'j', 'z', 'xp', 'x', 'pro', 'xt', 'xe', 'xa', 'xpro']:
                            w = ''.join(words[idx-1].split('-'))+w
                    if not re.findall(r'^\d+$', w) or len(w) >= 2:
                        types.append(w)
            elif brand == 'cannon':
                if len(''.join(re.findall(r'\d+', w))) > 5:
                    continue
                sentence = ''.join(words[0:min(10,len(words))])
                if 'powershot' in sentence or 'ixus' in sentence or 'elph' in sentence or 'ixy' in sentence:
                    if idx < len(words)-1 and words[idx+1] in ['zoom', 'is', 'x', 'a', 's', 'f', 'hs']:
                        w = w+words[idx+1]
                    if idx > 0 and words[idx-1] in ['sx', 'ixy','pro', 'a', 'g', 's', 'sd', 'ixus', 'elph']:
                        w = words[idx-1]+w
                elif 'eos' in sentence:
                    if idx < len(words)-1 and words[idx+1] in ['d', 'ds', 'dx', 'dsr', 'da', 'v']:
                        w = w+words[idx+1]
                    if idx > 0 and words[idx-1] in ['m', 'ds', 'kiss']:
                        w = words[idx-1]+w
                else:
                    if idx < len(words)-1 and words[idx+1] in ['d', 'ds', 'dx', 'dsr', 'da', 'v']:
                        w = w+words[idx+1]
                    if idx > 0 and words[idx-1] in ['m', 'ds', 'sd', 'kiss']:
                        w = words[idx-1]+w
                if not re.findall(r'^\d+$', w) or len(w) >= 2:
                    types.append(w)
            elif brand == 'nikon':
                if len(''.join(re.findall(r'\d+', w))) > 5:
                    continue
                if idx < len(words)-1 and words[idx+1] in ['a', 's', 'as', 't', 'hp', 'p', 'h', 'af', 'm', 'x']:
                    w = w+words[idx+1]
                if idx > 0 and words[idx-1] in ['s', 'aw', 'd', 'f', 'fa', 'fg', 'fe', 'fm', 'v', 'j', 'z']:
                    w = words[idx-1]+w
                if not re.findall(r'^\d+$', w) or len(w) >= 2:
                    types.append(w)
            elif brand == 'olympus':
                if len(''.join(re.findall(r'\d+', w))) > 5:
                    continue
                sentence = ''.join(words[0:min(10,len(words))])
                if idx < len(words)-1 and words[idx+1] in ['rs', 'wp', 'sw', 'ee', 'x']:
                    w = w+words[idx+1]
                    
                if idx > 0 and words[idx-1] in ['c' or 'camedia']:
                    w = 'c'+w
                elif idx > 0 and ''.join(words[idx-1].split('-')) in ['xz', 'vh','t','vr','vg','sz', 'x', 'ir', 'sp', 'd', 'e', 'f', 'fe', 'p', 'ep', 'tough', 'tg', 'e', 'em']:
                    w = ''.join(words[idx-1].split('-'))+w
                elif idx > 0 and ('stylus' in words[idx-1] or 'miu' in words[idx-1] or 'µ' in words[idx-1] or 'mju' in words[idx-1]):
                    w = 'µ'+w
                types.append(w)
            elif brand == 'panasonic':
                if idx > 0 and ''.join(words[idx-1].split('-')) in ['dmc', 'f','g','fx','fz','l','ls','lx','lz','ts','tz','zs']:
                    w = ''.join(words[idx-1].split('-'))+w
                types.append(w)
            elif brand == 'sony':
                if w == 'effio':
                    if words[idx+1] in ['p', 's', 'e', 'v', 'a']:
                        w = w+words[idx+1]
                if idx > 0 and ''.join(words[idx-1].split('-')) in ['cd','alpha','t','fd','dsc','nex','slt','s','a']:
                    w = ''.join(words[idx-1].split('-'))+w
                elif idx > 0 and ''.join(words[idx-1].split('-')) in ['alpha', 'ilce', 'ilca']:
                    w = 'alpha'+w
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
        if len(models) > 0:
            types = set(types) | set(models[0:1])
    if len(types) == 0 and not pd.isnull(row['camera_mpn']):
        types.append(str(row['camera_mpn']))
    types = set([''.join(t.split('-')) for t in types]) - stopped_words
    types = sorted(types, key=lambda x: tfidf(x, word_tfidf), reverse=False)
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

def some_extra_rule(row):
    mapping = {'600d':'t3i', '1100d':'t3', '550d':'t2i', '500d':'t1i','700d':'t5i', '1000d':'xs', '650d':'t4i',
           't3i':'600d', 't3':'1100d', 't2i':'550d', 't1i':'500d','t5i':'700d', 'xs':'1000d', 't4i':'650d',
               'xti':'400d','400d':'xti', 'xt':'350d', '350d':'xt', 't5':'1200d', '1200d':'t5',
               'xsi':'450d', '450d':'xsi', 'sl1':'100d', '100d':'sl1'}
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
        elif all_types[i] in ['7k', '7', 'ilce7', 'ilce7k', 'alpha7', 'alpha7k', 'nex7'] and str(row['blocking_key']) == 'sony':
            if 'nex' in row['page_title']:
                all_types[i] = 'nex7k'
            elif 'ilce' in row['page_title']:
                all_types[i] = 'a7k'
            elif 'alpha' in row['page_title']:
                all_types[i] = 'a7k'
        elif all_types[i] in ['a7'] and str(row['blocking_key']) == 'sony':
            if 'nex' not in row['page_title']:
                all_types[i] = 'a7k'
            else:
                all_types[i] = 'nex7k'
        elif all_types[i] in ['alpha7'] and str(row['blocking_key']) == 'sony':
            if 'nex' not in row['page_title']:
                all_types[i] = 'a7k'
            else:
                all_types[i] = 'nex7k'
        elif all_types[i] in ['nex7'] and str(row['blocking_key']) == 'sony':
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
    all_brands = set(['dsc','stylus', 'elph', 'powershot', 'iphone', 'enxun', 'fugi', 'electric', 'howell', 'kitty', 'blue', 'figi', 'go', 'sonydigital', 'cyber', 'fujifilm', 'digital blue', 'bell howell', 'bell+howell', 'hello kitty', 'vistaquest', 'blackmagic', 'superheadz', 'hasselblad', 'hikivision', 'fuijifilm', 'panasonic', 'fugi film', 'fiji film', 'hikvision', 'polaroid', 'insignia', 'keychain', 'yourdeal', 'lowrance', 'logitech', 'fijifilm', 'bushnell', 'fujufilm', 'fugifilm', 'sylvania', 'olympul', 'lowepro', 'toshiba', 'olympus', 'samsung', 'finepix', 'fuifilm', 'samyang', 'aquapix', 'vivitar', 'neopine', 'minolta', 'easypix', 'sandisk', 'vivicam', 'philips', 'coleman', 'yashica', 'coolpix', 'emerson', 'sealife', 'crayola', 'intova', 'tamron', 'cannon', 'garmin', 'pentax', 'barbie', 'konica', 'mustek', 'aiptek', 'keedox', 'contax', 'konika', 'go pro', 'olymus', 'philip', 'wopson', 'pextax', 'wespro', 'disney', 'rollei', 'sakar', 'lytro', 'apple', 'lexar', 'kodak', 'vibe ', 'drift', 'vtech', 'vizio', 'kinon', 'haier', 'epson', 'gopro', 'croco', 'minox', 'nokia', 'dahua', 'sanyo', 'intel', 'kodax', 'ricoh', 'argus', 'lumix', 'nikon', 'absee', 'nicon', 'bmpcc', 'canon', 'casio', 'cobra', 'leica', 'sigma', 'sony', 'fuji', 'b h ', 'lego', 'benq', 'asus', 'pov', 'jvc', 'b+w', 'ion', 'lg ', 'dji', 'tvc', 'eos', 'ge ', 'vpc', 'dxg', 'svp', 'hp'])
    stopped_words = set(['bin2','cam','ixus','pro','rebel','shot','cyber','catalog','stylus','mark','edition','elph','','digital camera', 'camera','wide','model','sport','network','action','full','ir','mini','metal','digital camera', 'indoor', 'canon eos', 'compact', 'digital', 'digital rebel', 'case', 'lcd', 'panasonic lumix', 'slr', 'cybershot', 'hd', '3x', '4x', '5x', '3gp', '']) | all_brands
    structured_data['type'], structured_data['version'] = zip(*structured_data.swifter.apply(lambda row: extract_type(row, word_tfidf, stopped_words, all_brands), axis=1))
    structured_data['type'] = structured_data.swifter.apply(lambda row: some_extra_rule(row), axis=1)
    structured_data['type_number'] = structured_data.swifter.apply(lambda row: extract_type_number(row), axis=1)
    structured_data['need_version'] = structured_data.swifter.apply(lambda row: isVersionNeeded(row), axis=1)
    return structured_data

def create_dataframe(dataset_path):
    """Function used to create a Pandas DataFrame containing specifications page titles

    Reads products specifications from the file system ("dataset_path" variable in the main function) and creates a Pandas DataFrame where each row is a
    specification. The columns are 'source' (e.g. www.sourceA.com), 'spec_number' (e.g. 1) and the 'page title'. Note that this script will consider only
    the page title attribute for simplicity.

    Args:
        dataset_path (str): The path to the dataset

    Returns:
        df (pd.DataFrame): The Pandas DataFrame containing specifications and page titles
    """

    print('>>> Creating dataframe...\n')
    columns_df = ['source', 'spec_number', 'spec_id', 'page_title']

    progressive_id = 0
    progressive_id2row_df = {}
    for source in tqdm(os.listdir(dataset_path)):
        for specification in os.listdir(os.path.join(dataset_path, source)):
            specification_number = specification.replace('.json', '')
            specification_id = '{}//{}'.format(source, specification_number)
            with open(os.path.join(dataset_path, source, specification)) as specification_file:
                specification_data = json.load(specification_file)
                page_title = specification_data.get('<page title>').lower()
                row = (source, specification_number, specification_id, page_title)
                progressive_id2row_df.update({progressive_id: row})
                progressive_id += 1
    df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=columns_df)
    print(df)
    print('>>> Dataframe created successfully!\n')
    return df


def __get_blocking_keys(df):
    """Private function used to calculate a set of blocking keys

    Calculates the blocking keys simply using the first three characters of the page titles. Each 3-gram extracted in
    this way is a blocking key.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing specifications and page titles
    Returns:
        blocking_keys (set): The set of blocking keys calculated
    """

    blocking_keys = set()
    for _, row in df.iterrows():
        page_title = row['page_title']
        #blocking_key = page_title[:3]
        blocking_key = page_title.split(' ')[0]
        if len(blocking_key) >= 3:
            blocking_keys.add(blocking_key)
    return blocking_keys


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
    brand_cnt = 0
#    blocking_keys = __get_blocking_keys(df)
#    df['blocking_key'] = ''
    for index, row in tqdm(df.iterrows()):
        page_title = str(row['camera_brand']) + ' ' + str(row['page_title'])
        positions = []
        for brand in brands_for_blocking:
            pos = page_title.find(brand)
            if pos >= 0 and not (brand == 'ge ' and pos > 0):
                positions.append((brand, pos))
            if len(positions) >= 3:
                break
        if len(positions) > 0:
            brand = sorted(positions, key=lambda x:x[1])[0][0]
            df.at[index, 'blocking_key'] = brand
            brand_cnt += 1
    print(df.head(5))
    print('>>> Blocking computed successfully!\nbrand_count: ', brand_cnt)
    return df

def get_block_pairs_df(df):
    """Function used to get a Pandas DataFrame containing pairs of specifications based on the blocking keys

    Creates a Pandas DataFrame where each row is a pair of specifications. It will create one row for every possible
    pair of specifications inside a block.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing specifications, page titles and blocking keys

    Returns:
        pairs_df (pd.DataFrame): A Pandas DataFrame containing pairs of specifications
    """
    print('>>> Creating pairs dataframe...\n')
    types = {}
    type_numbers = {}
    type_numbers_pad = {}
    titles = {}
    for idx, row in df.iterrows():
        types[idx] = row['type'].split(';')
        type_numbers_pad[idx] = [''.join(re.findall(r'\d+', x)) for x in types[idx]]
        type_numbers[idx] = row['type_number'].split(';')
        titles[idx] = row['page_title'].split(';')
    grouped_df = df[df['blocking_key'] != ''].groupby('blocking_key')
    index_pairs = []
    for _, block in grouped_df:
        block_indexes = list(block.index)
        index_pairs.extend(list(itertools.combinations(block_indexes, 2)))
    pairs = []
    #cores = mp.cpu_count()
    cores = 4
    num_process = 48
    pool = Pool(processes=cores)
    result=[]
    avg = int(len(index_pairs) / num_process)
    for i  in range(num_process):
       if i < num_process-1:
           result.append(pool.apply_async(entity_matching, args=(index_pairs[i*avg:(i+1)*avg], types, type_numbers, type_numbers_pad, titles, df,))) 
       else:    
           result.append(pool.apply_async(entity_matching, args=(index_pairs[i*avg:], types, type_numbers, type_numbers_pad, titles, df,))) 
    #pool.join()
    for i in result:
        pairs.append(i.get())
    final = pd.concat(pairs, axis=0)
    print (final.head(5))
    return final
    
def entity_matching(index_pairs, types, type_numbers, type_numbers_pad, titles, df):
    #types = {}
    #type_numbers = {}
    #type_numbers_pad = {}
    #titles = {}
    #for idx, row in df.iterrows():
    #    types[idx] = row['type'].split(';')
    #    type_numbers_pad[idx] = [''.join(re.findall(r'\d+', x)) for x in types[idx]]
    #    type_numbers[idx] = row['type_number'].split(';')
    #    titles[idx] = row['page_title'].split(';')

    left_spec_ids = []
    right_spec_ids = []
    for index_pair in tqdm(index_pairs):
        left_index, right_index = index_pair
        left_spec_id = df.loc[left_index, 'spec_id']
        right_spec_id = df.loc[right_index, 'spec_id']
        left_type = types[left_index]
        right_type = types[right_index]
        left_type_number = type_numbers[left_index]
        right_type_number = type_numbers[right_index]
        if len(set(left_type+left_type_number) & set(right_type+right_type_number)) > 0:
            #row = (left_spec_id, right_spec_id, left_spec_title, right_spec_title)
            #progressive_id2row_df.update({progressive_id: row})
            #progressive_id += 1
            if pair_matching(left_index, right_index, types, type_numbers_pad, titles, df) == 1:
                left_spec_ids.append(left_spec_id)
                right_spec_ids.append(right_spec_id)
    pairs_df = pd.DataFrame(data={'left_spec_id': left_spec_ids, 'right_spec_id': right_spec_ids})
    #columns_df = ['left_spec_id', 'right_spec_id', 'left_spec_title', 'right_spec_title']
    #pairs_df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=columns_df)
    #print(pairs_df.head(5))
    #print('>>> Pairs dataframe created successfully!\n')
    return pairs_df

def pair_matching(left_index, right_index, types, type_numbers, titles, df):
    if len(df.loc[left_index, 'type']) == 0 or len(df.loc[right_index, 'type']) == 0:
        left_title = titles[left_index]
        right_title = titles[right_index]
        if len(left_title) == 0 or len(right_title) == 0:
            return 0
        if len(set(left_title) & set(right_title)) / float(len(set(left_title) | set(right_title))) > 0.9:
            return 1
        return 0
    left_type = types[left_index]
    right_type = types[right_index]
    left_number = type_numbers[left_index] 
    right_number = type_numbers[right_index]
    brand_left = df.loc[left_index, 'blocking_key']
    brand_right = df.loc[right_index, 'blocking_key']
    version_left = str(df.loc[left_index, 'version'])
    version_right = str(df.loc[right_index, 'version'])
    need_version = (df.loc[left_index, 'need_version'] + df.loc[right_index, 'need_version'] > 0)
    for t in range(len(right_type)):
        if len(left_number[0]) > 0 and ('1' not in [left_type[0], right_type[t]]) and left_number[0] == right_number[t] and (left_type[0] in right_type[t] or right_type[t] in left_type[0]):
            if need_version:
                if version_left == version_right and (len(version_left) != 0 or left_type[0][-1] == right_type[t][-1]):
                    return 1
            else:
                return 1
        if left_type[0] == right_type[t] and version_left == version_right:
            return 1
    for t in range(len(left_type)):
        if len(right_number[0]) > 0 and ('1' not in [left_type[t], right_type[0]]) and left_number[t] == right_number[0] and (left_type[t] in right_type[0] or right_type[0] in left_type[t]):
            if need_version:
                if version_left == version_right and (len(version_left) != 0 or left_type[t][-1] == right_type[0][-1]):
                    return 1
            else:
                return 1
        if right_type[0] == left_type[t] and version_left == version_right:
            return 1
    if len(left_type) > 1 and len(right_type) > 1 and len(right_number[1]) > 0 and left_type[1] == right_type[1] and version_left == version_right and re.findall(r'[a-z]+', right_type[1]):
        return 1
    if brand_left == 'sony' and len(set(['3000','5000','6000']) & set(left_number) & set(left_number)) > 0:
        return 1
    return 0

def product_match(row):
    try:
        if pd.isnull(row['type_left']) or pd.isnull(row['type_right']):
            left_title = str(row['left_spec_title']).split(' ')
            right_title = str(row['right_spec_title']).split(' ')
            if len(left_title) == 0 or len(right_title) == 0:
                return 0
            if len(set(left_title) & set(right_title)) / float(len(set(left_title) | set(right_title))) > 0.9:
                return 1
            return 0
    except:
        print (pd.isnull(row['type_left']), pd.isnull(row['type_right']), row['type_left'], row['type_right'])
    left_type = str(row['type_left']).split(';')
    right_type = str(row['type_right']).split(';')
    left_number = [''.join(re.findall(r'\d+', x)) for x in left_type]
    right_number = [''.join(re.findall(r'\d+', x)) for x in right_type]
    for t in range(len(right_type)):
        if len(left_number[0]) > 0 and ('1' not in [left_type[0], right_type[t]]) and left_number[0] == right_number[t] and (left_type[0] in right_type[t] or right_type[t] in left_type[0]):
            if ((row['blocking_key_left'] == 'cannon' and (left_type[0].startswith('xt') or left_type[0].startswith('xs') or left_type[0].startswith('g1x') or left_type[0].startswith('5d') or left_type[0].startswith('1d') or left_type[0].startswith('7d') or left_type[0].startswith('t3') or left_type[0].startswith('t5') or left_type[0].startswith('t1') or left_type[0].startswith('t2') or left_type[0].startswith('t4'))) or (row['blocking_key_left'] == 'sony' and ('rx1' in left_type[0] or left_type[0].startswith('a77'))) or (row['blocking_key_left'] == 'nikon') or (row['blocking_key_left'] == 'fuji' and left_type[0].startswith('x100'))):
                if str(row['version_left']) == str(row['version_right']) and (not pd.isnull(row['version_left']) or left_type[0][-1] == right_type[t][-1]):
                    return 1
            else:
                return 1
        if left_type[0] == right_type[t] and str(row['version_left']) == str(row['version_right']):
            return 1
    for t in range(len(left_type)):
        if len(right_number[0]) > 0 and ('1' not in [left_type[t], right_type[0]]) and left_number[t] == right_number[0] and (left_type[t] in right_type[0] or right_type[0] in left_type[t]):
            if ((row['blocking_key_right'] == 'cannon' and (right_type[0].startswith('xt') or right_type[0].startswith('xs') or right_type[0].startswith('g1x') or right_type[0].startswith('5d') or right_type[0].startswith('1d') or right_type[0].startswith('7d') or right_type[0].startswith('t3') or right_type[0].startswith('t5') or right_type[0].startswith('t1') or right_type[0].startswith('t2') or right_type[0].startswith('t4'))) or (row['blocking_key_right'] == 'sony' and ('rx1' in right_type[0] or right_type[0].startswith('a77'))) or (row['blocking_key_right'] == 'nikon') or (row['blocking_key_right'] == 'fuji' and right_type[0].startswith('x100'))):
                if str(row['version_left']) == str(row['version_right']) and (not pd.isnull(row['version_left']) or right_type[0][-1] == left_type[t][-1]):
                    return 1
            else:
                return 1
        if right_type[0] == left_type[t] and str(row['version_left']) == str(row['version_right']):
            return 1
    if len(left_type) > 1 and len(right_type) > 1:
        if  re.findall(r'[a-z]+', right_type[1]) and len(right_number[1]) > 0 and left_type[1] == right_type[1] and str(row['version_left']) == str(row['version_right']):
            return 1
#    if len(left_type) == 1 and len(right_type) == 1 and len(left_number[0]) == 0 and len(right_number[0]) == 0 and not pd.isnull(row['blocking_key_left']) and (left_type[0] in right_type[0] or right_type[0] in left_type[0]):
#        return 1
    if row['blocking_key_left'] == 'sony' and row['blocking_key_right'] == 'sony' and len(set(['3000','5000','6000']) & set(str(row['type_number_left']).split(';')) & set(str(row['type_number_right']).split(';'))) > 0:
        return 1
#    if left_type[0] in right_type or right_type[0] in left_type:
#        return 1
#    if not (pd.isnull(row['type_number_left']) or pd.isnull(row['type_number_right'])):
#        left_numbers = set(str(row['type_number_left']).split(';'))
#        right_numbers = set(str(row['type_number_right']).split(';'))
#        if len(left_numbers_type) > 0 and len(right_numbers_type) > 0 and (left_number_type[0] in right_number_type or right_number_type[0] in left_number_type):
#            return 1
    #left_title = str(row['left_spec_title']).split(' ')
    #right_title = str(row['right_spec_title']).split(' ')
    #left_numbers = set(str(row['type_number_left']).split(';'))
    #right_numbers = set(str(row['type_number_right']).split(';'))
    #if (not pd.isnull(row['blocking_key_left'])) and len(left_numbers & right_numbers) > 0 and len(set(left_title) & set(right_title)) / float(len(set(left_title) | set(right_title))) > 0.5:
    #    return 1
    return 0

def compute_matching(pairs_df, isTest):
    """Function used to actually compute the matching specifications

    Iterates over the pairs DataFrame and uses a matching function to decide if they represent the same real-world
    product or not. Two specifications are matching if they share at least 2 tokens in the page title.
    The tokenization is made by simply splitting strings by using blank character as separator.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing pairs of specifications

    Returns:
        matching_pairs_df (pd.DataFrame): The Pandas DataFrame containing the matching pairs
    """

    print('>>> Computing matching...\n')
#    dataset_df = dataset_df.set_index('spec_id')
    pairs_df['predict'] = pairs_df.swifter.apply(lambda row: product_match(row), axis=1)
    if isTest:
        matching_pairs_df = pairs_df[pairs_df['label'] != pairs_df['predict']][['left_spec_id', 'right_spec_id', 'page_title_left', 'page_title_right', 'type_left', 'type_right', 'label', 'predict']]
    else:
        matching_pairs_df = pairs_df[pairs_df['predict'] == 1][['left_spec_id', 'right_spec_id']]
    print(matching_pairs_df.head(5))
    print('>>> Matching computed successfully!\n')
    return matching_pairs_df


"""
    This script will:
    1. create a Pandas DataFrame for the dataset. Note that only the <page title> attribute is considered (for example purposes);
    2. partition the rows of the Pandas DataFrame in different blocks, accordingly with a blocking function;
    3. create a Pandas DataFrame for all the pairs computed inside each block;
    4. create a Pandas DataFrame containing all the matching pairs accordingly with a matching function;
    5. export the Pandas DataFrame containing all the matching pairs in the "outputh_path" folder.
"""
if __name__ == '__main__':
    dataset_path = '/home/sunji/EM_sigmod/2013_camera_specs'

    dataset_df = load_dataset(dataset_path)
    print ('>>> Dataset Loading Completed')
    #dataset_df = create_dataframe(dataset_path)
    dataset_df = compute_blocking(dataset_df)
    dataset_df.loc[dataset_df['blocking_key'].isin(['hikvision','hikivision']), 'blocking_key'] ='hikvision'
    dataset_df.loc[dataset_df['blocking_key'].isin(['fuifilm','fuji','fugi','fujufilm', 'fijifilm','fiji film','finepix','fugi film', 'fugifilm','fuijifilm']), 'blocking_key'] ='fuji'
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
    print ('>>> Spec Models Prepared')
    
    #dataset_df = pd.read_csv('/home/sunji/EM_sigmod/total_with_key_type.csv')
    #dataset_df = dataset_df.fillna('')
    pairs_df = get_block_pairs_df(dataset_df)
    pairs_df.to_csv('/home/sunji/EM_sigmod/quickstart_package/candidate_pairs_key.csv', index=False)
    '''
    pairs_df = pd.read_csv('/home/sunji/EM_sigmod/quickstart_package/label_pairs_with_key_type.csv')
    matching_pairs_df = compute_matching(pairs_df, True)
    if (len(matching_pairs_df) > 2):
        print ('>>> Not Working.')
    else:
        pairs_df = pd.read_csv('/home/sunji/EM_sigmod/quickstart_package/candidate_pairs_with_key_type.csv')
        matching_pairs_df = compute_matching(pairs_df, False)
    # Save the submission as CSV file in the outputh_path
    matching_pairs_df.to_csv(outputh_path + '/submission.csv', index=False)
    print('>>> Submission file created in {} directory.'.format(outputh_path))
    '''
