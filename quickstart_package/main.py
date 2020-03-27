import os
import re
import json
import numpy as np
import swifter
import editdistance
import pandas as pd
import itertools
from tqdm import tqdm

all_brands = set(['cannon','canon','eos',
 'nikon','coolpix','nicon',
 'sony',
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
 'hp',
 'hasselblad',
 'benq',
 'coleman',
 'polaroid',
 'cobra',
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
 'intel',
 'aquapix',
 'apple','iphone',
 'keedox',
 'lego',
 'logitech',
 'crayola',
])
all_brands = sorted(all_brands, key=lambda x: len(x), reverse=True)
print (all_brands)

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
        for brand in all_brands:
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
    grouped_df = df.groupby('blocking_key')
    index_pairs = []
    for _, block in grouped_df:
        block_indexes = list(block.index)
        index_pairs.extend(list(itertools.combinations(block_indexes, 2)))

    progressive_id = 0
    progressive_id2row_df = {}
    for index_pair in tqdm(index_pairs):
        left_index, right_index = index_pair
        left_spec_id = df.loc[left_index, 'spec_id']
        right_spec_id = df.loc[right_index, 'spec_id']
        left_spec_title = df.loc[left_index, 'page_title']
        right_spec_title = df.loc[right_index, 'page_title']
        left_type = str(df.loc[left_index, 'type']).split(';')
        right_type = str(df.loc[right_index, 'type']).split(';')
        left_type_number = str(df.loc[left_index, 'type_number']).split(';')
        right_type_number = str(df.loc[right_index, 'type_number']).split(';')
        if len(set(left_type) & set(right_type)) > 0 or len(set(left_type_number) & set(right_type_number)) > 0 or (len(left_type) == 0 and len(right_type) == 0):
            row = (left_spec_id, right_spec_id, left_spec_title, right_spec_title)
            progressive_id2row_df.update({progressive_id: row})
            progressive_id += 1

    columns_df = ['left_spec_id', 'right_spec_id', 'left_spec_title', 'right_spec_title']
    pairs_df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=columns_df)
    print(pairs_df)
    print('>>> Pairs dataframe created successfully!\n')
    return pairs_df

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
        if len(left_number[0]) > 0 and left_number[0] != '1' and left_number[0] == right_number[t] and (left_type[0] in right_type[t] or right_type[t] in left_type[0]):
            if ((row['blocking_key_left'] == 'cannon' and (left_type[0].startswith('xt') or left_type[0].startswith('xs') or left_type[0].startswith('5d') or left_type[0].startswith('1d') or left_type[0].startswith('7d') or left_type[0].startswith('t3') or left_type[0].startswith('t5') or left_type[0].startswith('t1') or left_type[0].startswith('t2') or left_type[0].startswith('t4'))) or (row['blocking_key_left'] == 'sony' and ('rx100' in left_type[0] or left_type[0].startswith('a77'))) or (row['blocking_key_left'] == 'nikon')):
                if str(row['version_left']) == str(row['version_right']) and (not pd.isnull(row['version_left']) or left_type[0][-1] == right_type[t][-1]):
                    return 1
            else:
                return 1
        if left_type[0] == right_type[t] and str(row['version_left']) == str(row['version_right']):
            return 1
    for t in range(len(left_type)):
        if len(right_number[0]) > 0 and right_number[0] != '1' and left_number[t] == right_number[0] and (left_type[t] in right_type[0] or right_type[0] in left_type[t]):
            if ((row['blocking_key_right'] == 'cannon' and (right_type[0].startswith('xt') or right_type[0].startswith('xs') or right_type[0].startswith('5d') or right_type[0].startswith('1d') or right_type[0].startswith('7d') or right_type[0].startswith('t3') or right_type[0].startswith('t5') or right_type[0].startswith('t1') or right_type[0].startswith('t2') or right_type[0].startswith('t4'))) or (row['blocking_key_right'] == 'sony' and ('rx100' in right_type[0] or right_type[0].startswith('a77'))) or (row['blocking_key_left'] == 'nikon')):
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
    outputh_path = './output'
    '''    
    dataset_df = pd.read_csv('/home/sunji/EM_sigmod/extracted_dataset.csv')
    #dataset_df = create_dataframe(dataset_path)
    dataset_df = compute_blocking(dataset_df)
    dataset_df.loc[dataset_df['blocking_key'].isin(['hikvision','hikivision']), 'blocking_key'] ='hikvision'
    dataset_df.loc[dataset_df['blocking_key'].isin(['fuifilm','fuji','fugi','fujufilm', 'fijifilm','fiji film','finepix','fugi film', 'fugifilm','fuijifilm']), 'blocking_key'] ='fuji'
    dataset_df.loc[dataset_df['blocking_key'].isin(['bell+howell','b+w','bell howell','b h ']), 'blocking_key'] ='bell+howell'
    dataset_df.loc[dataset_df['blocking_key'].isin(['panasonic','lumix']), 'blocking_key'] ='panasonic'
    dataset_df.loc[dataset_df['blocking_key'].isin(['cannon','canon','eos']), 'blocking_key'] = 'cannon'
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
    print (dataset_df.head(5))
    dataset_df.to_csv('/home/sunji/EM_sigmod/quickstart_package/extracted_dataset_key.csv', index=False)
    
    dataset_df = pd.read_csv('/home/sunji/EM_sigmod/total_with_key_type.csv')
    dataset_df = dataset_df.fillna('')
    pairs_df = get_block_pairs_df(dataset_df)
    pairs_df.to_csv('/home/sunji/EM_sigmod/quickstart_package/candidate_pairs_key.csv', index=False)
    '''
    pairs_df = pd.read_csv('/home/sunji/EM_sigmod/quickstart_package/label_pairs_with_key_type.csv')
    matching_pairs_df = compute_matching(pairs_df, True)
    if (len(matching_pairs_df) > 1):
        print ('>>> Not Working.')
    else:
        pairs_df = pd.read_csv('/home/sunji/EM_sigmod/quickstart_package/candidate_pairs_with_key_type.csv')
        matching_pairs_df = compute_matching(pairs_df, False)
    # Save the submission as CSV file in the outputh_path
    matching_pairs_df.to_csv(outputh_path + '/submission.csv', index=False)
    print('>>> Submission file created in {} directory.'.format(outputh_path))
    
