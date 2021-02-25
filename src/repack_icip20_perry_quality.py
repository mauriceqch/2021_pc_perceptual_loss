import argparse
import os
import logging
import subprocess
import multiprocessing
from utils.cc_utils import cc_convert
from utils.parallel_process import parallel_process
import utils.icip20_perry_quality as ipq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

DEFAULT_RADIUS = 17

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', required=True, help='Path to icip20_perry_quality dataset.')
    parser.add_argument('--new_db_path', required=True, help='Path to CloudCompare.')
    parser.add_argument('--cc_path', required=True, help='Path to CloudCompare.')
    parser.add_argument('--format', default='ASCII',
                        help='Specifies the default output format for PLY files. ' +
                             'Format can be one of the following: ASCII, BINARY_BE (big endian) or BINARY_LE (little endian).',)
    parser.add_argument('--n_parallel', default=max(multiprocessing.cpu_count() // 4, 1))
    args = parser.parse_args()

    db_path, new_db_path, cc_path = args.db_path, args.new_db_path, args.cc_path
    # Example paths
    # db_path = r'C:\Users\User\data\icip2020_perry_quality'
    # cc_path = r'C:\Program Files\CloudCompare\CloudCompare.exe'
    # new_db_path = r'C:\Users\User\data\icip2020_perry_quality_repack'

    merged_df = ipq.load(db_path)['merged_df']

    temp_df = merged_df.copy()
    temp_df.loc[temp_df['codec_id'] == 'AI', 'codec_id'] = 'VPCC_AI'
    temp_df.loc[temp_df['codec_id'] == 'OCT', 'codec_id'] = 'GPCC_OCT_PRED'
    temp_df.loc[temp_df['codec_id'] == 'TRI', 'codec_id'] = 'GPCC_TRI_PRED'
    temp_df = temp_df.rename(columns={'relative_path': 'old_relative_path'})
    temp_df['relative_path'] = temp_df.apply(
        lambda x: '_'.join((x['pc_name'], x['codec_id'], x['codec_rate'])) + '.ply', axis=1)

    radiuses = []
    for _, row in temp_df.iterrows():
        codec_id = row['codec_id']
        codec_rate = row['codec_rate']
#         if codec_id == 'GPCC_OCT':
#             if codec_rate == 'r01':
#                 radius = 21
#             elif codec_rate == 'r02':
#                 radius = 15
#             else:
#                 radius = 9
#         else:
#             radius = 9
        radius = DEFAULT_RADIUS
        radiuses.append(radius)
    temp_df['radius'] = radiuses
    print(radiuses)

    os.makedirs(new_db_path, exist_ok=True)
    params = []
    for _, row in temp_df.iterrows():
        old_relative_path = row['old_relative_path']
        relative_path = row['relative_path']
        radius = row['radius']
        old_path = os.path.join(db_path, old_relative_path)
        path = os.path.join(new_db_path, relative_path)
        if not os.path.exists(path):
            logging.info(f'Process {old_path} to {path}')
            params.append((cc_path, old_path, path, radius, args.format))
        else:
            logging.info(f'Process {old_path} to {path}: already exists')
    parallel_process(cc_convert, params, args.n_parallel)

    final_df = temp_df.copy()
    final_df = final_df.drop(columns='old_relative_path')
    df_path = os.path.join(new_db_path, 'dataset.csv')
    final_df.to_csv(df_path, index=False)

    logger.info(f'Dataset information written to {df_path}: {len(final_df)} rows.')
