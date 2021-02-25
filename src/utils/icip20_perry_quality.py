import os
import pandas as pd

pcs = [{'filename': 'longdress_vox10_1300.ply', 'name': 'longdress', 'geometry_bits': 10},
       {'filename': 'loot_vox10_1200.ply', 'name': 'loot', 'geometry_bits': 10},
       {'filename': 'redandblack_vox10_1550.ply', 'name': 'redandblack', 'geometry_bits': 10},
       {'filename': 'ricardo10_frame0082.ply', 'name': 'ricardo', 'geometry_bits': 10},
       {'filename': 'sarah9_frame0023.ply', 'name': 'sarah', 'geometry_bits': 9},
       {'filename': 'soldier_vox10_0690.ply', 'name': 'soldier', 'geometry_bits': 10}]
codecs = [{'name': 'Reference', 'id': 'REF'},
          {'name': 'GPCC_Octree', 'id': 'OCT', 'rates': ['r01', 'r02', 'r03', 'r04', 'r05', 'r06']},
          {'name': 'GPCC_Trisoup', 'id': 'TRI', 'rates': ['r01', 'r02', 'r03', 'r04', 'r05', 'r06']},
          {'name': 'VPCC', 'id': 'AI', 'rates': ['r01', 'r02', 'r03', 'r04', 'r05', 'rb01']}]


def build_df(db_path):
    df = []
    for pc in pcs:
        fn = pc['filename']
        sn = pc['name']
        geometry_bits = pc['geometry_bits']
        pc_name = os.path.splitext(fn)[0]

        for codec in codecs:
            codec_name, codec_id = codec['name'], codec['id']
            if codec_id == 'REF':
                cur_fn = fn
                relative_path = cur_fn
                absolute_path = os.path.join(db_path, relative_path)

                assert os.path.exists(absolute_path), absolute_path
                df.append({'pc_name': pc_name, 'codec_id': codec_id, 'geometry_bits': geometry_bits,
                           'codec_rate': 'ref', 'relative_path': relative_path})
            else:
                # redandblack trisoup point clouds not available (octree point clouds in trisoup archive)
                for codec_rate in codec['rates']:
                    ext = '.ply'

                    filename_codec_id = codec_id
                    if codec_id == 'AI':
                        filename_codec_id = 'ai'

                    # Naming convention differs here
                    if fn == 'ricardo10_frame0082.ply' and codec_id == 'AI':
                        cur_fn = '_'.join(('ricardo10', filename_codec_id, codec_rate)) + ext
                    else:
                        cur_fn = '_'.join((os.path.splitext(fn)[0], filename_codec_id, codec_rate)) + ext

                    folder = '_'.join((codec_name, sn))
                    relative_path = os.path.join(folder, cur_fn)
                    absolute_path = os.path.join(db_path, relative_path)

                    # This file is in off (NCOFF) format instead of ply
                    # Currently supported by meshlab but not CloudCompare
                    if fn == 'loot_vox10_1200.ply' and codec_id == 'AI' and codec_rate == 'r05':
                        # ext = '.off'
                        if not os.path.exists(absolute_path):
                            print('Manual action required: use meshlab or any other tool to convert {absolute_path} (currently .off) to a ply file (geometry and color). The file is in NCOFF format.')
                    elif fn == 'redandblack_vox10_1550.ply' and codec_id == 'TRI':
                        if not os.path.exists(absolute_path):
                            print('Manual action required: use GPCC to reproduce results if files are missing')
                    else:
                        assert os.path.exists(absolute_path), absolute_path
                    df.append({'pc_name': pc_name, 'codec_id': codec_id, 'geometry_bits': geometry_bits,
                               'codec_rate': codec_rate, 'relative_path': relative_path})

    return pd.DataFrame(df)


def load_mos(db_path):
    mos_df = pd.read_csv(os.path.join(db_path, 'MOS', 'AllLabsResultsSummary.csv'))
    # Rename columns to same name
    mos_df = mos_df.rename(columns={'Point_Cloud_Name': 'pc_name', 'Codec': 'codec_id', 'Rate': 'codec_rate',
                                    'MOS': 'mos', 'CI': 'mos_ci'})
    # Fix RRR0X instead of R0X
    mask = (mos_df['pc_name'] == 'sarah9_frame0023') & (mos_df['codec_id'] == 'TRI')
    mos_df.loc[mask, 'codec_rate'] = mos_df[mask]['codec_rate'].map(lambda x: x[2:])
    # Lowercase codec rates
    mos_df['codec_rate'] = mos_df['codec_rate'].map(lambda s: s.lower())
    return mos_df


def merge_df_mos(df, mos_df):
    merged_df = pd.merge(df, mos_df, how='inner', on=['pc_name', 'codec_id', 'codec_rate'], sort=False,
                         suffixes=('', '_mos_df'))

    # Rows in df but without corresponding mos rows
    # Seems that r03 MOS are not in the provided file
    lmiss = pd.merge(df, mos_df, how='left', on=['pc_name', 'codec_id', 'codec_rate'], sort=False,
                     suffixes=('', '_mos_df'))

    # Rows not in df but with mos rows
    # redandblack TRI point clouds are missing
    # romanoillamp is missing (not available for download)
    rmiss = pd.merge(df, mos_df, how='right', on=['pc_name', 'codec_id', 'codec_rate'], sort=False,
                     suffixes=('', '_mos_df'))

    return {'merged_df': merged_df, 'lmiss': lmiss, 'rmiss': rmiss}


def load(db_path):
    df = build_df(db_path)
    mos_df = load_mos(db_path)
    data = merge_df_mos(df, mos_df)
    return data
