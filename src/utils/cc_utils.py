from .parallel_process import Popen

def cc_convert(cc_path, input_path, output_path, radius, format):
    radius_str = str(radius)
    cmd = [cc_path, '-SILENT', '-AUTO_SAVE', 'OFF', '-O', input_path,
           '-OCTREE_NORMALS', radius_str,
           '-ORIENT_NORMS_MST', radius_str,
           '-CURV', 'MEAN', radius_str,
           '-CURV', 'GAUSS', radius_str,
           '-ROUGH', radius_str,
           '-C_EXPORT_FMT', 'PLY', '-PLY_EXPORT_FMT', format, '-SAVE_CLOUDS', 'FILE', output_path]
    return Popen(cmd)
