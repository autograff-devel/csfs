from importlib import reload
import numpy as np
import time
import matplotlib.pyplot as plt
import autograff.geom as geom
import autograff.utils as utils
import autograff.plut as plut
import autograff.ttf as ttf
import autograff.geom.clipper_wrap as clip
import autograff.imaging as imaging
reload(imaging)

import os
import sys
import pdb

from PIL import Image, ImageDraw, ImageFont
import cv2

from collections import defaultdict

brk = pdb.set_trace


class perf_timer:
    def __init__(self, name=''):
        if name:
            print("Timing: " + name)
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.t)*1000
        if self.name:
            print('%s: elapsed time %.3f' % (self.name, self.elapsed))


class GridPlot:
    def __init__(self, iterator, callback, figsize=(2.3, 2.3), max_cols=5):
        self.iterator = iterator
        self.callback = callback
        self.timing = {}
        self.ax = None
        self.cur_name = ''
        self.S = []
        self.figs = {}
        self.figsize = figsize
        self.max_cols = max_cols
        self.glyph_data = {}

    def get_stats(self):
        stats = {key: np.mean(vals) for key, vals in self.timing.items()}
        for key, t in stats.items():
            print(key + ' average time:' + str(t/1000.))
        return stats

    def new_figure(self, name, key):
        self.figs[key] = plut.MultiFigure()
        self.figs[key].begin(max_cols=self.max_cols,
                             title=name + ' ' + key, size=self.figsize, id=key)

    def begin_draw(self, key):
        if not key in self.figs:
            self.new_figure(self.cur_name, key)
        self.ax = self.figs[key].add_subplot()

    def end_draw(self, padding=20):
        if self.glyph_data:
            #gbox = geom.make_rect(0, 0, self.glyph_data['height'], self.glyph_data['height'])
            sbox = geom.bounding_box(self.S, padding)
            cenp = geom.rect_center(sbox)
            w = max(geom.rect_w(sbox), self.glyph_data['height'])
            h = max(geom.rect_h(sbox), self.glyph_data['height'])
            box = geom.make_centered_rect(cenp, [w, h])

        else:
            box = geom.bounding_box(self.S, padding)
        # geom.bounding_box(self.S, 30), ax=self.ax)
        plut.setup(axis=False, ax=self.ax, axis_limits=box)
        self.ax = None

    def add_timing(self, id, t):
        if not id in self.timing:
            self.timing[id] = []
        self.timing[id].append(t)

    def run(self, save_path='', start_from="", max_count=200):
        k = 0
        cur_name = ''
        count = 0

        for S, name, char, glyph_data in self.iterator:
            self.cur_name = name
            self.S = S

            if name == 'Origins':
                print('skipping %s: %s' % (name, char))
                continue
            if start_from:
                if name != start_from:
                    continue
                else:
                    start_from = ''

            if 'type' in glyph_data:
                kind = glyph_data['type']
                self.glyph_data = glyph_data
            else:
                kind = 'unknown'
                self.glyph_data = {}

            def new_fig_cond():
                if kind == 'ttf':
                    return name != cur_name
                else:
                    return k > 25 or cur_name == ''

            bail = False
            k += 1

            if new_fig_cond():
                for key, mfig in list(self.figs.items()):
                    if cur_name:
                        if kind != 'ttf' and count >= max_count:
                            bail = True
                            break
                        else:
                            if save_path:
                                mfig.end(os.path.join(
                                    save_path, cur_name + '_' + key + '.pdf'))
                            else:
                                mfig.end()
                    if not bail:
                        self.figs[key] = plut.MultiFigure()
                        self.figs[key].begin(
                            max_cols=self.max_cols, title=name + ' ' + key, size=self.figsize, id=key)
                # endfor

                k = 1

                if count:
                    print('Finished ' + cur_name + ' group %d' % (count))
                    self.get_stats()

                if cur_name:
                    print(cur_name)

                cur_name = name
            # endif
            if bail:
                break

            count += 1
            print('%d Computing %s: %s' % (count, name, char))
            if not S:
                print('--- No shape')
                continue

            self.callback(self, S, name, char, glyph_data)
        #endfor (iterate)

        for key, mfig in self.figs.items():
            if save_path:
                mfig.end(os.path.join(save_path, name + '_' + key + '.pdf'))
            else:
                mfig.end()

        return self.get_stats()
    # endf
# endcls


def load_glyph(db, font_index, char, size=150, resample=True):
    try:
        S = db.get_shape(font_index, char)
        glyph_h = db.get_font_height(font_index)
        # pdb.set_trace()
        # BIG HACK! need to sort out winding functions
        # and add winding flag to path-symmmetry funcs
        S = clip.union(S, S)

        if not resample:
            return S

        #S = [P[:,::-1] for P in S]
        # brk()

        # pdb.set_trace()
        S, ratio = geom.rescale_and_sample_vertical(
            S, closed=True, height=glyph_h, dest_height=size, get_ratio=True)
        S = [P/ratio for P in S]
        S = geom.fix_shape_winding(S, True)

        glyph_data = {'height': glyph_h,
                      'width': db.char_width(font_index, char),
                      'ratio': ratio,
                      'target_size': size,
                      'type': 'ttf'}

    except (KeyError, IOError) as err:
        print('Failed loading font: ' + font_index)
        raise ValueError
    return S, glyph_data


def load(what, size=150, closed=True, chars='A', keep_scale=False):
    if type(what) == str:
        if '.svg' in what:
            return load_svg(what, size, union=False, keep_scale=keep_scale, closed=closed)
        elif '.pkl' in what:
            # Assume pkl is a bspline
            print('load: pkl file, assuming Bspline control points')
            pts = utils.load_pkl(what)
            P = geom.bspline(400, pts, ds=2., closed=closed)
            Ps = preprocess_shape(P, size, keep_scale=keep_scale)
            if closed:
                S = geom.fix_shape_winding([Ps], True)
            else:
                S = [Ps]
            return S
        elif '.png' in what or '.jpg' in what:
           im = cv2.imread(what, 0) #cv2.CV_LOAD_IMAGE_GRAYSCALE)
           S = imaging.find_contours(im)
           S = preprocess_shape(S, size, keep_scale=keep_scale)
           S = geom.fix_shape_winding(S, True)
           return S
        elif '.ttf' in what:
            path = what
            glyph_h = ttf.glyph_height(path)
            shapes = []
            for char in chars:
                S = ttf.glyph_shape(path, char, 40)

                S, ratio = geom.rescale_and_sample_vertical(S,
                                                            closed=True,
                                                            height=glyph_h,
                                                            dest_height=size,
                                                            get_ratio=True)
                if keep_scale:
                    S = [P/ratio for P in S]
                S = geom.fix_shape_winding(S, True)
                shapes.append(S)
            if len(shapes)==1:
                return shapes[0]
            return shapes
        else:
            print('load: Unsupported input type')
            raise ValueError
    else:
            print('load: Assuming GylphDatabase is provided')
            return load_glyph(*what, size=size)[0]  # <- assume it is a glyph

########################################################
# Shape rasterization utils


def shape_to_outline(S):
    s = ImageDraw.Outline()
    for P in S:
        P = P.T
        s.move(*P[0])
        for p in P[1:]:
            s.line(*p)
        s.close()
    return s
# end


def raster_shape(shape, raster_size, tm):
    im = Image.new('L', (raster_size, raster_size), (0))

    shape = [geom.affine_mul(tm, P) for P in shape]
    draw = ImageDraw.Draw(im)
    draw.shape(shape_to_outline(shape), 255)
    im = np.array(im)
    return im
# end


def rasterize_shape(shape, raster_size, src_rect=None, get_im_and_mat=True, im_mat=None, padding=3, outline_shape=[]):
    ''' Generates raster samples for the foreground a shape'''
    # Build character image
    im = Image.new('L', (raster_size, raster_size), (0))

    if im_mat is None:
        dst_rect = geom.make_rect(0, 0, raster_size, raster_size)
        if src_rect is None:
            # pdb.set_trace()
            src_rect = geom.bounding_box(shape)
        im_mat = geom.rect_in_rect_transform(
            src_rect, dst_rect, padding=padding)
    # endif

    shape = [geom.affine_mul(im_mat, P) for P in shape]

    draw = ImageDraw.Draw(im)

    draw.shape(shape_to_outline(shape), 255)
    if outline_shape:
        outline_shape = [geom.affine_mul(im_mat, P) for P in outline_shape]
        for P in outline_shape:
            draw.polygon([tuple(p) for p in P.T], None, 255)
        #draw.shape(shape_to_outline(outline_shape), None, 255)

    im = np.array(im)
    # dsds
    if get_im_and_mat:
        return im, im_mat

    return im
# end


def to_bmp(im):
    return (im > 0).astype(float)


def sample_shape(shape, raster_size, draw_samples=False, src_rect=None, get_im_and_mat=True, im_mat=None, padding=3):
    ''' Generates raster samples for the foreground a shape'''
    # Build character image
    im = Image.new('L', (raster_size, raster_size), (0))

    if im_mat is None:
        dst_rect = geom.make_rect(0, 0, raster_size, raster_size)
        if src_rect is None:
            # pdb.set_trace()
            src_rect = geom.bounding_box(shape)
        im_mat = geom.rect_in_rect_transform(
            src_rect, dst_rect, padding=padding)
    # endif

    shape = [geom.affine_mul(im_mat, P) for P in shape]

    draw = ImageDraw.Draw(im)
    draw.shape(shape_to_outline(shape), 255)
    im = np.array(im)
    # pdb.set_trace()
    #ernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16,16))
    #im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    im = im.T

    X = np.vstack(np.where(im > 0))
    #X[1,:] = raster_size - X[1,:]
    X = geom.affine_mul(np.linalg.inv(im_mat), X).T

    if draw_samples:
        plt.plot(X[:, 0], X[:, 1], 'ko', markersize=0.9, alpha=0.1)

    if get_im_and_mat:
        return X, im.T, im_mat

    return X
# end

# def shape_shape_intersections(A, B, closed=False):


def load_svg(path, size, flip_y=False, union=True, closed=True, get_data=False, keep_scale=True):
    import autograff.svg as svg
    # reload(svg)
    S = svg.load_svg(path)
    Sorig = S

    if union:
        # Sorig = []
        # for P in S:
        #     Po = clip.offset(P, 3)
        #     Sorig += Po
        #S = Sorig
        S = clip.offset(S, 5, 'round')
        S = clip.union(S, S)
        S = clip.offset(S, -5, 'round')

        #S = clip.offset(S, -5)
        #S = clip.offset(S, -0.5)
    else:
        Sorig = S

    S, ratio = geom.rescale_and_sample_vertical(
        S, closed=closed, height=0, dest_height=size, get_ratio=True)
    if flip_y:
        for i, P in enumerate(S):
            S[i][1, :] = -P[1, :]
    # brk()
    if keep_scale:
        S = [P/ratio for P in S]
    if closed:
        S = geom.fix_shape_winding(S, True)
    #S = [P[:,::-1] for P in S]
    glyph_data = {'height': size/ratio,
                  'width': size,
                  'ratio': ratio,
                  'target_size': size,
                  'type': 'svg'}
    if union:
        glyph_data['original_shapes'] = Sorig

    if not get_data:
        return S

    return S, glyph_data


def preprocess_shape(S, size, get_data=False, keep_scale=True):
    import autograff.svg as svg
    if type(S) != list:
        S = [S]
        is_contour = True
        print('Contour')
    else:
        print('Not ctr')
        is_contour = False

    S, ratio = geom.rescale_and_sample_vertical(
        S, closed=True, height=0, dest_height=size, get_ratio=True)
    if keep_scale:
        S = [P/ratio for P in S]
    S = geom.fix_shape_winding(S)

    glyph_data = {'height': size/ratio, 'width': size,
                  'ratio': ratio, 'target_size': size}
    if is_contour:
        S = S[0]

    if not get_data:
        return S

    return S, glyph_data


def iterate_chars(font_inds, chars):
    for font_ind in font_inds:
        for char in chars:
            yield (font_ind, char)


class FontIterator:
    ''' Iterate TTF fonts in dir'''

    def __init__(self, path):
        self.db = ttf.FontDatabase(path)
        print('Initializing font databse:')
        self.db.list_font_names()
        self.type = 'ttf'

    def load_glyph(self, name, char, size=100):
        return load_glyph(self.db, name, char, size=size)

    def load_original(self, name, char):
        return load_glyph(self.db, name, char, resample=False)

    def iterate_fonts(self):
        for name in self.db.db.keys():
            yield name

    def iterate(self, fonts=[], chars='', char_map={}, size=100):
        if fonts and type(fonts[0]) != str:
            char_map = {}
            for name, chrs in fonts:
                char_map[name] = chrs
            fonts = [name for name, _ in fonts]
            chars = ''
        for name in self.db.db.keys():
            if fonts and not name in fonts:
                continue

            if not chars:
                found_map = False
                if char_map:
                    for name_part, characters in char_map.items():
                        if name_part in name:
                            chrs = characters
                            found_map = True
                            break
                if not found_map:
                    if 'Hebrew Bold' in name:
                        chrs = 'אבגדהוזחטיכלמנסעפצקרשת'
                    elif 'Kaiti' in name:
                        chrs = '書法书'  # 法这句话后来演变成“饮水思源”这个成语' #披薩好吃' #
                    elif ('Kazuraki' in name) or ('Std B' in name):
                        # pdb.set_trace()
                        chrs = '未来を変えろ好きな言葉のカタチを造る'
                    else:
                        chrs = 'ABCDEFGHIJKLMNOPQRSTXYZW'  # 'hamburgefontsABCDEFGHIJKLMNOPQRSTUVWXYZ'
            else:
                chrs = chars

            for char in chrs:
                try:
                    shape, glyph_data = load_glyph(self.db, name, char, size)
                    yield shape, name, char, glyph_data
                except ValueError:
                    continue
# endcls


class SvgIterator:
    ''' Iterate svg's in dir'''

    def __init__(self, path, flip_y=True):
        self.paths = [f for f in utils.files_in_dir(path) if '.svg' in f]
        self.flip_y = flip_y
        self.type = 'svg'

    def iterate(self, size=100, paths=[], union=False):
        for path in self.paths:
            if paths and utils.filename_without_ext(path) not in paths:
                # print(utils.filename_without_ext(path))
                continue
            shape, glyph_data = load_svg(
                path, size=size, flip_y=self.flip_y, union=union, get_data=True)
            # pdb.set_trace()
            yield shape, utils.filename_without_ext(path), ' ', glyph_data
# endcls


def load_shape_structure(path, size, flip_y):
    ''' Load file from Shape Structure Dataset
        http://2dshapesstructure.github.io'''

    import json
    with open(path) as json_file:
        data = json.load(json_file)
        points = data['points']
        P = []
        for p in points:
            P.append([p['x'], p['y']])
        P = np.array(P).T
        S, ratio = geom.rescale_and_sample(
            [P], closed=True, scale=size, get_ratio=True)
        if flip_y:
            for i, P in enumerate(S):
                S[i][1, :] = -P[1, :]

        S = geom.fix_shape_winding(S)

        dest_ratio = 900/size
        S = [(P*dest_ratio) for P in S]

        return S


class ShapeStructureIterator:
    def __init__(self, path, flip_y=True):
        self.paths = [f for f in utils.files_in_dir(path) if '.json' in f]
        self.flip_y = flip_y
        self.type = 'shape_structure'

    def iterate(self, size=100, paths=[], union=False):
        for path in self.paths:
            if paths and utils.filename_without_ext(path) not in paths:
                # print(utils.filename_without_ext(path))
                continue
            yield load_shape_structure(path, size=size, flip_y=self.flip_y), utils.filename_without_ext(path), ' ', {'height': 900, 'width': 900, 'ratio': 1., 'target_size': 900}
# endcls


class FontDataSource:
    ''' Save/Load data for a font to a folder
        uses single files for entries, otherwise loading would be very slow
        and whole dataset takes waaay too much memory (potentially > 1gb)
    '''

    def __init__(self, path, prefix):
        self.prefix = prefix
        self.path = path
        utils.ensure_abspath(self.path)
        self.paths = {utils.filename_without_ext(
            f): f for f in utils.files_in_dir(path) if '.pkl' in f}

    def size(self):
        return len(self.paths)

    def make_name(self, name, char, forcelower=False):
        # so it seems that python/osx does not overwrite case sensitive. but loads case sensitive??
        if char.islower() and not forcelower:
            char = 'lower_'+char  # why.. but why
        return self.prefix + '_' + name + '_' + char

    def make_path(self, name, char):
        return os.path.join(self.path, self.make_name(name, char) + '.pkl')

    def add_entry(self, entry, name, char=' '):
        utils.save_pkl(entry, self.make_path(name, char))

    def has_entry(self, name, char=' ', forcelower=False):
        return self.make_name(name, char, forcelower) in self.paths

    def load_entry(self, name, char=' '):
        if self.has_entry(name, char):
            return utils.load_pkl(self.paths[self.make_name(name, char)])
        if self.has_entry(name, char.lower(), True):
            return utils.load_pkl(self.paths[self.make_name(name, char.lower(), True)])
        return {}

    def iterate_entries(self, font_subset=[], only_chars=''):
        entries = defaultdict(list)
        for filename, path in self.paths.items():
            parts = filename.split('_')
            name = '_'.join(parts[1:-1])
            char = parts[-1]
            #_, name, char = filename.split('_')
            entries[name].append(char)

        for name, chars in entries.items():
            chars = sorted(chars)
            if font_subset and not name in font_subset:
                continue
            for char in chars:
                if only_chars and not char in only_chars:
                    continue
                yield name, char
# endcls
