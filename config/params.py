import os
import numpy as np

class Struct:
  def __init__(self, *dicts, **fields):
    for d in dicts:
      for k, v in d.iteritems():
        setattr(self, k, v)
    self.__dict__.update(fields)

  def to_dict(self):
    return {a: getattr(self, a) for a in self.attrs()}

  def attrs(self):
    #return sorted(set(dir(self)) - set(dir(Struct)))
    xs = set(dir(self)) - set(dir(Struct))
    xs = [x for x in xs if ((not (hasattr(self.__class__, x) and isinstance(getattr(self.__class__, x), property))) \
        and (not inspect.ismethod(getattr(self, x))))]
    return sorted(xs)

  def updated(self, other_struct_=None, **kwargs):
    s = copy.deepcopy(self)
    if other_struct_ is not None:
      s.__dict__.update(other_struct_.to_dict())
    s.__dict__.update(kwargs)
    return s

  def copy(self):
    return copy.deepcopy(self)

  def __str__(self):
    attrs = ', '.join('%s=%s' % (a, getattr(self, a)) for a in self.attrs())
    return 'Struct(%s)' % attrs


class Params(Struct):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


def base(name, vid_dur=2.0, samp_sr=32000):
    if vid_dur is None:
        vid_dur = VidDur
    fps = 10
    frame_dur = 1./fps

    pr = Params(
        fps = 30,
        total_dur = 5,
        vid_dur = vid_dur,
        frame_rate = 1,
        samp_sr = samp_sr,
        log_spec = True,
        hop_length = 161,
        n_mel = 64,
        spec_min=-100.,
        spec_max = 100.,
        num_samples = 0,
        mono = True,

        dropout = 0.0,
        img_tau = 0.07,
        aud_tau = 0.07,
        epsilon = 0.01,
        psize = 32, # 48, 64
        patch_stride = 28,
        list_train = None,
        list_val = None,
        list_test = None,

        seed=1234
    ) 

    return pr


def music_multi_nodes(**kwargs): 
    pr = base('music_multi_nodes', **kwargs)
    pr.fps = 4
    pr.samp_sr = 16000
    pr.log_spec = True
    pr.vid_dur = 0.96
    pr.hop_length = 161
    pr.n_mel = 64
    pr.spec_max = 100.
    pr.num_samples = int(round(pr.samp_sr * pr.vid_dur))
    pr.mono = True
    pr.dropout = 0.0
    pr.img_tau = 0.07
    pr.aud_tau = 0.07
    pr.epsilon = 0.01
    pr.psize = 64 # 48, 32
    pr.patch_stride = 32
    pr.img_size = 224
    pr.fc_type = '2_layers' # options: '2_layers', "3_layers"
    pr.feat_dim = 128
    pr.list_train = 'data/MUSIC/data-splits/solo/train.csv'
    pr.list_val = 'data/MUSIC/data-splits/solo/val.csv'
    pr.list_test = 'data/MUSIC/data-splits/solo/test.csv'
    pr.data_base_path = 'data/MUSIC'
    return pr