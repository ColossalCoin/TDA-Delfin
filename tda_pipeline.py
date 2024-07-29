import numpy as np
from gtda.images import Binarizer, RadialFiltration, DensityFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, Amplitude, NumberOfPoints
from sklearn.pipeline import make_pipeline, make_union

thresholds = np.arange(0.4, 0.7, 0.35/10)
centers = [(84, 56), (56, 112), (84, 140), (140, 56), (168, 112), (140, 140)]

binarizers = ([Binarizer(threshold=threshold, n_jobs=-1) for threshold in thresholds])
radial_filtrations = [RadialFiltration(center=np.asarray(center), n_jobs=-1) for center in centers]
density_filtration = [DensityFiltration(radius=14, metric='l1', n_jobs=-1)]
filtrations = (radial_filtrations + density_filtration)

steps_original = [
    [
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1),
        Filtering(homology_dimensions=[0, 1], epsilon=0.1)
    ]
]

steps_binarizer = [
    [
        binarizer,
        filtration,
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1),
        Filtering(homology_dimensions=[0, 1], epsilon=0.1)
    ] for filtration in filtrations
    for binarizer in binarizers
]

metrics = [
    {'metric':'bottleneck'},
    {'metric':'wasserstein', 'metric_params':{'p':1}},
    {'metric':'wasserstein', 'metric_params':{'p':2}},
    {'metric':'betti', 'metric_params':{'p':1, 'n_bins':30}},
    {'metric':'betti', 'metric_params':{'p':2, 'n_bins':30}},
    {'metric':'landscape', 'metric_params':{'p':1, 'n_bins':20, 'n_layers':1}},
    {'metric':'landscape', 'metric_params':{'p':1, 'n_bins':20, 'n_layers':2}},
    {'metric':'landscape', 'metric_params':{'p':2, 'n_bins':20, 'n_layers':1}},
    {'metric':'landscape', 'metric_params':{'p':2, 'n_bins':20, 'n_layers':2}},
    {'metric':'silhouette', 'metric_params':{'p':1, 'n_bins':100, 'power':0.1}},
    {'metric':'silhouette', 'metric_params':{'p':2, 'n_bins':100, 'power':0.1}},
    {'metric':'heat', 'metric_params':{'p':1, 'sigma':0.5, 'n_bins':100}},
    {'metric':'heat', 'metric_params':{'p':2, 'sigma':0.5, 'n_bins':100}},
    {'metric':'persistence_image', 'metric_params':{'p':1, 'sigma':0.6, 'n_bins':100}},
    {'metric':'persistence_image', 'metric_params':{'p':2, 'sigma':0.6, 'n_bins':100}}        
]

amplitudes = ([Amplitude(**metric, n_jobs=-1) for metric in metrics])
amplitudes_union = make_union(
    *[
      PersistenceEntropy(nan_fill_value=-1, n_jobs=-1), NumberOfPoints(n_jobs=-1)
      ] + amplitudes
    )

pipe_original = make_union(
    *[make_pipeline(*step, amplitudes_union) for step in steps_original], n_jobs=-1
)
pipe_binarizer = make_union(
    *[make_pipeline(*step, amplitudes_union) for step in steps_binarizer], n_jobs=-1
)

tda_pipeline = make_union(pipe_original, pipe_binarizer)

def pipeline(images, original=False):
    if original:
        return pipe_original.fit_transform(images)
    return tda_pipeline.fit_transform(images)