from data.image_converter import get_images
import numpy as np
from itertools import product
from gtda.images import Binarizer, RadialFiltration, DensityFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split
import pickle
from os import getcwd

np.random.seed(123)
tumor_yes, tumor_no = get_images((224, 224))

thresholds = np.arange(0.1, 1, 0.1)
centers = product({37, 112, 187}, {37, 112, 187})

binarizers = ([Binarizer(threshold=threshold, n_jobs=-1) for threshold in thresholds])
radial_filtrations = ([RadialFiltration(center=np.asarray(center), n_jobs=-1) for center in centers])
density_filtrations = ([DensityFiltration(radius=1, metric='euclidean', n_jobs=-1), 
                        DensityFiltration(radius=10, metric='cityblock', n_jobs=-1)])

steps_original = [
    [
        filtration,
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1)
    ] for filtration in radial_filtrations 
]

steps_binarizer = [
    [
        binarizer,
        filtration,
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1)
    ] for filtration in density_filtrations
    for binarizer in binarizers
]

metric_iter = [
    {'metric':'bottleneck', 'metric_params':{}},
    {'metric':'wasserstein', 'metric_params':{'p':1}},
    {'metric':'wasserstein', 'metric_params':{'p':2}},
    {'metric':'landscape', 'metric_params':{'p':1, 'n_layers':1, 'n_bins':100}},
    {'metric':'landscape', 'metric_params':{'p':1, 'n_layers':2, 'n_bins':100}},
    {'metric':'landscape', 'metric_params':{'p':2, 'n_layers':1, 'n_bins':100}},
    {'metric':'landscape', 'metric_params':{'p':2, 'n_layers':2, 'n_bins':100}},
    {'metric':'betti', 'metric_params':{'p':1, 'n_bins':100}},
    {'metric':'betti', 'metric_params':{'p':2, 'n_bins':100}},
    {'metric':'heat', 'metric_params':{'p':1, 'sigma':1.6, 'n_bins':100}},
    {'metric':'heat', 'metric_params':{'p':1, 'sigma':3.2, 'n_bins':100}},
    {'metric':'heat', 'metric_params':{'p':2, 'sigma':1.6, 'n_bins':100}},
    {'metric':'heat', 'metric_params':{'p':2, 'sigma':3.2, 'n_bins':100}}
]
amplitudes = ([Amplitude(**metric, n_jobs=-1) for metric in metric_iter])
amplitudes_union = make_union(*[PersistenceEntropy(nan_fill_value=-1)] + amplitudes)

pipe_original = make_union(
    *[make_pipeline(*step, amplitudes_union) for step in steps_original], n_jobs=-1
)
pipe_binarizer = make_union(
    *[make_pipeline(*step, amplitudes_union) for step in steps_binarizer], n_jobs=-1
)
tda_pipeline = make_union(pipe_original, pipe_binarizer)

with open(r'models\tda_pipeline.pickle', 'wb') as file:
    pickle.dump(tda_pipeline, file)
    file.close()