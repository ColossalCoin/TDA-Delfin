from data.image_converter import get_images
import numpy as np
from itertools import product
from gtda.images import Binarizer, RadialFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split

np.random.seed(123)
tumor_yes, tumor_no = get_images((224, 224))

threshold_iter = np.arange(0.1, 1, 0.1)
center_iter = product({1, -1, 0}, {1, -1, 0})

binarizers = ([Binarizer(threshold=threshold) for threshold in threshold_iter])
radial_filtrations = ([RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_iter])

steps = [
    [
        binarizer,
        radial_filtration,
        #HeatKernel(sigma=0.15, n_bins=60, n_jobs=-1),
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1)
    ] for radial_filtration in radial_filtrations
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

tda_pipeline = make_union(
    *[make_pipeline(*step, amplitudes_union) for step in steps], n_jobs=-1
)

X = np.concatenate((tumor_yes, tumor_no))
y = np.concatenate((np.ones(tumor_yes.shape[0]), np.zeros(tumor_no.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

X_train_pipeline = tda_pipeline.fit_transform(X_train)
