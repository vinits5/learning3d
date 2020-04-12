from .rmse_features import RMSEFeaturesLoss
from .frobenius_norm import FrobeniusNormLoss
from .classification import ClassificationLoss
try:
	from .emd import EMDLoss
except:
	print("Sorry EMD loss is not compatible with your system!")
try:
	from .chamfer_distance import ChamferDistanceLoss
except:
	print("Sorry ChamferDistance loss is not compatible with your system!")