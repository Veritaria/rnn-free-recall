from .supervised import FreeRecallSumMSELoss, FreeRecallSumMSEMultipleOutputLoss, \
    FreeRecallSumMSETrainEncodeLoss, FreeRecallSumCETrainEncodeLoss, \
    EncodingCrossEntropyLoss, EncodingNBackCrossEntropyLoss
from .rl import A2CLoss
from .auxiliary import MemoryOneHotRegularization
from .criterion import MultiSupervisedLoss, MultiRLLoss, MultiAuxiliaryLoss
