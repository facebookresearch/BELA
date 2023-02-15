# HACK: Need to import protobuf before pytorch_lightning to prevent Segmentation Fault: https://github.com/protocolbuffers/protobuf/issues/11934
from google.protobuf import descriptor as _descriptor
