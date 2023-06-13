# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# HACK: Need to import protobuf before pytorch_lightning to prevent Segmentation Fault: https://github.com/protocolbuffers/protobuf/issues/11934
from google.protobuf import descriptor as _descriptor
