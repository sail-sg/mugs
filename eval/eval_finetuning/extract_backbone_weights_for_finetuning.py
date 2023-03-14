# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
extract the backbone weight from a pretraiend model.
Most are copyed from iBOT library:
https://github.com/bytedance/ibot
"""
import torch
import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--output', type=str, help='destination file name')
    parser.add_argument("--checkpoint_key", default="state_dict", type=str, help='Key to use in the checkpoint (example: "teacher")')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict()
    has_backbone = False
    for key, value in ck[args.checkpoint_key].items():
        if key.startswith('backbone'):
            output_dict[key[9:]] = value
            has_backbone = True
        elif key.startswith('module.backbone'):
            output_dict[key[16:]] = value
            has_backbone = True
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)

if __name__ == '__main__':
    main()