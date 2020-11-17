# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys

from pesq import pesq
from pystoi import stoi
import torch

from .data import NoisyCleanSet
from .enhance import add_flags, get_estimate
from . import distrib, pretrained
from .utils import bold, LogProgress

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    "denoiser.evaluate", description="Speech enhancement using Demucs - Evaluate model performance"
)
add_flags(parser)
parser.add_argument("--data_dir", help="directory including noisy.json and clean.json files")
parser.add_argument("--matching", default="sort", help="set this to dns for the dns dataset.")
parser.add_argument(
    "--no_pesq", action="store_false", dest="pesq", default=True, help="Don't compute PESQ."
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_const",
    const=logging.DEBUG,
    default=logging.INFO,
    help="More loggging",
)


def evaluate(args, model=None, data_loader=None):
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0
    updates = 5

    # Load model
    # if not model:
    #     model = pretrained.get_model(args).to(args.device)
    # model.eval()

    model = pretrained.Demucs(
        hidden=48,
        depth=4,
        kernel_size=8,
        stride=4,
        causal=True,
        resample=1,
        growth=2,
        max_hidden=10_000,
        normalize=True,
        glu=True,
        rescale=0.1,
        floor=1e-3,
    )
    a = 0
    for p in model.parameters():
        a += p.numel()
    print("model size", a)
    model.eval()

    # Load data
    if data_loader is None:
        dataset = NoisyCleanSet(args.data_dir, matching=args.matching, sample_rate=args.sample_rate)
        # data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)
    import torch

    x = dataset[0][0]
    length = x.shape[-1]
    from torch.nn import functional as F

    x = F.pad(x, (0, model.valid_length(length) - length))  # 222 (before: 81271)
    # x = F.pad(x, (0, 32_084-len(x)))
    # x = x.unsqueeze(1)
    x = x.unsqueeze(1)
    model(x)

    torch.onnx.export(model, x, "denoiser.onnx", verbose=True, opset_version=11)
    # from serialization import pytorch_converter
    # pytorch_converter.convert('denoiser.onnx', image_input_names=None)

    import onnx
    import onnx_coreml

    m = onnx.load_model("denoiser.onnx")
    coreml_model = onnx_coreml.convert(m)#, minimum_ios_deployment_target="13")

    # coreml_model.save("denoiser.mlmodel")
    from serialization import utils

    # preprocess_dict = {
    #     "divisibleBy" : 32084,
    #     "resizeStrategy" : "Fixed",
    #     "sideLength" : 32084,
    #     "classes" : None,
    # }
    # [kDivisibleBy, kResizeStrategy, kSideLength, kClasses]
    preprocess_dict = utils.create_preprocess_dict(
        model.valid_length(length), "Fixed", side_length=model.valid_length(length), output_classes="irrelevant"
    )
    # utils.compress_file('denoiser.mlmodel', 'denoiser.nnmodel')
    utils.compress_and_save(
        coreml_model,
        ".",
        "denoiser_48_depth_4",
        "1.0.0",
        "",
        preprocess_dict,
        "",
        convert_to_float16=False,
    )
    print('length ',model.valid_length(length) )
    import numpy as np

    inp = torch.from_numpy(np.random.random((1,1,model.valid_length(length)))).float()
    out1 = coreml_model.predict({"0": inp.numpy()})["138"]
    out2 = model(inp)
    np.testing.assert_array_almost_equal(out1.squeeze(), out2.detach().numpy().squeeze())

    import sys

    sys.exit()
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                noisy, clean = [x.to(args.device) for x in data]
                # If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(_estimate_and_run_metrics, clean, model, noisy, args))
                else:
                    estimate = get_estimate(model, noisy, args)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(_run_metrics, clean, estimate, args))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            pesq_i, stoi_i = pending.result()
            total_pesq += pesq_i
            total_stoi += stoi_i

    metrics = [total_pesq, total_stoi]
    pesq, stoi = distrib.average([m/total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance:PESQ={pesq}, STOI={stoi}.'))
    return pesq, stoi


def _estimate_and_run_metrics(clean, model, noisy, args):
    estimate = get_estimate(model, noisy, args)
    return _run_metrics(clean, estimate, args)


def _run_metrics(clean, estimate, args):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]
    if args.pesq:
        pesq_i = get_pesq(clean, estimate, sr=args.sample_rate)
    else:
        pesq_i = 0
    stoi_i = get_stoi(clean, estimate, sr=args.sample_rate)
    return pesq_i, stoi_i


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    pesq, stoi = evaluate(args)
    json.dump({'pesq': pesq, 'stoi': stoi}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
