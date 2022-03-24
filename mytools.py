def init_resnet_multiblocks(model, args):
    if args.distributed:
        for blockchoice in model.module.multiblocks:
            for layerchoice in blockchoice[1:]:
                if layerchoice.bn1.weight.data.shape == blockchoice[0].bn1.weight.data.shape:
                    layerchoice.bn1.weight.data = blockchoice[0].bn1.weight.data.clone().detach()
                    layerchoice.bn1.bias.data = blockchoice[0].bn1.bias.data.clone().detach()
                    layerchoice.bn1.running_mean.data = blockchoice[0].bn1.running_mean.data.clone().detach()
                    layerchoice.bn1.running_var.data = blockchoice[0].bn1.running_var.data.clone().detach()
                    layerchoice.bn1.num_batches_tracked.data = blockchoice[0].bn1.num_batches_tracked.data.clone().detach()
                if layerchoice.bn2.weight.data.shape == blockchoice[0].bn2.weight.data.shape:
                    layerchoice.bn2.weight.data = blockchoice[0].bn2.weight.data.clone().detach()
                    layerchoice.bn2.bias.data = blockchoice[0].bn2.bias.data.clone().detach()
                    layerchoice.bn2.running_mean.data = blockchoice[0].bn2.running_mean.data.clone().detach()
                    layerchoice.bn2.running_var.data = blockchoice[0].bn2.running_var.data.clone().detach()
                    layerchoice.bn2.num_batches_tracked.data = blockchoice[0].bn2.num_batches_tracked.data.clone().detach()
                if layerchoice.bn3.weight.data.shape == blockchoice[0].bn3.weight.data.shape:
                    layerchoice.bn3.weight.data = blockchoice[0].bn3.weight.data.clone().detach()
                    layerchoice.bn3.bias.data = blockchoice[0].bn3.bias.data.clone().detach()
                    layerchoice.bn3.running_mean.data = blockchoice[0].bn3.running_mean.data.clone().detach()
                    layerchoice.bn3.running_var.data = blockchoice[0].bn3.running_var.data.clone().detach()
                    layerchoice.bn3.num_batches_tracked.data = blockchoice[0].bn3.num_batches_tracked.data.clone().detach()
                if layerchoice.conv1.weight.data.shape == blockchoice[0].conv1.weight.data.shape:
                    layerchoice.conv1.weight.data = blockchoice[0].conv1.weight.data.clone().detach()
                if layerchoice.conv2.weight.data.shape == blockchoice[0].conv2.weight.data.shape:
                    layerchoice.conv2.weight.data = blockchoice[0].conv2.weight.data.clone().detach()
                if layerchoice.conv3.weight.data.shape == blockchoice[0].conv3.weight.data.shape:
                    layerchoice.conv3.weight.data = blockchoice[0].conv3.weight.data.clone().detach()
                if hasattr(layerchoice, 'downsample'):
                    if layerchoice.downsample is not None and blockchoice[0].downsample is not None:
                        if layerchoice.downsample[0].weight.data.shape == blockchoice[0].downsample[0].weight.data.shape:
                            layerchoice.downsample[0].weight.data = blockchoice[0].downsample[0].weight.data.clone().detach()
                        if layerchoice.downsample[1].weight.data.shape == blockchoice[0].downsample[1].weight.data.shape:
                            layerchoice.downsample[1].weight.data = blockchoice[0].downsample[1].weight.data.clone().detach()
                            layerchoice.downsample[1].bias.data = blockchoice[0].downsample[1].bias.data.clone().detach()
                            layerchoice.downsample[1].running_mean.data = blockchoice[0].downsample[1].running_mean.data.clone().detach()
                            layerchoice.downsample[1].running_var.data = blockchoice[0].downsample[1].running_var.data.clone().detach()
                            layerchoice.downsample[1].num_batches_tracked.data = blockchoice[0].downsample[1].num_batches_tracked.data.clone().detach()
                # blockchoice[0].bn1.eval()
                # blockchoice[0].bn2.eval()
                # blockchoice[0].bn3.eval()
                # if hasattr(blockchoice[0], 'downsample'):
                #     if blockchoice[0].downsample is not None:
                #         blockchoice[0].downsample[1].eval()
    else:
        for blockchoice in model.multiblocks:
            idx = 0
            for layerchoice in blockchoice[1:]:
                idx += 1
                if layerchoice.bn1.weight.data.shape == blockchoice[0].bn1.weight.data.shape:
                    layerchoice.bn1.weight.data = blockchoice[0].bn1.weight.data.clone().detach()
                    layerchoice.bn1.bias.data = blockchoice[0].bn1.bias.data.clone().detach()
                    layerchoice.bn1.running_mean.data = blockchoice[0].bn1.running_mean.data.clone().detach()
                    layerchoice.bn1.running_var.data = blockchoice[0].bn1.running_var.data.clone().detach()
                    layerchoice.bn1.num_batches_tracked.data = blockchoice[0].bn1.num_batches_tracked.data.clone().detach()
                    print(idx)
                if layerchoice.bn2.weight.data.shape == blockchoice[0].bn2.weight.data.shape:
                    layerchoice.bn2.weight.data = blockchoice[0].bn2.weight.data.clone().detach()
                    layerchoice.bn2.bias.data = blockchoice[0].bn2.bias.data.clone().detach()
                    layerchoice.bn2.running_mean.data = blockchoice[0].bn2.running_mean.data.clone().detach()
                    layerchoice.bn2.running_var.data = blockchoice[0].bn2.running_var.data.clone().detach()
                    layerchoice.bn2.num_batches_tracked.data = blockchoice[0].bn2.num_batches_tracked.data.clone().detach()
                    print(idx)
                if layerchoice.bn3.weight.data.shape == blockchoice[0].bn3.weight.data.shape:
                    layerchoice.bn3.weight.data = blockchoice[0].bn3.weight.data.clone().detach()
                    layerchoice.bn3.bias.data = blockchoice[0].bn3.bias.data.clone().detach()
                    layerchoice.bn3.running_mean.data = blockchoice[0].bn3.running_mean.data.clone().detach()
                    layerchoice.bn3.running_var.data = blockchoice[0].bn3.running_var.data.clone().detach()
                    layerchoice.bn3.num_batches_tracked.data = blockchoice[0].bn3.num_batches_tracked.data.clone().detach()
                    print(idx)
                if layerchoice.conv1.weight.data.shape == blockchoice[0].conv1.weight.data.shape:
                    layerchoice.conv1.weight.data = blockchoice[0].conv1.weight.data.clone().detach()
                    print(idx)
                if layerchoice.conv2.weight.data.shape == blockchoice[0].conv2.weight.data.shape:
                    layerchoice.conv2.weight.data = blockchoice[0].conv2.weight.data.clone().detach()
                    print(idx)
                if layerchoice.conv3.weight.data.shape == blockchoice[0].conv3.weight.data.shape:
                    layerchoice.conv3.weight.data = blockchoice[0].conv3.weight.data.clone().detach()
                    print(idx)
                if hasattr(layerchoice, 'downsample'):
                    if layerchoice.downsample is not None and blockchoice[0].downsample is not None:
                        if layerchoice.downsample[0].weight.data.shape == blockchoice[0].downsample[0].weight.data.shape:
                            layerchoice.downsample[0].weight.data = blockchoice[0].downsample[0].weight.data.clone().detach()
                        if layerchoice.downsample[1].weight.data.shape == blockchoice[0].downsample[1].weight.data.shape:
                            layerchoice.downsample[1].weight.data = blockchoice[0].downsample[1].weight.data.clone().detach()
                            layerchoice.downsample[1].bias.data = blockchoice[0].downsample[1].bias.data.clone().detach()
                            layerchoice.downsample[1].running_mean.data = blockchoice[0].downsample[1].running_mean.data.clone().detach()
                            layerchoice.downsample[1].running_var.data = blockchoice[0].downsample[1].running_var.data.clone().detach()
                            layerchoice.downsample[1].num_batches_tracked.data = blockchoice[0].downsample[1].num_batches_tracked.data.clone().detach()
                    print(idx)
