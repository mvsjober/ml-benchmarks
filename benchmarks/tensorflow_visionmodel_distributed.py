# TensorFlow 2.21 port of pytorch_visionmodel_ddp.py
#
# Vision-model throughput benchmark using synthetic data. Multi-GPU is handled
# with tf.distribute.MirroredStrategy (single process, all visible GPUs), so no
# torchrun / per-GPU process launching and no GPU-CPU core binding is needed.
# Models are loaded from tf.keras.applications with random weights.

# datetime is used for the wall-clock timing that backs the images/sec metric.
from datetime import datetime
# argparse exposes the same command-line knobs as the original PyTorch script.
import argparse

import tensorflow as tf

# Map torchvision-style model names to tf.keras.applications constructors.
# This lets "--model resnet50" keep working the same way it did in PyTorch,
# while resolving to the equivalent Keras model class.
MODELS = {
    'resnet50': tf.keras.applications.ResNet50,
    'resnet101': tf.keras.applications.ResNet101,
    'resnet152': tf.keras.applications.ResNet152,
    'inception_v3': tf.keras.applications.InceptionV3,
    'mobilenet_v2': tf.keras.applications.MobileNetV2,
    'vgg16': tf.keras.applications.VGG16,
    'densenet121': tf.keras.applications.DenseNet121,
    'efficientnet_b0': tf.keras.applications.EfficientNetB0,
}

# Keras applications expect channels-last (H, W, C) inputs, unlike PyTorch's
# channels-first (C, H, W). 224x224x3 matches the original benchmark.
IMAGE_SHAPE = (224, 224, 3)
# Number of output classes, matching ImageNet (and the original script).
NUM_CLASSES = 1000
# Number of samples in one synthetic "epoch"; sized like ImageNet so that the
# reported epoch length matches the PyTorch benchmark when --steps is not given.
DATASET_SIZE = 1281184


def build_model(name):
    """Construct a randomly-initialised Keras model by torchvision-style name."""
    # Fail loudly on an unknown name rather than silently picking a default.
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: "
                         f"{', '.join(sorted(MODELS))}")
    # weights=None  -> random initialisation (matches the original PyTorch
    #                  script, which trained from scratch on synthetic data).
    # classes=1000  -> ImageNet-sized classification head.
    # input_shape   -> fix the input size so the head dimensions are defined.
    # classifier_activation=None -> output raw logits, required for the
    #                  from_logits=True cross-entropy loss below.
    return MODELS[name](weights=None, classes=NUM_CLASSES,
                        input_shape=IMAGE_SHAPE, classifier_activation=None)


def make_synthetic_dataset(global_batch_size, num_parallel):
    """A tf.data pipeline that yields a fresh random batch each step.

    A new random image/label is generated per element (not a single fixed batch
    fed repeatedly), matching the modified PyTorch benchmark behaviour. Some
    systems can over-optimise a reused fixed batch, giving unrealistic numbers.
    """
    # Per-element generator. The element index is ignored; we only use it to
    # drive the right number of iterations.
    def gen(_):
        # Random normal image, same distribution as torch.randn in the original.
        img = tf.random.normal(IMAGE_SHAPE)
        # Random integer class label in [0, NUM_CLASSES).
        label = tf.random.uniform((), minval=0, maxval=NUM_CLASSES,
                                  dtype=tf.int32)
        return img, label

    # Start from a counter of DATASET_SIZE elements (the epoch length).
    ds = tf.data.Dataset.range(DATASET_SIZE)
    # Generate a fresh random sample for each element, in parallel.
    ds = ds.map(gen, num_parallel_calls=num_parallel)
    # Group into global batches; drop_remainder keeps every step the same size,
    # which matters for the per-step images/sec accounting.
    ds = ds.batch(global_batch_size, drop_remainder=True)
    # Overlap data production with model execution.
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def train(args):
    # Print the version so it is captured in benchmark logs (as the PyTorch
    # script printed torch.__version__).
    print('Using TensorFlow version:', tf.__version__)

    # Report how many GPUs TensorFlow can actually see on this node.
    gpus = tf.config.list_physical_devices('GPU')
    print(f'Visible GPUs: {len(gpus)}')

    # MirroredStrategy = synchronous all-reduce data parallelism across all
    # visible GPUs in a single process. This is the single-node equivalent of
    # PyTorch DDP.
    strategy = tf.distribute.MirroredStrategy()
    # Number of replicas (one per GPU); analogous to DDP's world_size.
    world_size = strategy.num_replicas_in_sync
    print(f'Number of replicas in sync: {world_size}')

    # Enable mixed precision if requested. Must be set before the model is built
    # so the layers pick up the float16 compute policy.
    if args.fp16:
        print('Using fp16 (mixed_float16 mixed precision)')
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # --batchsize is per-replica (per-GPU), as in the PyTorch DDP version, so
    # the global batch grows with the number of GPUs.
    global_batch_size = args.batchsize * world_size

    # <=0 means "let tf.data choose" (AUTOTUNE); otherwise honour the requested
    # number of parallel map calls. This is the rough analogue of DataLoader's
    # num_workers.
    num_parallel = tf.data.AUTOTUNE if args.workers <= 0 else args.workers
    # Build the synthetic input pipeline...
    dataset = make_synthetic_dataset(global_batch_size, num_parallel)
    # ...and shard/distribute it across the replicas.
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # Everything that owns variables (model, optimizer) must be created inside
    # the strategy scope so the variables are mirrored across replicas.
    with strategy.scope():
        print(f'Using {args.model} model')
        model = build_model(args.model)

        # Plain SGD with the same 1e-4 learning rate as the original.
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
        # Under mixed precision, wrap the optimizer so it scales the loss to
        # avoid float16 gradient underflow.
        if args.fp16:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        # Cross-entropy on integer labels. reduction=NONE returns the per-example
        # loss vector so we can average it ourselves over the *global* batch.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

    # Per-replica training step. Runs the forward/backward pass for the slice of
    # the global batch that this replica received.
    def step_fn(inputs):
        images, labels = inputs
        # Record operations for automatic differentiation.
        with tf.GradientTape() as tape:
            # Forward pass; training=True activates dropout/batchnorm updates.
            logits = model(images, training=True)
            # Cast logits back to float32 for a numerically stable loss under
            # mixed precision (no-op when fp16 is disabled).
            logits = tf.cast(logits, tf.float32)
            # Per-example losses for this replica's slice.
            per_example_loss = loss_fn(labels, logits)
            # Average over the GLOBAL batch size so that summing the replicas'
            # gradients gives the correct mean gradient (standard tf.distribute
            # pattern).
            loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=global_batch_size)
            # Scale the loss up before backprop under mixed precision.
            if args.fp16:
                scaled_loss = optimizer.get_scaled_loss(loss)

        # Compute gradients, unscaling them again under mixed precision.
        if args.fp16:
            scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients; the all-reduce across replicas happens here.
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Wrap the per-replica step so it runs on every replica and the per-replica
    # losses are summed into a single scalar for logging. tf.function compiles
    # it into a graph for speed.
    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_loss = strategy.run(step_fn, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss,
                               axis=None)

    # Total steps: either the user-specified cap, or one full epoch's worth of
    # global batches.
    total_steps = args.steps if args.steps is not None \
        else DATASET_SIZE // global_batch_size

    # --- Timing state, mirroring the original script ---
    # For the periodic (per print-block) throughput print:
    last_start = datetime.now()
    last_images = 0

    # For the final, warmup-excluded average:
    avg_images = 0
    avg_start = None
    avg_stop = None
    # Counts steps across all epochs.
    steps_counter = 0

    # Overall wall-clock start (used only for the "Training completed in" line).
    real_start = datetime.now()

    # If there is no warmup, start the average clock immediately.
    if args.warmup_steps == 0:
        avg_start = datetime.now()

    for epoch in range(args.epochs):
        for inputs in dist_dataset:
            # One optimization step across all replicas.
            loss = distributed_train_step(inputs)

            # Images processed this step = the whole global batch.
            li = global_batch_size
            last_images += li

            steps_counter += 1

            # Once warmup is finished, start the averaging clock.
            if steps_counter == args.warmup_steps:
                avg_start = datetime.now()
            # After warmup, accumulate images into the average...
            elif steps_counter > args.warmup_steps:
                # ...but exclude the final batch, whose extra teardown delays
                # would skew the average downward.
                if steps_counter < total_steps - 1:
                    avg_images += li
                    avg_stop = datetime.now()

            # Periodically print the throughput over the last print-block.
            if steps_counter % args.print_steps == 0:
                now = datetime.now()
                last_secs = (now - last_start).total_seconds()

                # global_batch_size already counts all replicas, so no extra
                # multiply by world_size is needed here.
                print(f'Epoch [{epoch+1}/{args.epochs}], '
                      f'Step [{steps_counter}/{total_steps}], '
                      f'Loss: {float(loss):.4f}, '
                      f'Images/sec: {last_images/last_secs:.2f} '
                      f'(last {args.print_steps} steps)', flush=True)

                # Reset the print-block counters.
                last_start = now
                last_images = 0

            # Stop early once we hit the requested step cap.
            if args.steps is not None and steps_counter >= args.steps:
                break
        # Propagate the early stop out of the epoch loop too.
        if args.steps is not None and steps_counter >= args.steps:
            break

    # Total elapsed wall-clock time.
    dur = datetime.now() - real_start

    # If we never got past warmup there is nothing meaningful to report.
    if avg_start is None or avg_stop is None:
        print("WARNING: stopped before warmup steps done, not printing stats.")
    else:
        # Final average throughput over the timed (post-warmup) window.
        avg_dur = (avg_stop - avg_start).total_seconds()
        print(f"Training completed in: {dur}")
        print(f"Images/sec: {avg_images/avg_dur:.2f} "
              f"(average, skipping {args.warmup_steps} warmup steps)")


def main():
    # Command-line interface, kept compatible with the surviving subset of the
    # original PyTorch script's arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model to benchmark')
    parser.add_argument('-b', '--batchsize', type=int, default=32,
                        help='Per-GPU (per-replica) batch size')
    parser.add_argument('-j', '--workers', type=int, default=10,
                        help='tf.data parallel map calls (<=0 means AUTOTUNE)')
    parser.add_argument('--steps', type=int, required=False,
                        help='Maximum number of training steps')
    parser.add_argument('--print-steps', type=int, default=100,
                        help='Print throughput every N steps')
    parser.add_argument('--warmup-steps', type=int, default=10,
                        help='Number of initial steps to ignore in average')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='enable mixed precision')

    args = parser.parse_args()
    train(args)


# Standard entry-point guard.
if __name__ == '__main__':
    main()
