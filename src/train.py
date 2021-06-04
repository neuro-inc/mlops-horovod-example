from pathlib import Path

import tensorflow as tf
import horovod.tensorflow.keras as hvd

hvd.init()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def main() -> None:
    ### to expose
    LR = 0.001
    epochs = 5
    checkpoint_prefix = Path(".")
    ###

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0

    model = get_model(learning_rate=LR)

    def on_state_reset():
        tf.keras.backend.set_value(model.optimizer.lr, LR * hvd.size())

    state = hvd.elastic.KerasState(model, batch=100, epoch=0)
    state.register_reset_callbacks([on_state_reset])

    callbacks = [
        hvd.elastic.CommitStateCallback(state),
        hvd.elastic.UpdateBatchStateCallback(state),
        hvd.elastic.UpdateEpochStateCallback(state),
    ]

    if hvd.rank() == 0:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_prefix) + "/checkpoint-{epoch}.h5"
            )
        )

    @hvd.elastic.run
    def train(state):
        model.fit(
            x_train,
            y_train,
            steps_per_epoch=500 // hvd.size(),
            callbacks=callbacks,
            epochs=epochs - state.epoch,
            verbose=1 if hvd.rank() == 0 else 0,
        )

    train(state)


def get_model(learning_rate: float) -> tf.keras.Model:

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
