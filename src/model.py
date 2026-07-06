from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


class TrafficSignModel:

    def __init__(self,
                 input_shape=(30, 30, 3),
                 num_classes=43):

        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):

        model = Sequential()

        # Block 1
        model.add(
            Conv2D(
                32,
                (5, 5),
                activation="relu",
                input_shape=self.input_shape
            )
        )

        model.add(
            Conv2D(
                32,
                (5, 5),
                activation="relu"
            )
        )

        model.add(
            MaxPooling2D(
                pool_size=(2, 2)
            )
        )

        model.add(
            Dropout(0.25)
        )

        # Block 2
        model.add(
            Conv2D(
                64,
                (3, 3),
                activation="relu"
            )
        )

        model.add(
            Conv2D(
                64,
                (3, 3),
                activation="relu"
            )
        )

        model.add(
            MaxPooling2D(
                pool_size=(2, 2)
            )
        )

        model.add(
            Dropout(0.25)
        )

        # Fully Connected
        model.add(Flatten())

        model.add(
            Dense(
                256,
                activation="relu"
            )
        )

        model.add(
            Dropout(0.5)
        )

        model.add(
            Dense(
                self.num_classes,
                activation="softmax"
            )
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model