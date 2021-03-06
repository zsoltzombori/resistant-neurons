import numpy as np
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


EPOCHS = 5
BATCH_SIZE = 32
SNIP_WEIGHT = 0.0# 0001
PRIMARY_LOSS_FN = tf.keras.losses.sparse_categorical_crossentropy

LR = 0.0002
BETA_1=0.5
BETA_2=0.9

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(10, (3,3), padding='valid', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(50, (4,4), padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(100, (3,3), padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28,28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     # tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])


model.summary()

class SnipModel(tf.keras.Model):
    def __init__(self, model, snip_weight):
        super(SnipModel, self).__init__()
        self.model = model
        self.snip_weight = snip_weight

    def call(self, inputs):
        return self.model(inputs)

    def compile(self, optimizer, primary_loss_fn):
        super(SnipModel, self).compile()
        self.optimizer = optimizer
        self.primary_loss_fn = primary_loss_fn

    def primary_loss(self, x, y):
        y_pred = self.model(x, training=True)
        primary_loss = self.primary_loss_fn(y, y_pred)
        y_max = tf.cast(tf.argmax(y_pred, axis=-1), tf.uint8)
        hits = tf.cast(tf.equal(y,y_max), tf.int32)
        accuracy = tf.reduce_sum(hits) / tf.shape(hits)[0]
        return primary_loss, accuracy
        

    def snip_loss(self, x, y):
        variables= self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(variables)
            primary_loss, accuracy = self.primary_loss(x, y)
        grads = tape.gradient(primary_loss, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        snip_loss = tf.constant(0.0)
        for i in range(len(variables)):
            var = variables[i]
            grad = grads[i]
            snip_loss += tf.reduce_sum(tf.sqrt(1e-14 + tf.abs(grad*var)))
        loss = primary_loss - self.snip_weight * snip_loss
        return loss, accuracy

    def train_step(self, data):
        x, y = data
        variables= self.model.trainable_variables
        with tf.GradientTape() as tape:
            # tape.watch(variables)
            loss, accuracy = self.snip_loss(x, y)
        gradient = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradient, variables))
        return {"loss": loss, "accuracy":accuracy}

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2
)


snipModel = SnipModel(model, SNIP_WEIGHT)
snipModel.compile(optimizer, primary_loss_fn = PRIMARY_LOSS_FN)
        
snipModel.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)

# evaluate model
snipModel.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=2)
