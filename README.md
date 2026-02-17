Oppgave 02. Create a neural network yourself, using the Keras API.

from tensorflow import keras // importer keras via tensorflow

from tensorflow.keras import layers

input_shape = (28, 28, 1) // formen på et input bilde28*28 1 kanal

model = keras.Sequential( 
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
) 

//lagene kommer i en fast rekkefølge

//først definere input laget. modellen skal få inn data med form (28, 28, 1)

// Conv finner mønster i bildet, 32 ulike mønstre, ser på 3*3 pixler av gangen, "relu" gjør modellen ikke-lineær og hjelper læring

// MaxPooling -> nedskalerer, gjør den raskere, reduserer overfitting, behold viktige trekk

// nytt conv lag med 64 filtre, lærer mer komplekse ting, flere filtre = avanserte mønstre

// Enda en pooling for å redusere dimensjonen, 

// Flatten gjør 2D feature til 1D vektor, nødvendig før en bruker Dense-lag

// Fult connected lag som lærer kombinasjoner med 128 neuroner

// Slår 50% av neuronene tilfelding under trening. hindrer overfitting, mer robust

// Output laget 10 fordi MNIST har 10 klasser

// softmax gjør output sannsynligheter som summerer til 1

Exercise for 04:
Decision boundary-plottene viser at modell 2 (ReLU + He-initialisering) presterer klart best. Den lærer en meningsfull separasjon av dataene, mens modell 1 forblir nær tilfeldig gjetting. Modell 2 kan gjøres mer stabil ved å justere learning rate, men modell 1 kan ikke oppnå tilsvarende ytelse kun ved å endre learning rate.

Gradient-plottene viser at gradientene ikke er like i alle lag. Tidlige lag har større og mer variable gradienter enn de senere lagene, noe som er forventet. I modell 2 er gradientene tydelige i alle lag, uten tegn til vanishing gradients. Dette forklarer hvorfor modellen lærer effektivt. Spikes i gradientene indikerer at learning rate er relativt høy, men ikke så høy at treningen kollapser.

Exercise for 05:
Kode:
batch_size = 128 
epochs = 10 

convnet.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

convnet.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=1
)

What do the conv2d_* layers do?
The Conv2D layers apply learned filters to the input image to detect local patterns such as edges, corners, and simple shapes. Deeper convolution layers combine these simple features into more complex patterns.

What do the max_pooling_* layers do?
The max pooling layers reduce the spatial resolution by keeping the strongest activations in each region. This makes the representation smaller, more robust to small shifts, and focuses on the most important features.

Does the information in the successive layers become more clear or less clear?
The information becomes less visually clear for humans but more abstract and meaningful for the network. Early layers look like the original image, while deeper layers represent high-level features.

Can you relate the pixels in the final layer to the different number predictions?
No, not directly. The final layer no longer represents individual pixels but high-level features that are combined by the dense layers to produce the final digit prediction.

07resnet: Har lastet opp en kopi med ferdig kode. 
