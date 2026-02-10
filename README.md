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
