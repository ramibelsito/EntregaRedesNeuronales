# Resumen del proceso de optimización del MLP

## 1. Punto de partida

El modelo inicial que entrené era un **MLP aplicado directamente sobre imágenes convertidas en vectores**. Es decir, cada imagen RGB se transformaba en un vector plano antes de entrar a la red.

Por ejemplo, para una imagen de tamaño `64x64`, el input efectivo del modelo era:

```text
64 × 64 × 3 = 12.288 features
```

y para una imagen de tamaño `96x96`, el input pasaba a ser:

```text
96 × 96 × 3 = 27.648 features
```

Esto implica una limitación importante desde el punto de vista teórico, porque el MLP no aprovecha la estructura espacial de la imagen. Al aplanar la imagen, el modelo pierde la noción de vecindad entre píxeles, por lo que no tiene un sesgo inductivo natural para detectar bordes, texturas locales, formas o patrones espaciales. Aun así, como el objetivo era mantener un modelo simple basado en MLP, enfoqué todo el trabajo posterior en optimizar esa arquitectura sin introducir capas convolucionales.

El output inicial mostraba dos problemas principales. Por un lado, aparecían warnings del tipo `UndefinedMetricWarning`, lo que indicaba que el modelo no estaba prediciendo algunas clases. Esto suele ocurrir cuando el clasificador colapsa parcialmente hacia las clases más fáciles o más frecuentes, dejando de asignar predicciones a otras etiquetas. Por otro lado, MLflow generaba mucho ruido en el output porque se estaban logueando modelos repetidamente y sin una configuración adecuada de `signature` o `input_example`.

---

## 2. Problemas metodológicos iniciales

En la primera versión del entrenamiento y de la búsqueda de hiperparámetros, identifiqué varias debilidades metodológicas.

Primero, la búsqueda de hiperparámetros estaba basada en combinaciones aleatorias poco controladas. Esto hacía que los resultados fueran difíciles de reproducir, porque no siempre era claro si una mejora se debía realmente a los hiperparámetros o simplemente a una inicialización favorable.

Segundo, la métrica principal era `val_accuracy`. Para este problema, esa métrica no era ideal, porque se trata de una clasificación multiclase con clases que tienen distintos niveles de dificultad. La accuracy puede ocultar problemas importantes: por ejemplo, el modelo puede clasificar bien clases relativamente fáciles como `Benign keratosis` o `Vascular lesion`, pero fallar en clases más difíciles como `Melanoma`, `Squamous cell carcinoma`, `Actinic keratosis` o `Dermatofibroma`.

Tercero, el entrenamiento generaba reportes completos y matrices de confusión durante muchas etapas del proceso. Esto ensuciaba el output y agregaba overhead innecesario. A su vez, el logueo de MLflow interfería visualmente con la corrida y, en algunos casos, hacía parecer que el notebook se había trabado, aunque el entrenamiento ya hubiera terminado.

---

## 3. Mejoras implementadas en la búsqueda de hiperparámetros

La primera mejora importante fue cambiar la métrica principal de selección desde `val_accuracy` hacia **`val_macro_f1`**.

Esto fue relevante porque `macro_f1` calcula el F1-score de cada clase por separado y luego promedia todas las clases con el mismo peso. En un problema con 9 clases, esta métrica penaliza fuertemente que el modelo ignore clases difíciles. Por eso, me pareció una métrica más adecuada para elegir hiperparámetros.

También agregué **balanced accuracy**, que mide el recall promedio por clase. En los mejores runs, `val_macro_f1` y `val_balanced_accuracy` quedaron bastante cercanos, lo cual fue una señal razonable de que el modelo no estaba obteniendo buenos resultados únicamente por predecir bien clases mayoritarias o fáciles.

La segunda mejora fue reemplazar la función de pérdida original por una versión ponderada y suavizada:

```python
CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
```

Con esto ataqué dos problemas diferentes. Los `class_weights` hacen que los errores en clases menos representadas o más difíciles tengan mayor peso durante el entrenamiento. Por su parte, `label_smoothing` evita que el modelo se vuelva excesivamente confiado en una sola clase, lo que puede ayudar a mejorar la generalización en datasets chicos.

La tercera mejora fue cambiar el optimizador a **AdamW** con `weight_decay`. AdamW es una variante de Adam que desacopla el decaimiento de pesos del update adaptativo. Esto suele ser más estable para redes densas con muchos parámetros. En mi caso era especialmente importante, porque incluso el MLP “simple” tenía millones de parámetros.

También agregué un scheduler:

```python
ReduceLROnPlateau(mode="max", factor=0.5, patience=4)
```

La lógica fue reducir el learning rate cuando `val_macro_f1` dejaba de mejorar. Esto permite que el modelo siga refinando la solución con pasos más chicos cuando la mejora se estanca.

La cuarta mejora fue modificar la arquitectura interna del MLP. Pasé a usar una estructura del tipo:

```text
Linear → LayerNorm → GELU → Dropout
Linear → LayerNorm → GELU → Dropout
Linear → output
```

Esto sigue siendo un MLP puro, sin capas convolucionales. La incorporación de `LayerNorm` ayuda a estabilizar las activaciones internas, especialmente cuando el input es un vector grande de píxeles. `GELU` ofrece una activación más suave que `ReLU`, y `Dropout` ayuda a reducir el overfitting.

La quinta mejora fue agregar **gradient clipping**:

```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Esto permite limitar la norma de los gradientes y evitar updates demasiado grandes. No necesariamente mejora siempre la métrica final, pero hace que el entrenamiento sea más estable.

La sexta mejora fue aplicar **early stopping usando `val_macro_f1`** como criterio principal. Esto significa que el entrenamiento se detenía cuando la métrica relevante dejaba de mejorar durante cierta cantidad de épocas. Por eso, las barras de progreso no llegaban al 100%: el modelo no corría necesariamente las 200 épocas máximas, sino que se detenía antes si ya no mejoraba.

La séptima mejora fue agregar una **evaluación de train sin augmentation**. Esto fue importante porque medir train usando augmentations activas puede distorsionar el rendimiento real del modelo sobre el conjunto de entrenamiento. Al crear un `train_eval_loader` separado, pude comparar de forma más justa:

```text
train sin augmentation vs validation sin augmentation
```

Esto me permitió estimar mejor el gap de generalización.

La octava mejora fue incorporar repeticiones por **diferentes seeds** y una **búsqueda local alrededor del trial 53**. Esto fue importante para distinguir una mejora real de un resultado alto obtenido solamente por azar en una corrida específica.

---

## 4. Resultado de la primera búsqueda optimizada

La primera búsqueda optimizada encontró como mejor configuración el **trial 53**, con aproximadamente los siguientes hiperparámetros:

```text
input_size = 64
batch_size = 64
lr = 1e-4
weight_decay = 1e-3
hidden_sizes = (1024, 512)
dropout = 0.35
HFlip = 0.5
VFlip = 0.5
RBContrast = 0.3
label_smoothing = 0.05
grad_clip_norm = 1.0
```

Ese modelo tenía alrededor de:

```text
13.116.425 parámetros
```

y logró los siguientes resultados:

```text
best_val_macro_f1 ≈ 0.6403
best_val_accuracy ≈ 0.6303
best_train_macro_f1 ≈ 0.7412
```

El gap aproximado entre train y validation fue:

```text
0.7412 - 0.6403 ≈ 0.101
```

Esto indicaba cierto nivel de overfitting, pero no parecía completamente descontrolado. Para un MLP aplicado sobre píxeles crudos, con 9 clases y un dataset chico, consideré que era un resultado razonable.

---

## 5. Aplicación del trial 53 al modelo simple

Después apliqué esos hiperparámetros al modelo simple. Este paso era importante porque una búsqueda de hiperparámetros puede tener condiciones ligeramente distintas respecto de un notebook final: seed, orden de batches, transforms, estado de la GPU, logging, etc.

Al correr el modelo simple con los hiperparámetros del trial 53, los resultados fueron parecidos, pero no idénticos. En algunos runs obtuve valores cercanos a:

```text
best_val_macro_f1 ≈ 0.57 - 0.59
accuracy ≈ 0.58
```

Por ejemplo, en una de las corridas el modelo terminó por early stopping cerca de la epoch 42, con mejor epoch alrededor de la 29 y un `best_val_macro_f1` cercano a `0.5916`.

Esto mostró algo importante: el trial 53 había dado un buen resultado en la búsqueda, pero al llevarlo al modelo simple la performance no siempre se reproducía exactamente. Esa diferencia podía venir de la variabilidad propia del entrenamiento, de la inicialización de pesos, del orden de los batches y de las augmentations aleatorias.

---

## 6. Búsqueda local v3 alrededor del trial 53

Después realicé una búsqueda más fina alrededor de la zona buena encontrada por el trial 53. La mejor configuración encontrada en esa búsqueda local fue:

```text
run_label = local_037_seed_2063
input_size = 96
batch_size = 64
lr = 5e-5
weight_decay = 1e-3
hidden_sizes = (1536, 768)
dropout = 0.35
HFlip = 0.5
VFlip = 0.5
RBContrast = 0.3
seed = 2063
```

El resultado de esa búsqueda fue:

```text
Final val macro F1 = 0.6362
Final val accuracy = 0.6242
Train eval macro F1 = 0.7950
Gap macro F1 = 0.1588
```

Teóricamente, este modelo era bastante más grande que el del trial 53. Con `input_size = 96`, el input plano era:

```text
96 × 96 × 3 = 27.648 features
```

y la primera capa con 1536 neuronas tenía aproximadamente:

```text
27.648 × 1.536 ≈ 42.5 millones de pesos
```

Es decir, el modelo v3 tenía mucha más capacidad que el modelo del trial 53 original. Sin embargo, esa mayor capacidad no se tradujo en una mejora clara de validación.

Comparando ambos resultados:

```text
Trial 53 original:    val macro F1 ≈ 0.6403
V3 local_037:         val macro F1 ≈ 0.6362
```

La diferencia fue mínima y, de hecho, favoreció levemente al trial 53 original. Además, el gap de generalización del modelo v3 fue más alto:

```text
Train eval macro F1 ≈ 0.7950
Val macro F1 ≈ 0.6362
Gap ≈ 0.1588
```

Esto sugiere que el modelo v3 tenía mayor tendencia al overfitting.

---

## 7. Aplicación de los hiperparámetros v3 al modelo simple

Luego apliqué los hiperparámetros v3 al modelo simple. En el primer run con esta configuración obtuve aproximadamente:

```text
Early stopping en epoch 35/200
Best epoch = 22
Best val macro F1 = 0.5869
Accuracy = 0.58
Macro avg F1 = 0.59
```

El tiempo visible de entrenamiento fue aproximadamente:

```text
34/200 épocas en 01:38
2.89 s/epoch
```

Este resultado fue claramente inferior al resultado obtenido durante la búsqueda v3.

Después volví a correr el mismo modelo y obtuve un resultado algo mejor:

```text
Early stopping en epoch 53/200
Best epoch = 40
Best val macro F1 = 0.6041
Best val accuracy = 0.60
```

El tiempo visible fue:

```text
52/200 épocas en 02:26
2.81 s/epoch
```

Este segundo rerun fue mejor que el primero, pero siguió por debajo tanto del trial 53 original como del mejor run de la búsqueda v3.

Finalmente, el modelo se guardó como:

```text
mlp_model_trial53_best.pth
```

con aproximadamente:

```text
Best val macro F1 = 0.6041
Best val accuracy = 0.60
```

---

## 8. Lectura de los reportes por clase

Los reportes de clasificación mostraron un patrón bastante consistente. Las clases donde el MLP funcionó mejor fueron:

```text
Benign keratosis
Vascular lesion
Atopic Dermatitis
Tinea Ringworm Candidiasis
Melanocytic nevus
```

En el último rerun, por ejemplo, obtuve aproximadamente:

```text
Benign keratosis: F1 = 0.88
Vascular lesion: F1 = 0.82
Atopic Dermatitis: F1 = 0.73
Melanocytic nevus: F1 = 0.73
Tinea Ringworm Candidiasis: F1 = 0.69
```

En cambio, las clases más difíciles siguieron siendo:

```text
Dermatofibroma
Squamous cell carcinoma
Melanoma
Actinic keratosis
```

En el último rerun, estas clases obtuvieron aproximadamente:

```text
Dermatofibroma: F1 = 0.33
Squamous cell carcinoma: F1 = 0.36
Melanoma: F1 = 0.45
Actinic keratosis: F1 = 0.45
```

Esto sugiere que el cuello de botella no era solamente la elección de hiperparámetros. El MLP podía separar razonablemente bien algunas clases visualmente más distinguibles, pero seguía teniendo dificultades con clases que probablemente requieren información espacial, textura local o patrones visuales finos.

---


## 9. Tiempos de corrida

En el primer rerun con la configuración v3, la barra de progreso mostró aproximadamente:

```text
34/200 épocas en 01:38
2.89 s/epoch
```

y el modelo terminó por early stopping en la epoch 35.

En el segundo rerun con la configuración v3, la barra mostró aproximadamente:

```text
52/200 épocas en 02:26
2.81 s/epoch
```

y el modelo terminó por early stopping en la epoch 53.

Para la búsqueda completa de hiperparámetros no tengo un tiempo total limpio en los outputs. Sí pude observar que las búsquedas largas generaban logs durante períodos extendidos y que MLflow imprimía warnings repetidos, pero no hay un inicio y fin inequívocos de cada búsqueda completa como para reportar un tiempo exacto sin estimarlo.

Una estimación razonable para la configuración v3 es que un run de 50 épocas tarda aproximadamente:

```text
50 × 2.8 s ≈ 140 s ≈ 2 min 20 s
```

Esto coincide con el segundo rerun. Si una búsqueda local ejecuta muchos trials, el tiempo total escala casi linealmente con la cantidad de trials efectivos y con la cantidad de épocas que cada trial sobrevive antes de early stopping.

---

## 10. Variabilidad entre corridas

Un punto importante es que tiene sentido que el mismo modelo, con los mismos hiperparámetros, dé resultados distintos al correrlo varias veces. El modelo no está “mejorando” por acumulación entre corridas; cada corrida es una nueva realización del proceso de entrenamiento.

Las principales fuentes de variabilidad son:

```text
inicialización aleatoria de pesos
orden aleatorio de batches
augmentations aleatorias
early stopping en distintos puntos
operaciones no perfectamente determinísticas en GPU
```

En mis resultados esto se vio claramente:

```text
v3 simple run 1: best val macro F1 = 0.5869
v3 simple run 2: best val macro F1 = 0.6041
v3 búsqueda:      best val macro F1 = 0.6362
```

Esto no implica una mejora progresiva. Si volviera a correr el mismo modelo, podría dar mejor o peor.

Además, el validation set tenía solamente 165 imágenes. En términos de accuracy, cada imagen representa aproximadamente:

```text
1 / 165 ≈ 0.0061
```

Es decir, 5 imágenes más o menos correctamente clasificadas pueden mover la accuracy alrededor de 3 puntos. En macro F1, el efecto puede ser incluso mayor, porque algunas clases tienen pocos ejemplos.

---

## 11. Conclusión técnica

A partir de todo el proceso, concluyo que las mejoras implementadas fueron metodológicamente correctas. Cambiar la métrica principal a `macro_f1`, usar class weights, label smoothing, AdamW, scheduler, LayerNorm, GELU, Dropout, gradient clipping, early stopping por F1, evaluación sin augmentation y búsqueda por seeds permitió construir un entrenamiento mucho más sólido que el inicial.

La mejora real apareció principalmente al pasar del setup inicial al primer modelo optimizado. El trial 53 fue un punto fuerte, porque logró alrededor de `0.64` de macro F1 en validación con un modelo relativamente más chico y más estable.

En cambio, la búsqueda v3 no demostró una mejora robusta. Encontró un modelo más grande, con `input_size = 96` y `hidden_sizes = (1536, 768)`, pero su mejor resultado no superó claramente al trial 53. Además, al aplicar esos hiperparámetros al modelo simple, el rendimiento quedó entre `0.5869` y `0.6041`, y el modelo mostró un gap de generalización mayor.

Mi interpretación final es que el MLP optimizado probablemente está cerca de su techo práctico con este dataset y esta representación de entrada. El rango esperable de performance parece estar aproximadamente entre:

```text
0.58 y 0.64 de macro F1 en validación
```

El trial 53 original parece ser un mejor candidato final que la configuración v3, porque es más chico, más rápido y no rindió peor. La búsqueda v3 exploró mayor capacidad, pero esa capacidad adicional pareció traducirse más en overfitting que en una mejora robusta de validación.

Para cerrar el experimento de forma más sólida, no elegiría el mejor pico individual. Compararía el trial 53 y la configuración v3 usando varias seeds y reportaría:

```text
mean_val_macro_f1 ± std_val_macro_f1
mean_val_accuracy ± std_val_accuracy
mean_generalization_gap
tiempo promedio por run
```

De esa forma podría justificar técnicamente cuál configuración conviene usar como modelo simple final.

