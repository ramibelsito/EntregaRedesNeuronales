# Scripts Clean repeated images 
 
Para correrlo: ./clean_duplicate_images.sh ./direccion_de_carpetas_train_y_val

# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP

## 1. Dataset y Preprocesamiento

- ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?

Respuesta: Es necesario estandarizar el tamaño porque la primera etapa de la red tiene una cantidad fija de entradas.

- ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como `torchvision.transforms`?

Respuesta: Albumentations es una librería más rápida, que ofrece una programación más directa sin tener que modificar por separado las máscaras y las imágenes en sí. Además Albumentations tiene transformaciones mejores en calidad y complejidad. `torchvision.transforms` utiliza principalmente transformaciones más simples.

- ¿Qué hace `A.Normalize()`? ¿Por qué es importante antes de entrenar una red?

Respuesta: Lo que hace es permitir usar la distribución normal estándar modificando el valor original, restándole la media y dividiendo por el desvío de la muestra. Es necesario porque las diferencias de escala hacen que algunas variables tengan velocidades distintas de entrenamiento, demorando entonces el entrenamiento de la red total.

- ¿Por qué convertimos las imágenes a `ToTensorV2()` al final de la pipeline?

Respuesta: Es importante convertirlo a `torch.Tensor` porque son los tipos de datos que reciben las redes que se entrenarán a continuación. Se implementa luego de las transformaciones y la normalización porque las funciones utilizadas trabajan sobre la imagen y la máscara, y no pueden ser aplicadas a tensores. Este tipo de dato permite trabajar con la GPU para optimizar el procesamiento de la red.

## 2. Arquitectura del Modelo

- ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?

Respuesta: Utilizamos MLP porque es rápido de entrenar y requiere menos cómputo y memoria, pudiendo correr el modelo incluso en laptops. Además funciona bien para problemas de regresión y clasificación si no se trata de un patrón demasiado complejo. Es útil este modelo para aprender el funcionamiento de las redes neuronales.

La complejidad de las redes MLP crece rápidamente cuando se aumenta el tamaño de la muestra. Además pierde la referencia espacial de las imágenes al tener que procesar los datos en una matriz 1D.

- ¿Qué hace la capa `Flatten()` al principio de la red?

Respuesta: Pasa los datos de la imagen a un tensor 1D para poder ser procesado en la red en uso. Si usásemos una red CNN podríamos usar la referencia espacial de las imágenes trabajando en matrices 2D o 3D.

- ¿Qué función de activación se usó? ¿Por qué no usamos `Sigmoid` o `Tanh`?

Respuesta: Se utilizó una función ReLU, Unidad Lineal Rectificada. La principal razón es porque genera 0 para valores negativos, haciendo que el entrenamiento de la red sea más eficiente y rápido. Evita el desvanecimiento de gradiente, porque durante el backpropagation los resultados de los gradientes de las primeras capas no se multiplican por valores menores a 1.

- ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?

Respuesta: `input_size`, que deberá ser el alto por el ancho de la imagen por 3 debido a los colores RGB.

## 3. Entrenamiento y Optimización

- ¿Qué hace `optimizer.zero_grad()`?

Respuesta: Resetea los valores de gradientes guardados entre iteraciones de entrenamiento.

- ¿Por qué usamos `CrossEntropyLoss()` en este caso?

Respuesta: Porque es la función de loss más simple y sirve para calcular la incerteza del resultado en cada iteración. En esta función de pérdida, el gradiente es muy alto cuando la red es muy mala, entonces acelera el entrenamiento.

- ¿Cómo afecta la elección del tamaño de batch (`batch_size`) al entrenamiento?

Respuesta: Un `batch_size` grande hace que el entrenamiento de la red sea más rápido, con gradientes más estables y precisos, pero consume más cantidad de memoria. Por otro lado, `batch_size` chicos ayudan a evitar el overfitting, ya que es más difícil que se sobreajuste porque evita que el modelo converja a mínimos locales.

- ¿Qué pasaría si no usamos `model.eval()` durante la validación?

Respuesta: Si no lo usásemos, seguiría entrenando y actualizando los gradientes con datos de validación, por lo que no serían válidos a largo plazo los resultados de las validaciones.

## 4. Validación y Evaluación

- ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?

Respuesta: Significa que el modelo tiene overfitting, ya que aprende bien los datos de entrenamiento pero no generaliza igual de bien con datos nuevos.

- ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?

Respuesta: Podrían usarse `precision`, `recall` y `f1-score`, especialmente si las clases están desbalanceadas o si algunos errores son más importantes que otros.

- ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?

Respuesta: Permite ver en qué clases se equivoca el modelo y entre cuáles se confunde. La accuracy solo muestra el porcentaje total de aciertos.

- En el reporte de clasificación, ¿qué representan `precision`, `recall` y `f1-score`?

Respuesta: `precision` mide cuántas predicciones de una clase fueron correctas. `recall` mide cuántos casos reales de esa clase fueron encontrados. `f1-score` combina ambas métricas.

## 5. TensorBoard y Logging

- ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?

Respuesta: Permite visualizar la evolución de métricas como loss y accuracy durante el entrenamiento. Esto ayuda a detectar overfitting, inestabilidad o problemas de convergencia.

- ¿Qué diferencias hay entre loguear `add_scalar`, `add_image` y `add_text`?

Respuesta: `add_scalar` guarda valores numéricos, como loss o accuracy. `add_image` guarda imágenes o gráficos. `add_text` guarda información escrita, como reportes o comentarios.

- ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?

Respuesta: Es útil para verificar que las imágenes y transformaciones se estén aplicando correctamente, y para detectar errores en los datos o etiquetas.

- ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?

Respuesta: Guardando cada experimento en una carpeta distinta dentro de `runs`. Así se pueden comparar las curvas de loss y accuracy entre modelos.

## 6. Generalización y Transferencia

- ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?

Respuesta: Habría que cambiar la última capa del modelo para que tenga 100 salidas, una por cada clase. También habría que ajustar las etiquetas del dataset.

- ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?

Respuesta: Porque una CNN conserva la información espacial de la imagen, mientras que una MLP aplana la imagen y pierde la relación entre píxeles cercanos.

- ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?

Respuesta: Podría aparecer overfitting, ya que el modelo memoriza las imágenes de entrenamiento en vez de aprender patrones generales.

- ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?

Respuesta: Habría que trabajar con un solo canal en vez de tres. Por eso el `input_size` pasaría a ser alto por ancho por 1.

## 7. Regularización

### Preguntas teóricas

- ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?

Respuesta: La regularización es un conjunto de técnicas que se usan para evitar que la red aprenda demasiado de memoria los datos de entrenamiento. El objetivo es reducir el overfitting, haciendo que el modelo aprenda patrones más generales y pueda funcionar mejor con datos nuevos.

- ¿Cuál es la diferencia entre `Dropout` y regularización `L2` (weight decay)?

Respuesta: `Dropout` apaga aleatoriamente algunas neuronas durante el entrenamiento, obligando a la red a no depender siempre de las mismas conexiones. En cambio, la regularización `L2` o `weight_decay` penaliza los pesos muy grandes, haciendo que el modelo tenga parámetros más chicos y menos extremos. Ambos ayudan a reducir el overfitting, pero actúan de formas distintas.

- ¿Qué es `BatchNorm` y cómo ayuda a estabilizar el entrenamiento?

Respuesta: `BatchNorm` normaliza las activaciones internas de la red en cada batch. Esto hace que las entradas de cada capa tengan una distribución más estable durante el entrenamiento. Como consecuencia, los gradientes suelen ser más estables y el modelo puede entrenar de forma más controlada.

- ¿Cómo se relaciona `BatchNorm` con la velocidad de convergencia?

Respuesta: `BatchNorm` puede acelerar la convergencia porque reduce los cambios bruscos en la distribución de los datos que recibe cada capa. Al mantener las activaciones más estables, el optimizador puede avanzar mejor y muchas veces permite usar learning rates un poco más altos sin que el entrenamiento se vuelva inestable.

- ¿Puede `BatchNorm` actuar como regularizador? ¿Por qué?

Respuesta: Sí, puede actuar como regularizador aunque no sea su función principal. Como calcula estadísticas por batch, introduce una pequeña variación en las activaciones durante el entrenamiento. Esa variación puede hacer que el modelo dependa menos de valores exactos y generalice mejor.

- ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?

Respuesta: En TensorBoard se podría ver que la loss de entrenamiento sigue bajando mientras la loss de validación se estanca o empieza a subir. También podría verse que la accuracy de entrenamiento aumenta mucho más que la accuracy de validación. Esa separación entre las curvas es una señal clara de overfitting.

- ¿Cómo ayuda la regularización a mejorar la generalización del modelo?

Respuesta: La regularización ayuda porque limita la capacidad del modelo de memorizar los datos de entrenamiento. Al hacer que los pesos sean menos extremos, apagar neuronas o aumentar la variedad de los datos con augmentation, el modelo aprende patrones más robustos y puede funcionar mejor con imágenes que no vio antes.

### Actividades de modificación

1. Agregar Dropout en la arquitectura MLP:

Respuesta: Para agregar `Dropout`, se pueden insertar capas `nn.Dropout(p=0.5)` entre las capas lineales y las activaciones. Esto hace que durante el entrenamiento se apaguen aleatoriamente algunas neuronas, reduciendo la dependencia entre ellas. El modelo puede tardar un poco más en entrenar, pero debería generalizar mejor si antes había overfitting.

```python
self.net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, num_classes)
)
```

2. Agregar Batch Normalization:

Respuesta: Para agregar `BatchNorm`, se coloca `nn.BatchNorm1d(...)` después de cada capa `Linear` y antes de la activación. Esto normaliza las activaciones antes de aplicar `ReLU`, haciendo que el entrenamiento sea más estable.

```python
self.net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
```

3. Aplicar Weight Decay (L2):

Respuesta: Para aplicar regularización `L2`, se agrega el parámetro `weight_decay` en el optimizador. Esto penaliza pesos muy grandes y ayuda a que el modelo no se ajuste demasiado a los datos de entrenamiento.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

4. Reducir overfitting con data augmentation:

Respuesta: Otra forma de reducir el overfitting es aumentar artificialmente la variabilidad del dataset usando transformaciones sobre las imágenes de entrenamiento. Por ejemplo, se pueden usar giros horizontales, cambios de brillo, contraste o pequeñas rotaciones. Esto obliga al modelo a aprender patrones más generales y no depender de una imagen exacta.

```python
train_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.4
    ),
    A.Normalize(),
    ToTensorV2()
])
```

5. Early Stopping:

Respuesta: `Early Stopping` permite detener el entrenamiento cuando la validación deja de mejorar durante varias épocas. Esto es útil porque evita seguir entrenando cuando el modelo empieza a memorizar los datos de entrenamiento y la performance de validación ya no mejora.

```python
best_val_loss = float("inf")
patience = 5
counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        break
```

### Preguntas prácticas

- ¿Qué efecto tuvo `BatchNorm` en la estabilidad y velocidad del entrenamiento?

Respuesta: `BatchNorm` debería hacer que el entrenamiento sea más estable, con curvas de loss menos irregulares. Además, puede acelerar la convergencia porque cada capa recibe activaciones más normalizadas. En general, el modelo llega más rápido a una zona de menor pérdida.

- ¿Cambió la performance de validación al combinar `BatchNorm` con `Dropout`?

Respuesta: Al combinar `BatchNorm` con `Dropout`, la performance de validación puede mejorar si el modelo tenía overfitting. `BatchNorm` estabiliza el entrenamiento y `Dropout` reduce la dependencia entre neuronas. Sin embargo, si se usa demasiado `Dropout`, también puede empeorar la performance porque el modelo queda demasiado limitado.

- ¿Qué combinación de regularizadores dio mejores resultados en tus pruebas?

Respuesta: La combinación más razonable suele ser usar `BatchNorm`, `Dropout` moderado y `weight_decay`. `BatchNorm` mejora la estabilidad, `Dropout` ayuda a reducir overfitting y `weight_decay` evita pesos demasiado grandes. Si además se agrega data augmentation, el modelo debería generalizar mejor.

- ¿Notaste cambios en la loss de entrenamiento al usar `BatchNorm`?

Respuesta: Sí, al usar `BatchNorm` la loss de entrenamiento suele bajar de forma más estable. Puede haber pequeñas variaciones entre batches, pero en general la curva se vuelve menos errática y el modelo converge mejor. También puede pasar que la loss inicial cambie respecto del modelo sin `BatchNorm`, porque las activaciones internas se están normalizando.

## 8. Inicialización de Parámetros

### Preguntas teóricas

- ¿Por qué es importante la inicialización de los pesos en una red neuronal?

Respuesta: La inicialización de los pesos es importante porque define desde dónde empieza a aprender la red. Si los pesos arrancan con valores muy grandes o muy chicos, los gradientes pueden explotar o desvanecerse. Una buena inicialización permite que el entrenamiento sea más estable y que el modelo converja mejor.

- ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?

Respuesta: Si todos los pesos se inicializan con el mismo valor, las neuronas de una misma capa aprenderían prácticamente lo mismo. Esto rompe la idea de tener muchas neuronas aprendiendo patrones distintos. Por eso se usa inicialización aleatoria, para romper esa simetría inicial.

- ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?

Respuesta: Xavier busca mantener estable la varianza de las activaciones considerando la cantidad de entradas y salidas de cada capa. Es útil en redes con funciones como `Tanh` o `Sigmoid`. He también controla la varianza, pero está pensada especialmente para redes con `ReLU`, porque `ReLU` anula los valores negativos y cambia la distribución de las activaciones.

- ¿Por qué en una red con ReLU suele usarse la inicialización de He?

Respuesta: En una red con `ReLU` suele usarse He porque esta inicialización tiene en cuenta que aproximadamente una parte de las activaciones puede quedar en cero. Entonces ajusta la escala inicial de los pesos para mantener mejor la varianza y evitar que los gradientes se vuelvan demasiado chicos.

- ¿Qué capas de una red requieren inicialización explícita y cuáles no?

Respuesta: Las capas con parámetros entrenables, como `Linear` o `Conv2d`, pueden requerir inicialización explícita porque tienen pesos y bias. En cambio, capas como `ReLU`, `Flatten` o `Dropout` no necesitan inicialización porque no tienen pesos entrenables. `BatchNorm` sí tiene parámetros, aunque normalmente se inicializa con valores estándar adecuados.

### Actividades de modificación

1. Agregar inicialización manual en el modelo:

Respuesta: Para agregar inicialización manual, se puede definir un método `init_weights` dentro de la clase del modelo. En este caso se inicializan las capas `Linear` con He, usando `kaiming_normal_`, y los bias en cero.

```python
def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
```

Luego se llama al método después de definir la arquitectura:

```python
model = MLP(in_features=input_size, num_classes=num_classes)
model.init_weights()
```

2. Probar distintas estrategias de inicialización:

Respuesta: Se pueden probar distintas inicializaciones cambiando la función usada sobre los pesos de las capas `Linear`. Después se comparan las curvas de loss y accuracy para ver cuál converge mejor y cuál tiene mayor estabilidad.

```python
def init_weights_xavier(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
```

```python
def init_weights_he(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
```

```python
def init_weights_uniform(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            nn.init.zeros_(m.bias)
```

3. Visualizar pesos en TensorBoard:

Respuesta: Para visualizar los pesos en TensorBoard, se pueden guardar histogramas de los parámetros del modelo. Esto permite ver cómo están distribuidos los pesos y si aparecen valores demasiado grandes, demasiado chicos o comportamientos raros durante el entrenamiento.

```python
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)
```

### Preguntas prácticas

- ¿Qué diferencias notaste en la convergencia del modelo según la inicialización?

Respuesta: La inicialización puede cambiar bastante la velocidad con la que converge el modelo. Con una inicialización adecuada, como He para redes con `ReLU`, la loss suele bajar de forma más estable. Con una inicialización menos adecuada, el entrenamiento puede avanzar más lento o tener curvas más irregulares.

- ¿Alguna inicialización provocó inestabilidad (pérdida muy alta o NaNs)?

Respuesta: Una inicialización con valores demasiado grandes puede provocar inestabilidad, pérdidas muy altas o incluso valores `NaN`. Esto pasa porque las activaciones y los gradientes pueden crecer demasiado. En cambio, si los valores son demasiado chicos, el modelo puede aprender muy lento porque los gradientes se vuelven muy bajos.

- ¿Qué impacto tiene la inicialización sobre las métricas de validación?

Respuesta: La inicialización puede afectar las métricas de validación porque influye en el punto inicial del entrenamiento. Una buena inicialización ayuda a que el modelo llegue a mejores soluciones y generalice mejor. Sin embargo, no garantiza por sí sola una buena accuracy de validación, porque también dependen el dataset, la arquitectura, el optimizador y la regularización.

- ¿Por qué `bias` se suele inicializar en cero?

Respuesta: El `bias` suele inicializarse en cero porque no genera el mismo problema de simetría que los pesos. Los pesos ya se inicializan de forma aleatoria, entonces las neuronas pueden aprender cosas distintas. Inicializar el bias en cero simplifica el modelo inicial y suele funcionar correctamente.
