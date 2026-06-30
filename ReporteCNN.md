## Ajustes realizados en el modelo CNN

Para mejorar el modelo CNN se reemplazó la arquitectura entrenada desde cero por un enfoque de **transfer learning** usando `ResNet18` preentrenada. Esta decisión se tomó porque el dataset es reducido, con **679 imágenes de entrenamiento**, **165 imágenes de validación** y **9 clases**. En este contexto, entrenar una CNN desde cero suele ser menos efectivo, ya que el modelo tiene pocos datos para aprender buenas representaciones visuales desde cero.

La configuración utilizada fue:

```python
config = {
    "model_name": "resnet18",
    "pretrained": True,
    "freeze_backbone": False,
    "input_size": 224,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "dropout": 0.4,
    "label_smoothing": 0.1,
    "use_class_weights": True,
    "epochs": 30,
    "patience": 6,
    "grad_clip_norm": 5.0,
    "use_tta_final": True,
    "seed": 42,
    "num_workers": 2
}
```

## Cambios principales

El primer cambio importante fue usar **ResNet18 preentrenada**. Esto permitió aprovechar filtros y representaciones visuales ya aprendidas en un dataset grande, en lugar de intentar aprenderlas desde cero con pocas imágenes. Para un problema de clasificación de imágenes con dataset chico, esta suele ser una mejora muy fuerte.

También se aumentó el tamaño de entrada a **224x224**, lo cual permite conservar más detalle visual que con imágenes de menor resolución. Esto es relevante porque, en imágenes dermatológicas, detalles como textura, color, bordes y patrones locales pueden ser importantes para distinguir entre clases similares.

Además, se aumentó la regularización del modelo. Se usó `weight_decay = 5e-4`, `dropout = 0.4` y `label_smoothing = 0.1`. Estos cambios buscan reducir el overfitting, penalizando modelos demasiado ajustados al conjunto de entrenamiento y evitando que la red genere predicciones excesivamente confiadas.

Se mantuvo el uso de **class weights** para compensar posibles desbalances entre clases, y también se utilizó **early stopping** con `patience = 6`, permitiendo que el modelo entrene hasta 30 épocas pero se detenga si la validación deja de mejorar. Finalmente, se usó **TTA** en la evaluación final para hacer predicciones más estables mediante transformaciones simples de las imágenes.

## Resultados obtenidos

El mejor resultado durante el entrenamiento fue:

```text
best_val_acc  = 87.88%
best_val_loss = 0.8545
best_epoch    = 20
```

La evaluación final obtuvo:

```text
final_val_acc  = 86.06%
final_val_loss = 0.8533
macro F1       = 0.87
weighted F1    = 0.86
```

El modelo tuvo muy buen desempeño en varias clases:

```text
Benign keratosis              F1 = 1.00
Tinea Ringworm Candidiasis    F1 = 1.00
Vascular lesion               F1 = 1.00
Dermatofibroma                F1 = 0.95
Atopic Dermatitis             F1 = 0.90
```

Las clases más difíciles fueron:

```text
Melanoma                    F1 = 0.70
Squamous cell carcinoma     F1 = 0.71
Actinic keratosis           F1 = 0.72
```

Esto es esperable, ya que estas clases pueden compartir características visuales similares, como patrones de color, textura o forma, lo que aumenta la probabilidad de confusión entre ellas.

## Interpretación de los resultados

El modelo mejoró claramente respecto de una CNN simple entrenada desde cero. La razón principal es que `ResNet18` preentrenada ya cuenta con representaciones visuales generales útiles, por lo que puede adaptarse mejor al problema aun con pocos datos.

Sin embargo, el modelo todavía muestra señales de **overfitting**. La accuracy de entrenamiento llega cerca del **99%**, mientras que la accuracy de validación se mantiene alrededor del **86% - 88%**. Esto indica que la red tiene capacidad suficiente para memorizar parte del conjunto de entrenamiento, pero no todo ese aprendizaje generaliza al conjunto de validación.

Este comportamiento es razonable por tres motivos principales:

1. **El dataset es chico**
   Con solo 679 imágenes de entrenamiento, una arquitectura como ResNet18 tiene mucha más capacidad que datos disponibles.

2. **Las clases son visualmente parecidas**
   Algunas categorías dermatológicas comparten rasgos visuales, lo que hace más difícil distinguirlas correctamente.

3. **La validación también es reducida**
   Con 165 imágenes de validación, cada imagen representa aproximadamente 0.61 puntos porcentuales de accuracy. Por eso, pequeñas variaciones entre corridas pueden depender de pocas imágenes clasificadas correctamente o incorrectamente.

En conclusión, la mejora más importante fue pasar a **transfer learning con ResNet18 preentrenada**, usando mayor resolución y más regularización. Aunque todavía hay overfitting, el uso de `dropout`, `weight_decay`, `label_smoothing`, `class weights`, `early stopping` y `TTA` permitió obtener un modelo bastante más sólido que la CNN entrenada desde cero.

