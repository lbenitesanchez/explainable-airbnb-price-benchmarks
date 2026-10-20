# Protocolo del reanálisis del primer artículo Airbnb

Protocolo establecido antes de ejecutar los resultados del nuevo pipeline. Fecha de decisión: 18 de octubre de 2026. Los datos originales y los resultados archivados se mantienen intactos. El estudio sigue siendo una revalidación interna de un snapshot histórico, sin nuevas observaciones externas.

## Objetivo y población

Predecir el precio publicado por noche en su escala original. No estimar ingresos, ocupación, demanda, efectos causales ni un precio óptimo. Conservar precios altos salvo evidencia verificable de error. Aplicar las correcciones documentadas de baños, piscina y formato de ocupación. Los baños con texto no interpretable permanecen perdidos, y la imputación se aprende dentro de cada muestra de entrenamiento.

## Diseños de evaluación

Tres diseños, cada uno con tres folds externos y dos repeticiones, usando semillas predefinidas 20261018 y 20261019: KFold aleatorio de anuncios; GroupKFold por anfitrión; GroupKFold por barrio. El agrupamiento por barrio usa las unidades administrativas disponibles y no representa bloques geográficos con buffer. Los modelos comparten exactamente los folds externos e internos dentro de cada diseño. Los diseños responden a destinos distintos y no se elige una semilla por su resultado.

La búsqueda interna usa tres folds, con el mismo criterio de agrupamiento que la evaluación externa. La métrica de selección es MAE del precio original. RMSE y R² son secundarios. La búsqueda no recibe el conjunto externo. Todos los transformadores, incluido TF-IDF/SVD, se ajustan dentro del pipeline y de cada fold interno.

## Bloques de variables

S: capacidad, dormitorios, camas, baños corregidos, formato de ocupación nominal y grupo de propiedad nominal. L: latitud, longitud, seis distancias heredadas a puntos de interés y barrio nominal. R: volumen de reseñas y siete puntuaciones agregadas. A: número de amenidades disponibles, conteos básico, lujo y playa e indicador de piscina corregido. T: título y descripción concatenados, TF-IDF de unigramas y bigramas y cuatro componentes SVD.

La especificación principal contiene S+L+R+A+T. No utiliza sentimiento heredado, cuya construcción no está documentada. H, el tamaño de cartera y el indicador multianuncio observados en todo el snapshot, aparece únicamente como sensibilidad secundaria que agrega H al modelo principal. H usa covariables del inventario completo, es transductivo y no constituye una feature estrictamente ex ante.

Los numéricos se imputan con mediana y un indicador de ausencia cuando corresponde, y se estandarizan. Las variables categóricas se imputan con una categoría explícita y se codifican one-hot ignorando categorías no vistas. Antes de TF-IDF, una expresión regular fija sustituye importes acompañados por S/, PEN, USD, US$, $ o sufijos soles/dólares por un marcador monetario, y retira URLs y correos. Los números sin marcador monetario, como capacidad y dormitorios, se conservan. La regla no consulta el precio objetivo y no garantiza ausencia de referencias de precio no marcadas; se guardan conteos agregados de las máscaras. Los textos originales permanecen en los datos restringidos. El texto se procesa con max_features=1500, min_df=3, max_df=0,95, sublinear_tf=True y SVD de cuatro componentes. La reducción incorpora una salvaguarda de rango para folds pequeños, sin usar datos de validación. Estas decisiones nuevas no buscan replicar exactamente los valores archivados.

## Familias y grillas cerradas

Todas las familias usan la misma representación, con tres configuraciones por familia principal:

- Ridge: alpha 0,1; 10; 1000.
- Elastic Net: alpha 0,01; 0,1; 1, con l1_ratio=0,5, max_iter=20000 y tol=0,00001.
- Gradient Boosting: 100 árboles, profundidad 3, hoja mínima 2; 300 árboles, profundidad 3, hoja mínima 5; 300 árboles, profundidad 5, hoja mínima 5. learning_rate=0,05 y pérdida cuadrática.
- Random Forest: 300 árboles con hoja mínima 2 y max_features=0,7; hoja mínima 5 y max_features=0,7; hoja mínima 5 y max_features=1,0. Profundidad sin límite y min_samples_split=2.

Comparadores secundarios preespecificados: Decision Tree con profundidad 4 u 8 y hoja mínima 5; Bagging de 100 árboles con hoja mínima 3 y max_features=0,7 o 1,0; XGBoost, cuando esté instalado y activado, con 200 árboles, profundidad 3 o 5 y learning_rate=0,05. Todos los estimadores usan semilla 42; los ensembles paralelos usan un hilo internamente. Los comparadores secundarios se informan, pero no participan en la selección de familia para ablaciones ni intervalos. Un predictor constante de mediana de entrenamiento aporta referencia de MAE.

## Ablaciones y selección

En cada fold externo se selecciona una de las cuatro familias principales según MAE interno con la especificación completa. Empates exactos se resuelven por el orden fijo Ridge, Elastic Net, Gradient Boosting y Random Forest. Se repite la búsqueda interna de esa misma familia, retirando por turno S, L, R, A y T. Se retunea también full+H. Los cambios de MAE se calculan sobre idénticas observaciones externas. Esta comparación mide valor predictivo marginal condicionado al resto de señales y a una estrategia de selección; no identifica mecanismos causales ni supone que los bloques contienen información independiente.

## Intervalos secundarios

Cuando se activa esta opción, se reserva el 25% del entrenamiento externo para calibración mediante división aleatoria o por grupos según el diseño. La familia y los hiperparámetros se vuelven a seleccionar entre las cuatro familias principales utilizando exclusivamente el subconjunto de fitting; la representación textual tampoco ve las covariables de calibración durante el ajuste. Se calibra un intervalo simétrico del 80% con el cuantil de residuos absolutos de rango ceil((n_cal+1)*0,80), y se limita el extremo inferior a cero porque el precio objetivo es positivo.

Se calibra además una segunda banda con el score |residuo|/max(predicción, 0,25 por mediana del precio del fitting). La escala mínima se fija en fitting y la calibración solo estima el cuantil del score con el mismo rango finito. Ambas bandas se reportan; ninguna se selecciona usando su cobertura externa.

Se informa cobertura, ancho y métricas de la predicción puntual de este modelo por separado, pues usa menos datos de fitting que los modelos principales. La corrección de rango permite la interpretación split-conformal bajo intercambiabilidad; en diseños agrupados, dependencia o desplazamiento entre barrios/anfitriones no justifican una garantía automática. Se informan cobertura y ancho por formato, banda de reseñas y banda de precio predicho, con tamaños de cada celda y sin elegir el mejor intervalo. Los umbrales de reseñas y precio se aprenden exclusivamente en fitting; las tres bandas operacionales de precio usan sus terciles y se asignan mediante la predicción externa, sin consultar el precio externo observado. Se agrega un diagnóstico retrospectivo por banda de precio observado, con los mismos umbrales de fitting, exclusivamente para describir heteroscedasticidad y sin intervenir en el ajuste o calibración.

## Resultados y límites de inferencia

Se guardan métricas por fold, predicciones externas con identificador estable, índices de particiones externas/internas y de calibración, todas las configuraciones CV y sus resultados, hiperparámetros elegidos, hashes de fuente/datos/configuración y versiones del entorno. Cada observación tendrá una predicción externa por repetición, diseño y método. Se presentan métricas agrupadas de todas las predicciones externas dentro de cada repetición; los promedios de R² de folds no sustituyen esas métricas.

Las diferencias de ablación se reportan por fold y por repetición. No se construyen intervalos convencionales tratando folds o repeticiones solapados como observaciones independientes. La inspección de nuevas semillas demuestra sensibilidad del mismo conjunto histórico, no confirmación en una muestra nueva. El bloque H y el texto heredado limitan la generalización operacional; las distancias continúan siendo variables precalculadas cuya fuente externa requiere documentación.

## Presupuesto y ejecución

Con XGBoost, siete familias, las cinco ablaciones, H e intervalos, el límite previsto es 2916 ajustes de regresión: 162 por fold externo multiplicados por 18 folds. Incluye las refit de cada búsqueda; las transformaciones se cachean. Sin intervalos son 2196 ajustes. Sin XGBoost se reducen 126. El presupuesto excluye transformaciones y cualquier ajuste de un modelo final para despliegue. Dos a cuatro procesos de búsqueda y un hilo por estimador. No se ejecuta automáticamente: el CLI prepara protocolo y configuración, y necesita `--execute` para ajustar. Se guardan checkpoints por fold, y solo se reanudan si coinciden hashes de entrada, código, protocolo y versiones del entorno.

La entrada portable principal es `--prepared-csv data/model_ready_listings_restricted.csv`, que conserva el orden de filas y evita exigir los identificadores directos del CSV original. Su hash se fija antes del ajuste. La alternativa `--source` prepara los originales en un destino nuevo. Se conserva el modelo seleccionado únicamente en el primer fold de la primera repetición de cada diseño, para una explicación SHAP posterior cuya elección no depende de su resultado externo. Las clases de transformación viven en `features.py` para permitir persistencia y carga desde otros procesos.
