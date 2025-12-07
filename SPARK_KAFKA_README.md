# Configuración para Spark + Kafka - Clasificación de Frutas

## 🚀 Sistema de Procesamiento en Tiempo Real

Este sistema utiliza **Apache Spark** y **Apache Kafka** para procesar imágenes de frutas en tiempo real y en lotes usando el modelo MobileNetV2.

## 📋 Prerequisitos

### Dependencias Python
```bash
pip install pyspark kafka-python tf-keras tensorflow pillow numpy
```

### Software Requerido
- **Apache Kafka** (v2.8+)
- **Apache Spark** (v3.5+)  
- **Java** (v8 o v11)

## ⚙️ Configuración de Kafka

### 1. Descargar e Iniciar Kafka
```bash
# Descargar Kafka
wget https://downloads.apache.org/kafka/2.8.2/kafka_2.12-2.8.2.tgz
tar -xzf kafka_2.12-2.8.2.tgz
cd kafka_2.12-2.8.2

# Iniciar Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Iniciar Kafka (en otra terminal)
bin/kafka-server-start.sh config/server.properties
```

### 2. Crear Tópico
```bash
# Crear tópico para imágenes
bin/kafka-topics.sh --create --topic fruit-images --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Verificar tópico
bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

## 🖥️ Uso del Sistema

### 1. Procesamiento en Lotes (Batch)
```bash
# Procesar directorio de imágenes
python spark_consumer.py --mode batch --model FV_Fruits_Only.h5 --input ./images --output ./results

# Ejemplo con directorio específico
python spark_consumer.py --mode batch --model FV_Fruits_Only.h5 --input ./upload_images --output ./batch_results
```

### 2. Procesamiento en Tiempo Real (Streaming)
```bash
# Terminal 1: Iniciar consumer de Spark
python spark_consumer.py --mode streaming --model FV_Fruits_Only.h5 --kafka-servers localhost:9092 --topic fruit-images

# Terminal 2: Enviar imágenes individuales
python kafka_producer.py --mode single --image ./sample_apple.jpg --user-id user123

# Terminal 3: Enviar múltiples imágenes
python kafka_producer.py --mode batch --directory ./test_images --delay 2.0

# Terminal 4: Stream continuo
python kafka_producer.py --mode stream --directory ./test_images --interval 5.0
```

## 📊 Características del Sistema

### Consumer (Spark)
- ✅ **Procesamiento distribuido** con Spark
- ✅ **Batch processing** para grandes volúmenes
- ✅ **Stream processing** en tiempo real
- ✅ **Broadcast variables** para eficiencia
- ✅ **Checkpoint** para recovery
- ✅ **Múltiples formatos** de salida (CSV, JSON)

### Producer (Kafka)
- ✅ **Envío individual** de imágenes
- ✅ **Envío en lotes** con delay configurable
- ✅ **Stream continuo** para testing
- ✅ **Compresión automática** de imágenes grandes
- ✅ **Manejo de errores** robusto

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
export SPARK_HOME=/path/to/spark
export KAFKA_HOME=/path/to/kafka
export JAVA_HOME=/path/to/java
```

### Configuración Spark
```bash
# Para más memoria
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=4g

# Para debugging
export SPARK_LOG_LEVEL=INFO
```

## 📈 Monitoreo

### Spark UI
- Local: http://localhost:4040
- Historico: http://localhost:18080

### Kafka Monitoring
```bash
# Ver mensajes del tópico
bin/kafka-console-consumer.sh --topic fruit-images --bootstrap-server localhost:9092

# Ver métricas del tópico
bin/kafka-topics.sh --describe --topic fruit-images --bootstrap-server localhost:9092
```

## 🏗️ Arquitectura del Sistema

```
[Imágenes] → [Kafka Producer] → [Kafka Topic] → [Spark Consumer] → [Resultados]
    ↓              ↓                   ↓              ↓              ↓
  Files         JSON+Base64       Message Queue   MobileNetV2    CSV/JSON
                                                  Classification
```

## 🧪 Testing

### 1. Crear Datos de Prueba
```bash
mkdir -p test_images
# Copiar algunas imágenes de frutas a test_images/
```

### 2. Test Completo
```bash
# Terminal 1: Consumer
python spark_consumer.py --mode streaming --model FV_Fruits_Only.h5

# Terminal 2: Producer (esperar que consumer esté listo)
python kafka_producer.py --mode batch --directory test_images --delay 3.0
```

## 📁 Estructura de Archivos

```
fruit-classification-mobilenet/
├── spark_consumer.py          # Consumer principal
├── kafka_producer.py          # Producer de imágenes
├── FV_Fruits_Only.h5         # Modelo entrenado
├── App.py                    # Aplicación Streamlit
├── test_images/              # Imágenes de prueba
├── output/                   # Resultados del procesamiento
│   ├── streaming_results/    # Resultados del streaming
│   └── batch_results/        # Resultados del batch
├── checkpoint/               # Checkpoints de Spark
└── logs/                     # Logs del sistema
```

## ⚡ Performance Tips

1. **Increase Spark Parallelism**:
   ```python
   spark.conf.set("spark.sql.shuffle.partitions", "200")
   ```

2. **Optimize Image Processing**:
   - Redimensionar imágenes antes de enviar
   - Usar compresión JPEG para reducir tamaño

3. **Kafka Optimization**:
   - Aumentar `num.partitions` para mayor paralelismo
   - Ajustar `batch.size` y `linger.ms` en producer

## 🐛 Troubleshooting

### Errores Comunes

1. **"Topic no existe"**:
   ```bash
   bin/kafka-topics.sh --create --topic fruit-images --bootstrap-server localhost:9092
   ```

2. **"Modelo no encontrado"**:
   - Verificar ruta en `--model`
   - Verificar que `FV_Fruits_Only.h5` existe

3. **"Spark out of memory"**:
   ```bash
   export SPARK_DRIVER_MEMORY=8g
   export SPARK_EXECUTOR_MEMORY=8g
   ```

4. **"Kafka connection error"**:
   - Verificar que Kafka esté ejecutándose
   - Verificar `--kafka-servers` parameter

## 📞 Soporte

Para problemas o preguntas sobre la implementación, revisar:
- Logs de Spark en `./logs/`
- Spark UI en http://localhost:4040
- Kafka logs en `$KAFKA_HOME/logs/`